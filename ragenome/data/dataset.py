import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from pysam import FastaFile, faidx
from torch.utils.data import Dataset

from ragenome.loading.collators import DataCollatorForLanguageModelingSpan


class RetrievalDataset(Dataset):
    """
    Samples a window of the query species together with homologous sequences retrieved from a
    whole-genome alignment (WGA).

    Each item contains
      * the span-masked query window (`masked_seq`, `genome_labels`, `weight_matrix`);
      * the retrieved species, gaps removed and concatenated into one array
        (`retrieved_packed`, `retrieved_lengths`), with the original alignment column of every
        retained token (`retrieved_aligned_pos_packed`);
      * the NCBI lineage indices of the query and of every retrieved species.

    Args:
        data_dir: root of the data directory.
        genome: FASTA of the query genome.
        segment_length: length of the alignment window.
        samples_per_epoch: number of samples per epoch.
        sequence_mask_fn: span-masking collator.
        msa_path: zarr store of the WGA.
        clade_path: JSON file with the phylogenetic clades (lists of species indices).
        species_lineages_dir: directory with NCBI lineage of every alignment species.
        training_windows_path: BED file with the candidate training windows.
        species_budget: number of total retrieved tokens.
    """

    def __init__(
        self,
        data_dir: str,
        genome: str,
        segment_length: int,
        samples_per_epoch: int,
        sequence_mask_fn: DataCollatorForLanguageModelingSpan,
        msa_path: str,
        clade_path: str,
        species_lineages_dir: str,
        training_windows_path: str,
        species_budget: int,
    ):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.samples_per_epoch = samples_per_epoch
        self.sequence_mask_fn = sequence_mask_fn
        self.tokenizer = sequence_mask_fn.tokenizer
        self.species_budget = species_budget

        _wins = pd.read_csv(
            training_windows_path, sep="\t", header=None,
            usecols=[0, 1, 2], names=["chrom", "start", "end"],
            dtype={"chrom": str, "start": int, "end": int},
        )
        _wins["chrom"] = _wins["chrom"].str.replace("^chr", "", regex=True)
        # (chrom, window start, window size)
        self.training_windows = list(zip(
            _wins["chrom"], _wins["start"], _wins["end"] - _wins["start"],
        ))

        self.msa_species_lineages, self.taxid_to_idx = self._build_species_lineage_lookup(
            species_lineages_dir
        )

        with open(clade_path, "r") as f:
            clades = json.load(f)["clades"]
        self._species_to_clade = {}
        for clade_id, (_, members) in enumerate(clades.items()):
            for m in members:
                self._species_to_clade[m] = clade_id
        # Species 0 is the query species; its whole clade is excluded from retrieval.
        query_clade = next((members for members in clades.values() if 0 in members), [])
        self.excluded_species = set(query_clade)

        zarr_msa = zarr.open(str(msa_path), mode="r")
        self.msa = {chrom: zarr_msa[chrom] for chrom in zarr_msa.keys()}
        n_species = next(iter(self.msa.values())).shape[1]
        self.msa_pool = [i for i in range(1, n_species) if i not in self.excluded_species]
        self._clade_members: dict = {}
        for sp in self.msa_pool:
            self._clade_members.setdefault(self._species_to_clade.get(sp, -1), []).append(sp)

        self.fasta = self.load_genome(genome)
        # Contig names without the "chr" prefix (as in the alignment) -> names in the FASTA.
        self.contig_names = {name.removeprefix("chr"): name for name in self.fasta.references}

    def _build_species_lineage_lookup(self, species_lineages_dir: str):
        """Build per-species lineage tensors with a dense taxid vocabulary.

        species_lineages_dir contains species.txt (alignment species, in alignment order),
        common_to_scientific.json and species_lineages.json (scientific name -> 8 NCBI taxids:
        species, genus, family, order, class, phylum, kingdom, superkingdom; taxid 1 for a
        missing rank).

        Returns:
            msa_species_lineages: list of LongTensors of shape (8,), indexed by MSA position.
            taxid_to_idx: dict mapping raw taxid (int) -> dense index (0-based).
        """
        d = Path(species_lineages_dir)
        with open(d / "species.txt") as f:
            species_list = [l.strip() for l in f if l.strip()]
        with open(d / "common_to_scientific.json") as f:
            common_to_sci = json.load(f)
        with open(d / "species_lineages.json") as f:
            lineages = json.load(f)

        all_taxids = sorted({int(t) for lin in lineages.values() for t in lin})
        taxid_to_idx = {t: i for i, t in enumerate(all_taxids)}

        msa_species_lineages = []
        for common in species_list:
            sci = common_to_sci[common]
            indices = [taxid_to_idx[int(t)] for t in lineages[sci]]
            msa_species_lineages.append(torch.tensor(indices, dtype=torch.long))

        return msa_species_lineages, taxid_to_idx

    def load_genome(self, genome: str):
        fasta_file_path = self.data_dir / Path(genome)
        index_file_path = fasta_file_path.with_suffix(fasta_file_path.suffix + ".fai")
        if not os.path.exists(index_file_path):
            print(f"Index file {index_file_path} does not exist. Generating it now...")
            faidx(str(fasta_file_path))
        return FastaFile(filename=fasta_file_path, filepath_index=index_file_path)

    def __len__(self):
        return self.samples_per_epoch

    def seq_to_tensor(self, seq):
        return self.tokenizer(
            seq,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors="pt",
        )

    def weight_matrix_np(self, sequence):
        """Loss weights: 1 for uppercase (non-repeat) nucleotides, 0.1 for lowercase and N."""
        arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        is_upper = (arr >= ord("A")) & (arr <= ord("Z"))
        is_N = arr == ord("N")
        weights = np.where(is_upper & ~is_N, 1, 0.1)
        return weights.astype(np.float32)

    def _clade_priority_order(self):
        """
        Randomized clade round-robin priority order over the retrieval pool: round 0 takes
        one (randomly chosen) representative from every clade, round 1 takes a second member
        from every clade that has one, and so on. The clade order and the member order inside
        every clade are freshly shuffled at every call, so both the representatives and how
        deep into each clade the selection reaches vary across windows and epochs.
        """
        clade_ids = list(self._clade_members.keys())
        random.shuffle(clade_ids)
        shuffled = {
            cid: random.sample(members, len(members))
            for cid, members in self._clade_members.items()
        }
        max_round = max(len(m) for m in shuffled.values())
        priority = []
        for r in range(max_round):
            for cid in clade_ids:
                if r < len(shuffled[cid]):
                    priority.append(shuffled[cid][r])
        return priority

    def get_msa(self, chrom: str, start: int, length: int):
        """
        Retrieved species for the window [start, start + length), gaps removed.

        Species are fetched in `_clade_priority_order()` order and kept greedily until the
        cumulative non-gap token count reaches `species_budget`; the rest of the fetched
        block is dropped (the species that crosses the budget is kept in full).

        Returns:
            results: list of N LongTensors, the tokens of every kept species (no gaps)
            aligned_pos_results: list of N LongTensors, alignment column of every kept token
            sampled_indices: list of N MSA species indices
        """
        arr = self.msa[chrom]

        win_start = max(0, start)
        win_end   = min(arr.shape[0], start + length)

        selected = self._clade_priority_order()
        block = arr.oindex[slice(win_start, win_end), np.array(selected, dtype=int)]

        results = []
        aligned_pos_results = []
        running_real = 0
        for species_seq in block.T:
            if running_real >= self.species_budget:
                break
            is_gap = species_seq == b"-"
            seq = bytes(species_seq).decode("ascii").lower()
            tokens = self.seq_to_tensor(seq)["input_ids"].squeeze(0)
            not_gap = ~torch.from_numpy(is_gap)
            col_indices = torch.where(not_gap)[0]  # alignment columns of the kept tokens
            tokens = tokens[not_gap]
            running_real += tokens.shape[0]
            aligned_pos_results.append(col_indices)
            results.append(tokens)

        return results, aligned_pos_results, list(selected[:len(results)])

    def __getitem__(self, _):
        output = {}
        # Random training window (BED), re-centred (or cropped) to `segment_length` bases.
        contig, window_start, window_size = random.choice(self.training_windows)
        start = window_start + window_size // 2 - self.segment_length // 2
        contig_name = self.contig_names[contig]
        chrom_len = self.fasta.get_reference_length(contig_name)
        sequence = self.fasta.fetch(
            contig_name, max(0, start), min(chrom_len, start + self.segment_length)
        )
        # Lineage of the query species: alignment species 0 (human).
        output["taxonomy"] = self.msa_species_lineages[0]

        retrieved_tokens, aligned_pos_list, sampled_indices = self.get_msa(
            contig, start, self.segment_length
        )
        tax_rows = [self.msa_species_lineages[i] for i in sampled_indices]
        output["retrieved_taxonomy"] = (
            torch.stack(tax_rows, dim=0) if tax_rows else torch.zeros((0, 8), dtype=torch.long)
        )  # (N, 8)

        output["weight_matrix"] = self.weight_matrix_np(sequence)
        tokenized_seq = self.seq_to_tensor(sequence)

        # Span-masked query.
        output["masked_seq"], output["genome_labels"], _ = self.sequence_mask_fn.torch_mask_tokens(
            inputs=tokenized_seq["input_ids"]
        )

        true_lengths = [t.shape[0] for t in retrieved_tokens]

        mask_token = self.tokenizer.mask_token_id
        clade_to_slots: dict = {}
        for slot, species_idx in enumerate(sampled_indices):
            clade_id = self._species_to_clade.get(species_idx, -1)
            clade_to_slots.setdefault(clade_id, []).append(slot)
        for clade_id, slots in clade_to_slots.items():
            column_actions = self.sequence_mask_fn.sample_column_actions(self.segment_length)
            for slot in slots:
                actions = column_actions[aligned_pos_list[slot]]  # (true_lengths[slot],)
                tok = retrieved_tokens[slot].clone()
                is_real = tok != 0
                tok[(actions == 1) & is_real] = mask_token
                to_randomize = (actions == 2) & is_real
                n_random = int(to_randomize.sum())
                if n_random:
                    tok[to_randomize] = torch.from_numpy(
                        np.random.choice(
                            self.sequence_mask_fn.non_special_tokens_ids, n_random, replace=True
                        )
                    )
                retrieved_tokens[slot] = tok

        # Concatenate all real tokens — no inter-species padding.
        output["retrieved_packed"] = (
            torch.cat(retrieved_tokens) if retrieved_tokens else torch.zeros(0, dtype=torch.long)
        )  # (total_true_len,)
        output["retrieved_lengths"] = torch.tensor(true_lengths, dtype=torch.long)  # (N,)
        output["retrieved_aligned_pos_packed"] = (
            torch.cat(aligned_pos_list) if aligned_pos_list else torch.zeros(0, dtype=torch.long)
        )  # (total_true_len,)

        return output
