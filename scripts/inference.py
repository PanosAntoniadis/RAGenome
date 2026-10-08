"""
Run pretrained RAGenome on a window of the human genome with retrieval from the whole-genome alignment.

The window is centred on --pos. A few query nucleotides are masked and the model predicts them from the
unmasked query context and the retrieved homologous sequences.

    python scripts/inference.py --chrom 1 --pos 1000000 \
        --data_dir $RAGENOME_DATA_DIR --device cuda
"""

import argparse
import os
import random
import tempfile

import numpy as np
import torch
from transformers import AutoModel

from ragenome.data.dataset import RetrievalDataset
from ragenome.loading.collators import DataCollatorForLanguageModelingSpan

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chrom", default="1", help="chromosome")
    p.add_argument("--pos", type=int, default=1_000_000, help="0-based centre of the window")
    p.add_argument("--length", type=int, default=13_312, help="query length L")
    p.add_argument("--budget", type=int, default=80_000, help="retrieval budget B")
    p.add_argument("--n_masked", type=int, default=6, help="number of central query nucleotides to mask")
    p.add_argument("--data_dir", default=os.environ.get("RAGENOME_DATA_DIR"))
    p.add_argument("--model", default="pantoniadis/RAGenome")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.chrom = args.chrom.removeprefix("chr")
    random.seed(args.seed)

    # The dataset class provides everything needed to build the retrieval inputs from the alignment.
    # It only needs a BED file to know the training windows, which are not used here.
    with tempfile.NamedTemporaryFile("w", suffix=".bed") as bed:
        bed.write(f"{args.chrom}\t0\t{args.length}\n")
        bed.flush()
        ds = RetrievalDataset(
            data_dir=args.data_dir,
            genome="genomes/hg38_softmasked.fa",
            segment_length=args.length,
            samples_per_epoch=1,
            sequence_mask_fn=DataCollatorForLanguageModelingSpan(
                tokenizer=os.path.join(REPO_ROOT, "ragenome", "tokenizers")
            ),
            msa_path=os.path.join(args.data_dir, "msa", "99.zarr"),
            clade_path=os.path.join(REPO_ROOT, "metadata", "clades.json"),
            species_lineages_dir=os.path.join(REPO_ROOT, "metadata", "species_lineages"),
            training_windows_path=bed.name,
            species_budget=args.budget,
        )

    # Query window (human), with a few central nucleotides masked.
    contig = ds.contig_names[args.chrom]
    start = min(max(0, args.pos - args.length // 2), ds.fasta.get_reference_length(contig) - args.length)
    query = ds.fasta.fetch(contig, start, start + args.length)
    true_ids = ds.seq_to_tensor(query)["input_ids"]  # (1, L)
    masked_ids = true_ids.clone()
    m0 = args.length // 2 - args.n_masked // 2
    masked_slice = slice(m0, m0 + args.n_masked)
    masked_ids[:, masked_slice] = ds.tokenizer.mask_token_id

    # Retrieved homologous sequences: gaps removed, packed into one array, with the alignment column of
    # every token as its position.
    tokens, aligned_pos, species = ds.get_msa(args.chrom, start, args.length)
    lengths = torch.tensor([[t.shape[0] for t in tokens]])  # (1, N)
    print(f"retrieved {len(species)} species, {int(lengths.sum())} tokens")

    inputs = dict(
        genome_input_ids=masked_ids,
        packed_retrieved_ids=torch.cat(tokens)[None],
        retrieved_lengths=lengths,
        packed_aligned_pos=torch.cat(aligned_pos)[None],
        retrieved_taxonomy=torch.stack([ds.msa_species_lineages[i] for i in species])[None],  # (1, N, 8)
        query_taxonomy=ds.msa_species_lineages[0][None],  # human, (1, 8)
    )

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).bfloat16().eval()
    with torch.no_grad():
        logits = model(**{k: v.to(args.device) for k, v in inputs.items()}).logits  # (1, L, 11)

    pred_ids = logits[0, masked_slice].argmax(-1).cpu()
    decode = lambda ids: "".join(ds.tokenizer.convert_ids_to_tokens(ids.tolist())).upper()
    print(f"masked positions {args.chrom}:{start + m0}-{start + m0 + args.n_masked}")
    print(f"true:      {decode(true_ids[0, masked_slice])}")
    print(f"predicted: {decode(pred_ids)}")


if __name__ == "__main__":
    main()
