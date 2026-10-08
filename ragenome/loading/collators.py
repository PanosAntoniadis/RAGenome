from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import geom
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer


def msa_variable_collate_fn(batch):
    """Collate for packed retrieved sequences (variable length across the batch).

    "retrieved_packed" and "retrieved_aligned_pos_packed" are 1-D (total_true_len,) tensors
    (all retrieved species concatenated, gaps removed) — padded with 0 to the batch-max
    total length.
    "retrieved_lengths" is (N,) — padded with 0s to the batch-max N. N varies per item
    because the number of retrieved species depends on how many real tokens each window
    contributes under the token budget.
    "retrieved_taxonomy" is (N, 8) — padded with 0 rows to the batch-max N; the padded rows
    are never gathered since no token belongs to a nonexistent species slot.

    All other keys use default_collate.
    """
    variable_1d = {"retrieved_packed": 0, "retrieved_aligned_pos_packed": 0}
    variable_k = {"retrieved_lengths", "retrieved_taxonomy"}
    skip_keys = variable_1d.keys() | variable_k
    standard_keys = {k for k in batch[0] if k not in skip_keys}

    collated = default_collate([{k: item[k] for k in standard_keys if k in item} for item in batch])

    for key, pad_value in variable_1d.items():
        if key not in batch[0]:
            continue
        tensors = [item[key] for item in batch]  # list of (total_len_i,)
        max_L = max(t.shape[0] for t in tensors)
        padded = [
            t if t.shape[0] == max_L
            else F.pad(t, (0, max_L - t.shape[0]), value=pad_value)
            for t in tensors
        ]
        collated[key] = torch.stack(padded, dim=0)  # (B, max_total_len)

    for key in variable_k:
        if key not in batch[0]:
            continue
        tensors = [item[key] for item in batch]  # list of (K_i,) or (K_i, 8)
        max_K = max(t.shape[0] for t in tensors)
        padded = [
            t if t.shape[0] == max_K
            else F.pad(t, (0, 0, 0, max_K - t.shape[0])) if t.dim() == 2
            else F.pad(t, (0, max_K - t.shape[0]))
            for t in tensors
        ]
        collated[key] = torch.stack(padded, dim=0)  # (B, max_K) or (B, max_K, 8)

    return collated


class DataCollatorForLanguageModelingSpan(DataCollatorForLanguageModeling):
    """Span masking: spans of `min_span`..`max_span` tokens (geometric length distribution)."""

    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer] = None,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        min_span=3,
        max_span=6,
    ):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # Keyword args on purpose: newer transformers versions insert `whole_word_mask` before
        # `mlm_probability`, so a positional call would silently misassign it.
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)

        self.min_span = min_span
        self.max_span = max_span
        rv = geom(0.1)
        probs = np.array(
            [rv.pmf(i) for i in range(1, self.max_span + 2 - self.min_span)]
        )
        probs = probs / sum(probs)
        self.probs = torch.tensor(probs).float()
        values = torch.arange(self.min_span, self.max_span + 1).float()
        self.span_mean = torch.dot(self.probs, values)

        self.non_special_tokens_ids = list(
            set(self.tokenizer.get_vocab().values())
            - set(self.tokenizer.all_special_ids)
        )

    def torch_mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare masked inputs/labels for span-masked language modeling:
        80% of the selected spans are replaced by [MASK], 10% by random tokens, 10% kept.
        `inputs` are already tokenized, shape (1, sequence_length).

        Returns (masked inputs, labels with -100 at non-selected positions, selected-position mask),
        each with the batch dimension removed.
        """
        inputs = inputs.long()
        labels = inputs.clone()

        # probability for each token to be the start of a mask span
        probability_matrix = torch.full(
            labels.shape, self.mlm_probability / self.span_mean
        )
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # avoid sampling spans that start with a special token
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        mask_idx = torch.nonzero(masked_indices)
        if mask_idx.numel() == 0:
            labels[~masked_indices] = -100
            return inputs.squeeze(0), labels.squeeze(0), masked_indices.squeeze(0)
        # sample the length of each span
        span = self.min_span + torch.multinomial(
            self.probs, len(mask_idx), replacement=True
        )
        # A span is kept only if it (1) does not overlap or touch a previously selected span
        # and (2) does not contain a special token.
        selected_span_start = []
        selected_span_end = []
        for (i, j), s in zip(mask_idx, span):
            start = j
            end = min(j + s, masked_indices.shape[1])
            masked_indices[i, start] = False
            if torch.any(
                masked_indices[
                    i, max(start - 1, 0) : min(end + 1, masked_indices.shape[1])
                ]
            ) or torch.any(special_tokens_mask[i, start:end]):
                continue
            masked_indices[i, start:end] = True
            selected_span_start.append(start)
            selected_span_end.append(end)

        labels[~masked_indices] = -100  # loss only on selected positions

        num_spans = len(selected_span_start)
        shuffled_indices = np.random.permutation(num_spans)
        num_mask_spans = int(0.8 * num_spans)
        num_replace_spans = int(0.1 * num_spans)

        indices_to_mask = shuffled_indices[:num_mask_spans]
        for idx in indices_to_mask:
            inputs[0, selected_span_start[idx] : selected_span_end[idx]] = (
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            )

        indices_to_replace = shuffled_indices[
            num_mask_spans : num_mask_spans + num_replace_spans
        ]
        for idx in indices_to_replace:
            random_tokens = torch.from_numpy(
                np.random.choice(
                    self.non_special_tokens_ids,
                    (selected_span_end[idx] - selected_span_start[idx]).tolist(),
                    replace=True,
                )
            )
            inputs[0, selected_span_start[idx] : selected_span_end[idx]] = random_tokens

        return inputs.squeeze(0), labels.squeeze(0), masked_indices.squeeze(0)

    def sample_column_actions(self, length: int) -> torch.Tensor:
        """
        Span selection over an axis of `length` alignment columns, drawn with exactly the same
        procedure as `torch_mask_tokens` (span starts, span lengths, spacing between spans and
        the 80/10/10 split). Used to mask the retrieved sequences: all species of a clade share
        the returned per-column actions.

        Returns a uint8 tensor of shape (length,):
          0 = column not selected, 1 = replace by [MASK], 2 = replace by a random nucleotide,
          3 = selected but kept unchanged.
        """
        unk = self.tokenizer.unk_token_id
        mask_id = self.tokenizer.mask_token_id
        # [UNK] is neither [MASK] nor a possible random replacement (those are non-special ids),
        # so the three outcomes can be read off the masked dummy sequence.
        dummy = torch.full((1, length), unk, dtype=torch.long)
        masked, _, selected = self.torch_mask_tokens(
            dummy, special_tokens_mask=torch.zeros_like(dummy, dtype=torch.bool)
        )
        actions = torch.zeros(length, dtype=torch.uint8)
        actions[selected] = 3
        actions[masked == mask_id] = 1
        actions[selected & (masked != mask_id) & (masked != unk)] = 2
        return actions
