import torch
import torch.nn.functional as F


class WeightedCE(torch.nn.Module):
    """Cross-entropy with per-token weights, normalised by the total weight of non-ignored tokens."""

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets, weights=None):
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        weights = torch.ones_like(targets, dtype=logits.dtype) if weights is None else weights.view(-1)

        loss = F.cross_entropy(
            logits, targets, ignore_index=self.ignore_index, reduction="none"
        )

        mask = (targets != self.ignore_index).float()
        loss = loss * weights * mask

        Z = (weights * mask).sum().clamp_min(1e-8)
        return loss.sum() / Z
