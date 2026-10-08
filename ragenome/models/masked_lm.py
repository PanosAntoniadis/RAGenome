"""
MaskedLM — PyTorch Lightning module for retrieval-augmented MLM pretraining.
"""

import torch
import transformers
from lightning.pytorch import LightningModule


class MaskedLM(LightningModule):
    def __init__(
        self,
        nn: torch.nn.Module,
        mlm_criterion: torch.nn.Module,
        optimizer,
        lr_scheduler=None,
        downweight_lowercase: bool = False,
        init_from: str = None,
    ):
        super().__init__()
        self.nn = nn
        self.mlm_criterion = mlm_criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.downweight_lowercase = downweight_lowercase
        if init_from is not None:
            state_dict = torch.load(init_from, map_location="cpu", weights_only=False)["state_dict"]
            self.load_state_dict(state_dict)

    def _forward(self, batch: dict) -> torch.Tensor:
        return self.nn(
            genome_input_ids=batch["masked_seq"].long(),
            packed_retrieved_ids=batch["retrieved_packed"].long(),
            retrieved_lengths=batch["retrieved_lengths"].long(),
            packed_aligned_pos=batch["retrieved_aligned_pos_packed"],
            retrieved_taxonomy=batch["retrieved_taxonomy"],
            query_taxonomy=batch["taxonomy"],
        )

    def get_loss(self, batch: dict) -> torch.Tensor:
        seq_logits = self._forward(batch)
        B, L = batch["masked_seq"].shape
        V = seq_logits.shape[-1]
        genome_labels = batch["genome_labels"]
        if self.downweight_lowercase and "weight_matrix" in batch:
            return self.mlm_criterion(
                seq_logits.view(B * L, V),
                genome_labels.flatten(),
                batch["weight_matrix"].flatten(),
            )
        return self.mlm_criterion(seq_logits.view(B * L, V), genome_labels.flatten())

    def training_step(self, train_batch):
        loss = self.get_loss(train_batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.trainer.model.parameters())

        if self.lr_scheduler is None:
            return {"optimizer": optimizer}

        assert (
            self.lr_scheduler.func == transformers.get_cosine_schedule_with_warmup
        ), "only the step-based cosine schedule with warmup is supported"
        if self.trainer.max_steps == -1:
            raise ValueError(
                "`get_cosine_schedule_with_warmup` is step-based. "
                "Please set `max_steps > 0` in the Trainer."
            )
        scheduler = self.lr_scheduler(
            optimizer=optimizer, num_training_steps=self.trainer.max_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
