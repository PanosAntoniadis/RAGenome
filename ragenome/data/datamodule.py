from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ragenome.data.dataset import RetrievalDataset


class RetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_size: int,
        num_workers: int,
        sequence_mask_fn,
        pad_collator,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sequence_mask_fn = sequence_mask_fn
        self.pad_fn = pad_collator

    def setup(self, stage=None):
        self.train_dataset = RetrievalDataset(**self.dataset, sequence_mask_fn=self.sequence_mask_fn)
        print("Train dataset length:", len(self.train_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.pad_fn,
        )
