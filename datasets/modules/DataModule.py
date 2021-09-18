import pytorch_lightning as pl
import urllib
import subprocess
from torch.utils.data import random_split, DataLoader, RandomSampler

from datasets.train.FoodDataset import FoodDataset
from datasets.GPTDataset import GPTDataset
from tokenizer import tokenizer
from config import config

BATCH_SIZE = config["batch_size"]


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name

    def setup(self, stage=None):
        self.food_data = FoodDataset(f"{self.dataset_name}")
        dataset = GPTDataset(self.food_data, tokenizer, max_length=768)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=8,
            sampler = RandomSampler(self.train_dataset), # Select batches randomly
            batch_size = BATCH_SIZE
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=8,
            batch_size = BATCH_SIZE
        )
