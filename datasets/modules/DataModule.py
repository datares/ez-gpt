from torch.utils.data import random_split, DataLoader, RandomSampler
import pytorch_lightning as pl
import subprocess
import urllib

from datasets.train.ShakespeareDataset import ShakespeareDataset
from datasets.train.StackOverflowDataset import SODataset
from datasets.train.DrakeDataset import DrakeDataset
from datasets.train.FoodDataset import FoodDataset
from datasets.GPTDataset import GPTDataset
from tokenizer import tokenizer
from config import config

BATCH_SIZE = config["batch_size"]


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, dataset_type):
        super().__init__()
        self.dataset_type = dataset_type 
        self.dataset_path = dataset_path

    def setup(self, stage=None):
        if self.dataset_type == "stack_overflow":
            self.data = SODataset(self.dataset_path)
        elif self.dataset_type == "food": 
            self.data = FoodDataset(self.dataset_path)
        elif self.dataset_type == "shakespeare":
            self.data = ShakespeareDataset(self.dataset_path)
        elif self.dataset_type == "drake":
            self.data = DrakeDataset(self.dataset_path)
        else:
            raise KeyError(f"dataset {self.dataset_type} specified in the config.json is not allowed")

        dataset = GPTDataset(self.data, tokenizer, max_length=768)
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
