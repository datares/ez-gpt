import pytorch_lightning as pl
import urllib
import subprocess
from torch.utils.data import random_split, DataLoader, RandomSampler

from FoodDataset import FoodDataset
from TrainDataset import TrainDataset
from tokenizer import tokenizer
from config import config

BATCH_SIZE = config["batch_size"]


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, outdir):
        super().__init__()
        self.dataset_name = dataset_name
        self.outdir = outdir

    def prepare_data(self):
        urllib.request.urlretrieve("https://storage.googleapis.com/recipe-box/recipes_raw.zip")
        subprocess.run(["mkdir", self.outdir])
        subprocess.run(["unzip", "recipes_raw.zip", "-d", self.outdir])
    
    def setup(self, stage=None):
        self.food_data = FoodDataset(f"{self.outdir}/{self.dataset_name}", maxlen=100)
        dataset = TrainDataset(self.food_data, tokenizer, max_length=768)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # The training samples.
            sampler = RandomSampler(self.train_dataset), # Select batches randomly
            batch_size = BATCH_SIZE # Trains with this batch size.
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.val_dataset,  # The training samples.
            sampler = RandomSampler(self.val_dataset), # Select batches randomly
            batch_size = BATCH_SIZE # Trains with this batch size.
        )
