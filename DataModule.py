from dataloader import BATCH_SIZE
import pytorch_lightning as pl
import urllib
import subprocess
from torch.utils.data import random_split, DataLoader, RandomSampler

from FoodDataset import FoodDataset
from TrainDataset import TrainDataset
from tokenizer import tokenizer

BATCH_SIZE = 10

class GPTDataModule(pl.LightningDataModule):
    def __init(self, path, outdir):
        self.path = path
        self.outdir = outdir

    def prepare_data():
        urllib.urlretrieve("https://storage.googleapis.com/recipe-box/recipes_raw.zip")
        subprocess.Popen(["mkdir", "data"])
        subprocess.Popen(["unzip", "recipes_raw.zip", "-d" "data"])
    
    def setup(self, stage=None):
        self.food_data = FoodDataset("data/recipes_raw_nosource_fn.json", maxlen=100)
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
    
