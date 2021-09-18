from transformers import GPT2LMHeadModel, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from TrainDataset import TrainDataset
from FoodDataset import FoodDataset
from random import random_split
from torch.utils.data import random_split, RandomSampler, SequentialSampler, DataLoader
from tokenizer import tokenizer
BATCH_SIZE = 2


food_data = FoodDataset("data/recipes_raw_nosource_fn.json")
dataset = TrainDataset(food_data, tokenizer, max_length=768)


train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = BATCH_SIZE # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = BATCH_SIZE # Evaluate with this batch size.
        )

