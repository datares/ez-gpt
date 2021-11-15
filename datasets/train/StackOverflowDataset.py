from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import glob

# dataset can be found at https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate/download

class SODataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = self.ParseData(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return str(self.data.iloc[idx])

    def ParseData(self, path):
        df = pd.read_csv(path) 
        df = df[['Id', 'Title']] 
        print(df.head())

        return df


# data  = SODataset("./recipes-dataset/datares-stack-overflow/train.csv")

# for i in range(10): 
#     print(data[i])
