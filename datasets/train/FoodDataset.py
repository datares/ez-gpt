from torch.utils.data import Dataset
import json
import glob


class FoodDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.recipies = self.iter_over_json()

    def __len__(self):
        return len(self.recipies)

    def __getitem__(self, idx):
        return self.recipies[idx]

    def read_json(self):
        files = glob.glob(f"{self.path}")
        data = {}
        for file in files:
            with open(file) as f:
                recipes = json.load(f)
                # merge the dicts
                data = {**data, **recipes}
        return data

    def iter_over_json(self):
        data = self.read_json()
        recipie_strings = []
        for idx, key in enumerate(data):
            try: 
                text = ""
                recipie = data[key]
                text += recipie["title"]
                text += " "
                for el in recipie["ingredients"]:
                    text += el.replace("ADVERTISEMENT", "")
                    text += " "
                text += recipie["instructions"].strip()
                recipie_strings.append(text)
            except Exception:
                pass
        return recipie_strings
