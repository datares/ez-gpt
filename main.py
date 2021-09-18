from pytorch_lightning import Trainer, seed_everything

from trainer import Model
from DataModule import GPTDataModule


def main():
    seed_everything(42)
    model = Model()

    datamodule = GPTDataModule()

    trainer = Trainer(gpus=1)
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
