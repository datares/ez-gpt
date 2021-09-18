from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from logging import basicConfig

from model import Model
from datasets.modules.DataModule import GPTDataModule
from config import config


def main():
    basicConfig(level=config["logging_level"])

    seed_everything(42)
    model = Model()
    wandb_logger = WandbLogger(name="Run 1", project="recipe-gpt")

    datamodule = GPTDataModule(dataset_name=config["dataset"])

    # train on 1% of dataset to make sure it converges
    fast_split = .01 if config["fast_train"] else 1.0

    trainer = Trainer(gpus=1,
                      limit_train_batches=fast_split,
                      limit_val_batches=fast_split,
                      logger=wandb_logger)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
