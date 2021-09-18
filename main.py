from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from logging import basicConfig
import logging
from datetime import datetime

from model import Model
from datasets.modules.DataModule import GPTDataModule
from config import config


def main():
    basicConfig(level=config["logging_level"])

    seed_everything(42)

    time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    if config["load_from_checkpoint"]:
        ckpt_path = config["checkpoint_path"]
        logging.info(f"reloading {ckpt_path}")
        model = Model().load_from_checkpoint(ckpt_path)
    else:
        model = Model()

    wandb_logger = WandbLogger(name="Test 2", project="recipe-gpt")

    datamodule = GPTDataModule(dataset_name=config["dataset"])

    # train on 1% of dataset to make sure it converges
    fast_split = .01 if config["fast_train"] else 1.0

    # automatically save model checkpoints based on min valid_loss
    checkpoint_callback = ModelCheckpoint(monitor="valid_loss",
                                          dirpath=f"checkpoints/{time}",
                                          filename="recipe-gpt{epoch:02d}-{valid_loss:.2f}",
                                          save_top_k=3,
                                          mode="min")

    trainer = Trainer(gpus=1,
                      limit_train_batches=fast_split,
                      limit_val_batches=fast_split,
                      logger=wandb_logger,
                      callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
