from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import logging
import os

from datasets.modules.DataModule import GPTDataModule
from model import Model
from config import config


def main():
    logging.basicConfig(level=config["logging_level"])

    seed_everything(42)

    time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    if config["load_from_checkpoint"]:
        ckpt_path = config["checkpoint_path"]
        if ckpt_path == "" or not os.path.exists(ckpt_path):
            raise ValueError(f"checkpoint path {ckpt_path} does not exist")
        logging.info(f"reloading {ckpt_path}")
        model = Model().load_from_checkpoint(ckpt_path)
    else:
        model = Model()

    wandb_logger = WandbLogger(name=time, project="recipe-gpt") if not config["fast_train"] else None

    datamodule = GPTDataModule(dataset_path=config["dataset"], dataset_type=config["dataset_type"])

    # train on 1% of dataset to make sure it converges
    fast_split = .1 if config["fast_train"] else 1.0

    # automatically save model checkpoints based on min valid_loss
    checkpoint_callback = ModelCheckpoint(monitor="valid_loss",
                                          dirpath=f"checkpoints/{time}",
                                          filename="recipe-gpt-{epoch:02d}-{valid_loss:.2f}",
                                          save_top_k=3,
                                          mode="min")

    trainer = Trainer(gpus=1,
                      limit_train_batches=fast_split,
                      limit_val_batches=fast_split,
                      max_epochs=config["max_epochs"],
                      precision=config["precision"],
                      logger=wandb_logger,
                      callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
