from transformers import GPT2LMHeadModel, GPT2Config, GPT2LMHeadModel, AdamW
import pytorch_lightning as pl
import logging
import random

from tokenizer import tokenizer
from config import config

LEARNING_RATE = config["learning_rate"]
EPSILON = config["epsilon"]


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, labels, attention_mask, token_type_ids=None):
        # forward pass our model, and get loss
        outputs = self.model(input_ids,
                             labels=labels, 
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_labels = batch[0]
        b_masks = batch[1]

        # calls the forward method
        loss = self(b_input_ids,
                    labels=b_labels, 
                    attention_mask=b_masks,
                    token_type_ids=None)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if config["logging_level"] == "DEBUG" and batch_idx % config["sample_every"] == 0 and batch_idx != 0:
            sample_outputs = self.model.generate(bos_token_id=random.randint(1,30000),
                                                 do_sample=True,   
                                                 top_k=50, 
                                                 max_length=200,
                                                 top_p=0.95, 
                                                 num_return_sequences=1)
            for i, sample_output in enumerate(sample_outputs):
                  logging.debug("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_labels = batch[0]
        b_masks = batch[1]

        loss = self(b_input_ids,
                    attention_mask=b_masks,
                    labels=b_labels)

        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(),
                  lr=LEARNING_RATE,
                  eps=EPSILON
                )
