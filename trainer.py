import pytorch_lightning as pl
from transformers import AdamW
from transformers import GPT2LMHeadModel, GPT2Config, GPT2LMHeadModel
from config import config

LEARNING_RATE = config["learning_rate"]
EPSILON = config["epsilon"]

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

    def forward(self, x):
        b_input_ids = x[0]
        b_labels = x[0]
        b_masks = x[1]

        outputs = self.model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_labels = batch[0]
        b_masks = batch[1]

        outputs = self(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_labels = batch[0]
        b_masks = batch[1]
        
        outputs  = self.model(b_input_ids, 
                            attention_mask = b_masks,
                        labels=b_labels)
        
        loss = outputs[0]
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        return AdamW(self.model.parameters(),
                  lr = LEARNING_RATE,
                  eps = EPSILON
                )

