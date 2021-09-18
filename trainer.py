import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

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
                  lr = learning_rate,
                  eps = epsilon
                )

