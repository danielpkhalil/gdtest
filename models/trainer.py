import os
import time
import torch
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

### Modified from https://github.com/ngruver/NOS/blob/main/seq_models/trainer.py ###
class BaseModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.discr_batch_ratio = None
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

    def training_step(self, batch):        
        out = self.forward(batch
            #labels=batch["labels"] if "labels" in batch else None,
        )        

        log_dict = {f"train_{k}" : v for k, v in out.items()}
        self.log_dict(log_dict)  # Don't seem to need rank zero or sync dist

        # if "labels" in batch:
        #     if self.discr_batch_ratio is None:
        #         out["loss"] = out["loss"] + out["regression_mse"]
        #     elif batch_idx % self.discr_batch_ratio == 0:
        #         out["loss"] = out["regression_mse"]
        
        return out["loss"]

    def validation_step(self, batch):
        with torch.no_grad():
            out = self.forward(batch
                #labels=batch["labels"] if "labels" in batch else None,
            )
                
        log_dict = {f"val_{k}" : v for k, v in out.items()}
        self.log_dict(log_dict, rank_zero_only=True)

        return {"val_loss": out['loss']}

    def configure_optimizers(self):
        config = {
            "optimizer": self.opt
        }

        if self.lr_scheduler is not None:
            self.lr_scheduler.step() #avoid lr=0 at start for warmup

            config["lr_scheduler"] = {
                "scheduler": self.lr_scheduler,
                "frequency": 1,
                "interval": "epoch",    # Call after 1 epoch
            }

        return config
    
def get_trainer(config):
    save_path = os.path.join("checkpoints", config.exp_name)
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, filename="best_model", monitor="val_loss", mode="min", save_top_k=1)

    callbacks = [checkpoint_callback]
    wandb_logger = WandbLogger(name=config.exp_name, project=config.wandb_project) #dir=config.exp_name

    accelerator, strategy = "cpu", None
    if config.train.ngpu > 0:
        accelerator = "gpu"
        strategy = "ddp"

    trainer = pl.Trainer(
        default_root_dir=config.exp_name,
        gradient_clip_val=config.train.gradient_clip,
        min_epochs=config.train.min_epochs,
        max_epochs=config.train.max_epochs,
        check_val_every_n_epoch=config.train.val_interval,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=config.train.log_interval,
        accelerator=accelerator,
        strategy=strategy,
        devices=config.train.ngpu,
        enable_progress_bar=True,
    )

    return trainer