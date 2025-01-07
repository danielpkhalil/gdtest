import hydra
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


import transformers

#from evodiff.model import ByteNetLMTime
from evodiff.losses import D3PMCELoss, D3PMLVBLoss
from sequence_models.metrics import MaskedAccuracy
from sequence_models.constants import MSA_ALPHABET

from models.trainer import BaseModel


class ByteNetDiffusion(BaseModel):
    """
    ByteNet convolutional model with time, used in the diffusion model. Inherits from pl.LightningModule.
    """
    #TODO: uses src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
    #outputs loss and any other relevant outputs
    def __init__(self, model_config, network, tokenizer, optimizer, lr_scheduler):
        """
        Initializes the ByteNetDiffusion model.
        config: Hydra config object
        network: ByteNetLMTime model
        tokenizer: Tokenizer object
        optimizer: Optimizer object
        lr_scheduler: Learning rate scheduler object
        device: torch.device
        """
        super().__init__()

        #seems like you don't need to set device here
        #self.device = torch.device(model_config.device)

        if model_config.mask == 'uniform':
            tokenizer = hydra.utils.instantiate(tokenizer, sequences=True)

        self.padding_idx = tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
        self.masking_idx = tokenizer.mask_id

        #network.n_tokens = len(MSA_ALPHABET)
        #network.padding_idx = self.masking_idx

        self.network = hydra.utils.instantiate(network, n_tokens=len(MSA_ALPHABET), padding_idx=self.masking_idx)

        self.opt = hydra.utils.instantiate(optimizer, params=self.network.parameters())

        self.lr_scheduler = None
        if lr_scheduler:
            self.lr_scheduler = hydra.utils.instantiate(lr_scheduler, self.opt)
        
        self.loss_func1 = D3PMLVBLoss(tmax=self.network.embedder.timesteps, tokenizer=tokenizer)
        self.loss_func2 = D3PMCELoss(tokenizer=tokenizer)
        self._lambda = model_config._lambda

        self.accuracy_function = MaskedAccuracy()
    
    def forward(self, batch):
        
        ### Modified from https://github.com/microsoft/evodiff/blob/main/train.py ###
        src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
        q = q.to(self.device)
        Q = Q.to(self.device)
        Q_bar = Q_bar.to(self.device)
        src_onehot = src_onehot.to(self.device)
        tgt_onehot = tgt_onehot.to(self.device)

        timestep = timestep.to(self.device)
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        input_mask = (src != self.padding_idx).float()

        n_tokens = input_mask.sum()

        n_processed = input_mask.sum()
        n_seqs = torch.tensor(len(src), device=self.device)

        outputs = self.network(src, timestep, input_mask=input_mask.unsqueeze(-1))

        lvb_loss = self.loss_func1(src_onehot, q, outputs, tgt, tgt_onehot, input_mask, timestep, Q, Q_bar)
        ce_loss = self.loss_func2(outputs, tgt, input_mask)
        lvb_loss = lvb_loss.to(torch.float32)
        ce_loss = ce_loss.to(torch.float32)

        loss = (lvb_loss + (self._lambda * ce_loss)) * n_tokens
        nll_loss = ce_loss * n_tokens
        accuracy = self.accuracy_function(outputs, tgt, input_mask) * n_tokens

        out = {"loss": loss, "nll_loss": nll_loss, "accuracy": accuracy}

        return out