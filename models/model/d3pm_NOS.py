import os
import time
import tqdm
import wandb
import hydra
import numpy as np

from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import pytorch_lightning as pl

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertPooler,
    BertEncoder,
    BertEmbeddings,
    BertOnlyMLMHead,
)

from models.trainer import BaseModel
from models.util import timestep_embedding

### Modified from https://github.com/ngruver/NOS/blob/main/seq_models/model/mlm_diffusion.py#L222 ###

class MLMTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        dropout=0,
        bert_config_name='bert-base-uncased',
        target_channels=2,
        discr_stop_grad=True,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(bert_config_name)
        config.hidden_dropout_prob = dropout
        config.vocab_size = vocab_size
        # config.hidden_size = 512

        self.target_channels = target_channels
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertOnlyMLMHead(config)

        # self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight

        self.time_embed_dim = config.hidden_size
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # if target_channels > 0:
        #     self.regression_head = RegressionHead(
        #         config, 
        #         target_channels, 
        #         stop_grad=discr_stop_grad
        #     )

    def forward(
        self, 
        corrupted_ids,
        timesteps, 
        attn_mask=None,
        token_embed=None,
    ):
        if token_embed is None:
            token_embed = self.embeddings(input_ids=corrupted_ids)

        time_embed = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        time_embed = time_embed.unsqueeze(1).expand(-1, token_embed.size(1), -1)
        embed = self.dropout(self.LayerNorm(token_embed + time_embed))

        sequence_output = self.encoder(embed, encoder_attention_mask=attn_mask)[0]
        prediction_scores = self.cls(sequence_output)

        out = {
            "logits": prediction_scores,
            "sequence_output": sequence_output,
            "embeds": token_embed,
        }

        return out
    
class MLMDiffusion(BaseModel):
    """
    Transformer with time, used in the diffusion model. Inherits from pl.LightningModule.
    """
    def __init__(
        self,
        network,
        noise_schedule,
        optimizer,
        lr_scheduler,
    ):
        super().__init__()

        self.network = hydra.utils.instantiate(network)
        self.noise_schedule = hydra.utils.instantiate(noise_schedule)
        self.opt = hydra.utils.instantiate(optimizer, params=self.parameters())
        self.lr_scheduler = None
        if lr_scheduler:
            self.lr_scheduler = hydra.utils.instantiate(lr_scheduler, self.opt)

    def forward(
        self,
        input_ids,
        corrupt_mask,
        attn_mask,
        labels=None,
        return_by_timestep=False,
    ):
        
        #TODO: replace input_ids, corrupt_mask, with the batch containing src and timestep used in evodiff
        #src, src_onehot, timestep, tgt, tgt_onehot, Q, Q_bar, q = batch
        #don't need to perform corruption or use noise schedule here because it is already done by the collater?
        #TODO: update output as well if needed

        timesteps = self.noise_schedule.timesteps
        t = torch.randint(
            timesteps, 
            size=(input_ids.shape[0],),
            device=input_ids.device,
            dtype=torch.int64,
        )

        corrupt_ids, corrupt_mask = (
            self.noise_schedule.corrupt(input_ids, t, corrupt_mask)
        )

        model_output = self.network(
            corrupt_ids,
            t, 
            attn_mask,
        )
        logits = model_output['logits']
        hiddens = model_output['sequence_output']
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')  # -100 index = padding token
        nll = loss_fct(logits.view(-1, logits.shape[-1]), input_ids.view(-1))
        nll = nll.view(*input_ids.shape[:2])

        loss_mask = attn_mask * corrupt_mask

        denom = loss_mask.sum(dim=-1)
        denom[denom == 0] = 1

        nll = (nll * loss_mask).sum(dim=-1) / denom
        accuracy = ((logits.argmax(-1) == input_ids) * loss_mask).sum(dim=-1) / denom
        loss = nll.mean()

        out = {}
        out["loss"] = loss.mean()
        out["nll"] = nll.mean()
        out["accuracy"] = accuracy.mean()

        if labels is not None:
            pred_labels = self.network.regression_head(hiddens.detach())
            regression_loss = (pred_labels - labels).pow(2)
            out["regression_mse"] = regression_loss.mean()
            out["regression_spearman"] = spearmanr(
                pred_labels[:,0].detach().cpu().numpy(),  
                labels[:,0].detach().cpu().numpy(),
            ).correlation

        if not return_by_timestep:
            return out
        
        num_buckets = 4
        step_size = timesteps // num_buckets
        for t_lower in np.arange(0, timesteps, step_size):
            t_upper = t_lower + step_size
            t_mask = (t > t_lower) * (t < t_upper)
            
            tag = f"accuracy_{t_lower}-{t_upper}"
            out[tag] = accuracy[t_mask].mean()
            
            if labels is not None:
                tag = f"regression_mse_{t_lower}-{t_upper}"
                out[tag] = regression_loss[t_mask].mean()

        return out
