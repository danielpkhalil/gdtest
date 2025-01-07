import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import hydra

from evodiff.utils import Tokenizer
from evodiff.collaters import D3PMCollater
from models.collaters import ContinuousCollater

#from Bio import SeqIO

class SequenceDataset(Dataset):
    """
    Loads protein sequences from a processed csv file with splits.
    """
    def __init__(self, config, split):
        super().__init__()
        self.config = config
        dataset_path = config.data.dataset_path

        df = pd.read_csv(dataset_path)

        df = df[df['Split'] == split]
        self.seqs = df['Sequence'].tolist()
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index) -> str:
        return (self.seqs[index],) #to be consistent with sequence_models.datasets.UniRefDataset from https://github.com/microsoft/protein-sequence-models/blob/main/sequence_models/datasets.py

def get_loaders(config):
    if "d3pm" in config.model.model_config.name:
        # Discrete: timesteps is stored in the 'network' section
        diffusion_timesteps = config.model.network.timesteps
        if config.model.model_config.mask == 'uniform':
            tokenizer = hydra.utils.instantiate(config.model.tokenizer, sequences=True)

    elif "continuous" in config.model.model_config.name:
        # Continuous: timesteps is stored in the noise_schedule section
        diffusion_timesteps = config.model.noise_schedule.timesteps
        # if config.model.model_config.mask == 'uniform':
        tokenizer = hydra.utils.instantiate(config.tokenizer, sequences=True)


    if "d3pm" in config.model.model_config.name:
        if config.model.model_config.mask == 'uniform':
            Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
        # Currently not supported
        # if config.mask == 'blosum':
        #     Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=diffusion_timesteps)
        collater = D3PMCollater(tokenizer=tokenizer, num_timesteps=diffusion_timesteps, Q=Q_t, Q_bar=Q_prod)
    
    #TODO: Daniel fills in this part
    elif "continuous" in config.model.model_config.name:
        collater = ContinuousCollater(tokenizer=tokenizer)

    dsets = [SequenceDataset(config, split) for split in config.data.splits]

    effective_batch_size = config.train.batch_size
    if config.train.ngpu > 0:
        effective_batch_size = int(config.train.batch_size / torch.cuda.device_count())

    loaders = [
        DataLoader(
            dataset=ds,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=config.train.workers,
            #pin_memory=True,
            collate_fn=collater, #this modifies the data associated with sequences
        )
        for ds in dsets
    ]

    return loaders

