from .base import DiscreteData
import pandas as pd
import torch
import numpy as np
class ProteinPredictorDataset(DiscreteData):
    def __init__(self, data_config, tokenizer):
        self.df = pd.read_csv(data_config.data_path)
        self.full_seq = data_config.full_seq
        self.seq_len = data_config.seq_len
        self.alphabet_size = data_config.alphabet_size
        self.residues = data_config.residues
        self.n_residues = len(self.residues)

        #TODO: make this or use the one in EvoDiff
        #or could just use a dictionary
        # self.tokenizer = data_config.tokenizer
        self.tokenizer = tokenizer

        #select n_samples from the dataset
        self.df = self.df.sample(n=data_config.n_samples, random_state=data_config.seed)
        self.df["full_sequence"] = self.df['Combo'].apply(self.get_full_sequence)
        # self.x = self.df['full_sequence'].apply(self.tokenizer.tokenize).tolist()
        # get sequence:
        self.x = []
        for seq in self.df["full_sequence"]:
            data = [self.tokenizer.tokenize(s) for s in seq]
            self.x.append(torch.from_numpy(np.array(data)))
        self.x = torch.stack(self.x).squeeze(-1)

        self.y = torch.tensor(self.df['fitness'].values).float()

        self.mask = torch.zeros(self.seq_len)
        self.mask[self.residues] = 1
        self.full_seq = self.x[0]
        self.mapping = 'ACDEFGHIKLMNPQRSTVWYBZXJOU'
    
    def project(self, x):
        return x * self.mask.to(x.device) + self.full_seq.to(x.device) * (1 - self.mask).to(x.device)

    def get_full_sequence(self, seq):
        """
        Generate the full mutated sequence from the "combo" of only the mutated positions.
        """
        full_seq = list(self.full_seq)
        for i, res in enumerate(seq):
            full_seq[self.residues[i]] = res #zero indexing
        return ''.join(full_seq)
    
    def get_length(self):
        return self.n_residues

    def get_dim(self):
        return self.alphabet_size

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def update_data(self, x):
        self.x = x
        residues = x[:,self.residues] # (B, #residues) This is Combo
        residues = residues.cpu().numpy()
        residues = [[self.mapping[i] for i in r] for r in residues]
        residues = [''.join(r) for r in residues]
        fitness = []
        for r in residues:
            if r not in self.df['Combo'].values:
                v = 0
            else:
                v = self.df[self.df['Combo'] == r]['fitness'].values.item()
            fitness.append(v)
        self.y = torch.from_numpy(np.array(fitness)).float()
