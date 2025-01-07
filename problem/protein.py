from .base import BaseOperator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinPredictor(nn.Module):
    '''
    Protein predictor class used for classifier guidance 
    '''

    def __init__(self, predictor_config, checkpoint=None, device='cuda'):
        super().__init__()
        '''
        Load predictor model
        '''
        self.device=device
        self.model = MLPModel(predictor_config)
        self.loss_fn = nn.CrossEntropyLoss()
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))
        self.model.to(device)

    def update_model(self, classifier):
        self.model = classifier
        self.model.to(self.device)


    def __call__(self, inputs, t):
        """
        inputs: (B, D)
        t: (B)
        """
        return self.model(inputs, t)
    
    def loss(self, inputs, t, y):
        return self.loss_fn(self.model(inputs, t), y)

### Adapted from https://github.com/HannesStark/dirichlet-flow-matching/tree/main/model and https://github.com/hnisonoff/discrete_guidance/blob/main/applications/enhancer/models.py ###

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class MLPModel(nn.Module):
    """
    MLP model for classifying protein fitness.
    TODO: implement as  a regressor instead and perform classification after?
    """
    def __init__(self, predictor_config):
        super().__init__()
        self.alphabet_size = predictor_config.alphabet_size
        # self.residues = np.ndarray(predictor_config.residues)
        self.residues = np.array(predictor_config.residues)
        self.n_residues = len(self.residues)
        self.hidden_dim = predictor_config.hidden_dim
        
        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embed_dim=self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim))
        self.embedder = nn.Linear(self.alphabet_size, self.hidden_dim) #* self.n_residues

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.cls_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim, 1)) #binary classification

    def forward(self, 
                seq: torch.tensor = None,
                t: torch.tensor = None):
        """
        Args:
            seq: Input sequence with tokens as integers, shape (B, D)
            t: Input time, shape (B)
        """

        # (B,D), (C) -> (B,C)
        #select only the mutated indices of the sequence
        seq = seq[:, self.residues]
        seq_encoded = F.one_hot(seq.long(), num_classes=self.alphabet_size).float()
        #seq_encoded = seq_encoded.reshape(seq_encoded.shape[0], -1) #flatten the onehot encoding
        feat = self.embedder(seq_encoded)
        time_embed = self.time_embedder(t)
        feat = feat + time_embed[:,None,:] #sum the feature and time embeddings
        feat = self.mlp(feat)
        return self.cls_head(feat.mean(dim=1)) #mean across the sequence length
        
