from .base import DiscreteDiffusion
from SEDD.load_model import load_model

import torch
import torch.nn.functional as F


def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    
class SEDD(DiscreteDiffusion):
    def __init__(self, model_path, device='cuda'):
        model, graph, noise = load_model(model_path, device)
        self.model = model
        self.graph = graph
        self.noise = noise
        self.device = device

    def score(self, x, t):
        sigma, dsigma = self.noise(t)
        return self.model(x, sigma)
    
    def pred_mean(self, x, t):
        # This could be bad since score is modeled independently for each dimension
        # according to the paper (Theorem 4.1).
        sigma, dsigma = self.noise(t)
        score = self.model(x, t)

        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        return sample_categorical(probs)

    def get_start(self, batch_size):
        return self.graph.sample_limit((batch_size,)).to(self.device)
    
