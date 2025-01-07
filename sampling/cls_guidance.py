from .base import Algo
import torch
import tqdm
import torch.nn.functional as F
import numpy as np

class Classifier_Guidance(Algo):
    '''
        Implementation of Discrete Classifier Guidance with D3PM
        https://arxiv.org/abs/2406.01572
        forward_op is a predictor that outputs log p(y|x_t,t)
    '''
    def __init__(self, net, forward_op, temperature=1, device='cuda'):
        super().__init__(net, forward_op)
        self.device = device
        self.temperature = temperature

    def inference(self, predictor_model, num_samples=1, verbose=False):
        """
        TODO: Enable loading different predictor model each time when running inference.
        """
        timesteps = torch.linspace(self.net.timestep-1,1,int((self.net.timestep-1)/1), dtype=int) # iterate over reverse timesteps
        timesteps = tqdm.tqdm(timesteps) if verbose else timesteps
        x = self.net.get_start(num_samples)
        for t in timesteps:
            x_next_prob = self.net.p_sample(x, t, predictor_model=predictor_model, hard=False)
            # guidance:
            x_ohe = F.one_hot(x, self.net.S).to(torch.float)
            with torch.enable_grad():
                x_ohe.requires_grad = True
                log_probs = self.forward_op(x_ohe, t/self.net.timesteps) / self.temperature
                log_probs.backward()
                grad = x.grad

            log_prob_ratio = grad - (x_ohe * grad).sum(dim=-1, keepdim=True)
            log_prob_ratio = log_prob_ratio.clamp(max=80) # for numerical stability

            x_next_prob = x_next_prob * torch.exp(log_prob_ratio)
            x_next_prob = x_next_prob / x_next_prob.sum(dim=-1, keepdim=True)
            x = torch.multinomial(x_next_prob.view(-1, x_next_prob.shape[-1]), num_samples=1).view(x_next_prob.shape[:-1])
        return x
    
class Classifier_Guidance_Inpaint(Algo):
    '''
        Implementation of Discrete Classifier Guidance with D3PM
        https://arxiv.org/abs/2406.01572
        forward_op is a predictor that outputs log p(y|x_t,t)
    '''
    def __init__(self, net, forward_op, seq_len, full_seq, residues, temperature=1, device='cuda'):
        super().__init__(net, forward_op)
        self.device = device
        self.temperature = temperature
        self.seq_len = seq_len
        self.full_seq = full_seq
        self.residues = residues
        self.mask = torch.zeros(self.seq_len)
        self.mask[self.residues] = 1
        self.mask = self.mask.to(device).int()
        self.full_seq = torch.from_numpy(np.array(self.full_seq)).to(device).int()


    def project(self, x):
        return x * self.mask + self.full_seq * (1 - self.mask)

    def inference(self, num_samples=1, verbose=True):
        timesteps = torch.linspace(self.net.timestep-1,1,int((self.net.timestep-1)/1), dtype=int) # iterate over reverse timesteps
        timesteps = tqdm.tqdm(timesteps) if verbose else timesteps
        x = self.net.get_start(num_samples)
        for t in timesteps:
            x_next_prob = self.net.p_sample(x, t, hard=False)

            # x_next_prob: N x L x S, x: N x L in [0, S-1]
            # x_next_prob[x] -= 1 # prob difference
            x_next_prob[torch.arange(x_next_prob.shape[0])[:, None], torch.arange(x_next_prob.shape[1]), x] -= 1


            # guidance:

            x_next_prob = self.net.get_guided_rates(self.forward_op, x, t, x_next_prob, guide_temp=self.temperature)
            
            # print(q_t.min(), x_next_prob.min())
            # x_next_prob = x_next_prob * (q_t**self.temperature)
            # print(x_next_prob.min())
            x_next_prob[torch.arange(x_next_prob.shape[0])[:, None], torch.arange(x_next_prob.shape[1]), x] += 1
            x_next_prob = x_next_prob.clamp(min=0)
            x_next_prob = x_next_prob / x_next_prob.sum(dim=-1, keepdim=True)
            x = torch.multinomial(x_next_prob.view(-1, x_next_prob.shape[-1]), num_samples=1).view(x_next_prob.shape[:-1])

            # project to subspace
            x = self.project(x)
        return x