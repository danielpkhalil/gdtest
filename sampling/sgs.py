from .base import Algo
import torch
import tqdm
import numpy as np

class SGS():
    """
        Implementation of split Gibbs sampling for discrete diffusion.
        https://arxiv.org/abs/2405.18782 (continuous version)
    """

    def __init__(self, model, graph, noise, num_steps=200, p = 1, ode_steps=128, eps=1e-5, mh_steps=10, alpha=1, max_dist = 1, device='cuda'):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super().__init__()

        self.uncond_sampler = get_pc_sampler(graph, noise, (1,1024), 'analytic', ode_steps , device=device)
        self.model = model
        self.graph = graph
        self.noise = noise
        self.device = device
        self.num_steps = num_steps
        self.sigma_fn = lambda t: t
        self.get_time_step_fn = lambda r: (1 ** (1 / p) + r * (eps ** (1 / p) - 1 ** (1 / p))) ** p
        steps = torch.linspace(0, 1-eps, num_steps)
        self.time_steps = self.get_time_step_fn(steps)
        self.ode_steps = ode_steps
        self.mh_steps = mh_steps
        self.alpha = alpha
        self.max_dist = max_dist

    def log_ratio(self, sigma, hm_dist):
        
        alpha = (1 - np.exp(-sigma)) * (1 - 1/self.graph.dim)
        # print(sigma, alpha/(1-alpha))
        log_alpha = np.log(alpha+1e-5)
        log_1alpha = np.log(1 - alpha)
        log_ratio = hm_dist * log_alpha + (self.model.length - hm_dist) * log_1alpha
        return log_ratio
    
    def metropolis_hasting(self, x0hat, op, y, sigma, steps):
        x = x0hat.clone()
        dim = self.graph._dim
        N, L = x0hat.shape[0], x0hat.shape[1]
        for _ in range(steps):

            # Get proposal
            
            for _ in range(self.max_dist):
                proposal = x.clone() # proposal, shape = [N, L]
                # for _ in range(self.max_dist):
                idx = torch.randint(L, (N,), device=x.device)
                v = torch.randint(dim, (N,), device=x.device)
                proposal.scatter_(1, idx[:, None], v.unsqueeze(1))

            # Compute log likelihood difference
            log_ratio = op.log_likelihood(proposal,y) - op.log_likelihood(x,y)
            hm_dist_1 = (proposal != x0hat).sum(dim=-1)
            hm_dist = (x != x0hat).sum(dim=-1)
            log_ratio += self.log_ratio(sigma, hm_dist_1) - self.log_ratio(sigma, hm_dist)

            # Metropolis-Hasting step
            rho = torch.clip(torch.exp(log_ratio), max=1.0)
            seed = torch.rand_like(rho)
            x = x * (seed > rho).unsqueeze(-1) + proposal * (seed < rho).unsqueeze(-1)
            
        return x

    def sample(self, x_start=None, batch_size=1, op=None, y=None, verbose=True):

        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        if x_start is None:
            x_start = self.graph.sample_limit(batch_size, self.model.length).to(self.device)
        
        xt = x_start.to(self.device)
        
        for i in pbar:

            # 1. reverse diffusion
            x0hat = self.uncond_sampler(self.model, xt, t_start=self.time_steps[i])
            # 2. langevin dynamics
            # x0y = x0hat
            sigma, _ = self.noise(self.time_steps[i])
            # print(sigma)
            if op is None or y is None:
                x0y = x0hat
            else:
                if i == self.num_steps - 1 and not self.cf:
                    x0y = x0hat
                else:
                    x0y = self.metropolis_hasting(x0hat, op, y, sigma*self.alpha, steps=self.mh_steps)
            xt = x0y

        return xt
    
    # def compile(self):
    #     self.x0hat_traj = torch.stack(self.x0hat)
    #     self.x0y_traj = torch.stack(self.x0y)
    #     self.xt_traj = torch.stack(self.xt)
        
    # def record(self, x0hat, x0y, xt):
    #     self.x0hat.append(x0hat.clone())
    #     self.x0y.append(x0y.clone())
    #     self.xt.append(xt.clone())
