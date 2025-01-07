from .base import Algo
import torch
import tqdm

class Uncond(Algo):
    def __init__(self, net, forward_op, device='cuda'):
        super().__init__(net, forward_op)
        self.device = device

    def inference(self, num_samples=1, verbose=False):
        return self.net.uncond_sample(num_samples, verbose, detokenize=False)

        # Initialize samples
        # x = self.net.get_start(num_samples)
        # eps = 1e-5
        # timesteps = torch.linspace(1, eps, self.num_steps + 1, device=self.device)
        # dt = (1 - eps) / self.num_steps
        # pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        # for i in pbar:
        #     t = timesteps[i] * torch.ones(num_samples, 1, device=self.device)
        #     score = self.net.score(x, )

        # return samples
