from .base import BaseOperator
import torch

class XOR(BaseOperator):
    def __init__(self, ratio, length, **kwargs):
        super().__init__()
        self.length = length
        self.ratio = ratio
        self.pairs = torch.randint(0, length, (2,int(ratio*length)))

    def __call__(self, inputs):
        return inputs[:, self.pairs[0]] ^ inputs[:, self.pairs[1]]
    
    def loss(self, inputs, y):
        # hamming distance
        return ((self(inputs) != y)).flatten(1).sum(-1)
