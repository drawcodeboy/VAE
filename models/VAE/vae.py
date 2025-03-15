from .encoder import Encoder
from .decoder import Decoder

import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self,
                 dims=[1, 32, 64, 128]):
        super().__init__()
        
        self.encoder = Encoder(dims)
        self.decoder = Decoder(dims[::-1])

    def forward(self, x):
        z, mu, std = self.encoder(x)
        x = self.decoder(z)
        return x, mu, std