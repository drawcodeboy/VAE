from .encoder import Encoder
from .decoder import Decoder

import torch
from torch import nn

from einops import rearrange
from typing import List

class VAE(nn.Module):
    def __init__(self,
                 dims:List = [1, 32, 64, 128],
                 latent:int = 10):
        super().__init__()
        
        self.encoder = Encoder(dims, latent)
        self.decoder = Decoder(dims[::-1], latent)

    def forward(self, x):
        b, c, w, h = x.size()
        z, mu, std = self.encoder(x)
        x = self.decoder(z, (c, w, h))
        return x, mu, std