from .encoder import Encoder
from .decoder import Decoder

import torch
from torch import nn

from einops import rearrange
from typing import List, Tuple

class VAE(nn.Module):
    def __init__(self,
                 dims:List = [1, 32, 64, 128],
                 latent:int = 10,
                 img_size:Tuple = (1, 28, 28)):
        super().__init__()
        
        self.latent = latent
        self.img_size = img_size
        factor = 2 ** (len(dims) - 1)
        self.decode_input_size = (dims[-1], img_size[1] // factor, img_size[2] // factor)
        
        self.encoder = Encoder(dims, self.latent, self.img_size)
        self.decoder = Decoder(dims[::-1], self.latent, self.decode_input_size)

    def forward(self, x):
        z, mu, std = self.encoder(x)
        x = self.decoder(z)
        return x, mu, std

    def sample(self, num_samples, device):
        z = torch.randn((num_samples, self.latent)).to(device)
        
        samples = self.decoder(z)
        
        return samples