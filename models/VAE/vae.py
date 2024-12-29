from .encoder import Encoder
from .decoder import Decoder

import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self,
                 latent_size=200,
                 x_size=(1, 28, 28)):
        super().__init__()
        
        self.encoder = Encoder(latent_size=latent_size)
        self.decoder = Decoder(latent_size=latent_size,
                               x_size=x_size)

    def forward(self, x):
        z, mu, std = self.encoder(x)
        x = self.decoder(z)
        return x, mu, std