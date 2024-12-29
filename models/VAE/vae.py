from .encoder import Encoder
from .decoder import Decoder

import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self,
                 latent_size=10,
                 enc_conv_in_channels=1,
                 enc_conv_out_channels=16,
                 x_size=(1, 28, 28)):
        super().__init__()
        
        self.encoder = Encoder(latent_size=latent_size,
                               in_channels=enc_conv_in_channels,
                               out_channels=enc_conv_out_channels)
        self.decoder = Decoder(latent_size=latent_size,
                               x_size=x_size)

    def forward(self, x):
        z, mu, std = self.encoder(x)
        x = self.decoder(z)
        return x, mu, std