import torch
from torch import nn
from torch.distributions.normal import Normal

from typing import List

from .block import Block
from .up_down import DownSample

class Encoder(nn.Module):
    def __init__(self,
                 dims:List = [1, 32, 64, 128]):
        super().__init__()
        
        in_out = [x for x in zip(dims[:-1], dims[1:])]
        
        self.block_li = nn.ModuleList([])
        
        for dim_in, dim_out in in_out:
            self.block_li.append(nn.Sequential(Block(dim_in, dim_out),
                                               DownSample(dim_out, dim_out)))
            
        self.latent_size = dims[-1]
        
        self.mu = nn.Conv2d(self.latent_size, self.latent_size, 3, 1, 1)
        self.log_var = nn.Conv2d(self.latent_size, self.latent_size, 3, 1, 1)
        
    def forward(self, x):
        for block in self.block_li:
            x = block(x)
        
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        z, mu, std = self.reparameterization_trick(mu, log_var)

        return z, mu, std
    
    def reparameterization_trick(self, mu, log_var):
        gaussian = Normal(loc=torch.zeros(mu.size()),
                          scale=torch.ones(log_var.size()))
        
        eps = gaussian.sample().to(log_var.device)
        
        std = torch.exp(0.5 * log_var) # (1) 0.5 * log_var -> log_std, (2) torch.exp(log_std) -> std
        
        z = mu + eps * std # Reparameterization
        return z, mu, std