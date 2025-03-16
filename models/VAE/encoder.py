import torch
from torch import nn
from torch.distributions.normal import Normal

from typing import List, Tuple
from einops import rearrange

from .block import Block
from .up_down import DownSample

class Encoder(nn.Module):
    def __init__(self,
                 dims:List = [1, 32, 64],
                 latent:int = 10,
                 img_size:Tuple = (1, 28, 28)):
        super().__init__()
        
        self.dims = dims
        
        in_out = [x for x in zip(dims[:-1], dims[1:])]
        
        self.block_li = nn.ModuleList([])
        
        for idx, (dim_in, dim_out) in enumerate(in_out, start=1):
            self.block_li.append(DownSample(dim_in, dim_out))
            
        self.latent = latent
        
        factor = (2 ** (len(dims)-1))
        self.mu = nn.Linear(dims[-1] * (img_size[1] // factor) * (img_size[2] // factor), self.latent)
        self.log_var = nn.Linear(dims[-1] * (img_size[1] // factor) * (img_size[2] // factor), self.latent)
        
    def forward(self, x):
        for block in self.block_li:
            x = block(x)
        
        # Global Average Pooling
        # x = torch.mean(x, dim=(2, 3))
        x = rearrange(x, 'b c h w -> b (c h w)')
        
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        z = self.reparameterization_trick(mu, log_var)

        return z, mu, log_var
    
    def reparameterization_trick(self, mu, log_var):
        gaussian = Normal(loc=torch.zeros(mu.size()),
                          scale=torch.ones(log_var.size()))
        
        eps = gaussian.sample().to(log_var.device)
        
        std = torch.exp(0.5 * log_var) # (1) 0.5 * log_var -> log_std, (2) torch.exp(log_std) -> std
        
        z = mu + eps * std # Reparameterization
        return z