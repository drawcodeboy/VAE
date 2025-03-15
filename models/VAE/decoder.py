import torch
from torch import nn

from typing import List

from .block import Block

from einops import rearrange

class Decoder(nn.Module):
    def __init__(self,
                 dims:List = [64, 32, 1],
                 latent:int = 10):
        super().__init__()
        
        self.latent = latent
        print(self.latent)
        self.li = nn.Linear(self.latent, dims[0])
        
        in_out = [x for x in zip(dims[:-1], dims[1:])]
                
        self.block_li = nn.ModuleList([])
        
        for idx, (dim_in, dim_out) in enumerate(in_out, start=1):
            self.block_li.append(Block(dim_in, dim_out, acti=False if idx == len(in_out) else True))
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, img_size):
        x = self.li(x)
        
        for block in self.block_li:
            x = block(x)
            
        x = self.sigmoid(x) # p(x|z)
        
        c, w, h = img_size
        
        x = rearrange(x, 'b (c w h) -> b c w h', c=c, w=w, h=h)
    
        return x