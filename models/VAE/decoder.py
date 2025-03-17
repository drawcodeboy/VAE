import torch
from torch import nn

from typing import List, Tuple

from .up_down import UpSample

from einops import rearrange

class Decoder(nn.Module):
    def __init__(self,
                 dims:List = [64, 32, 1],
                 latent:int = 10,
                 decode_input_size:Tuple = (64, 4, 4)):
        super().__init__()
        
        self.dims = dims
        
        self.latent = latent
        
        self.decode_input_size = decode_input_size
        decode_in_dim = 1
        for x in decode_input_size:
            decode_in_dim *= x
        
        self.li = nn.Linear(self.latent, decode_in_dim)
        
        in_out = [x for x in zip(dims[:-1], dims[1:])]                
        self.block_li = nn.ModuleList([])
        
        for idx, (dim_in, dim_out) in enumerate(in_out, start=1):
            if idx < len(in_out):
                self.block_li.append(UpSample(dim_in, dim_out))
            else:
                self.block_li.append(nn.Sequential(UpSample(dim_in, dim_in),
                                                   nn.Conv2d(dim_in, dim_out, 3, 1, 1),
                                                   nn.Sigmoid()))
        
    def forward(self, x):
        x = self.li(x)
        c, w, h = self.decode_input_size
        
        x = rearrange(x, 'b (c w h) -> b c w h', c=c, w=w, h=h)
        
        for block in self.block_li:
            x = block(x)
            
        return x