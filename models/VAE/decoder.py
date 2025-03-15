import torch
from torch import nn

from typing import List

from .block import Block
from .up_down import UpSample

class Decoder(nn.Module):
    def __init__(self,
                 dims:List = [128, 64, 32, 1]):
        super().__init__()
        
        in_out = [x for x in zip(dims[:-1], dims[1:])]
                
        self.block_li = nn.ModuleList([])
        
        for dim_in, dim_out in in_out:
            self.block_li.append(nn.Sequential(UpSample(dim_in, dim_in),
                                               Block(dim_in, dim_out)))
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        for block in self.block_li:
            x = block(x)
            
        x = self.sigmoid(x) # p(x|z)
    
        return x