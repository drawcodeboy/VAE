import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 acti:bool = True):
        super().__init__()
        
        self.li1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        if acti == True:
            self.acti = nn.ReLU()
        else:
            self.acti = nn.Identity()
    
    def forward(self, x):
        x = self.acti(self.li1(x))
        
        return x