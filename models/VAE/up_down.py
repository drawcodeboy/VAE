import torch
from torch import nn

from einops.layers.torch import Rearrange

class DownSample(nn.Module):
    def __init__(self, dim_in, dim_out, acti=True):
        super().__init__()
        
        self.down = nn.Conv2d(dim_in, dim_out, 3, 2, 1)
        self.bn = nn.BatchNorm2d(dim_out)

        self.acti = None
        if acti == True:
            self.acti = nn.Tanh()
        else:
            self.acti = nn.Identity()
        
    def forward(self, x):
        return self.acti(self.bn(self.down(x)))

class UpSample(nn.Module):
    def __init__(self, dim_in, dim_out, acti=True):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(dim_in, dim_out, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.conv = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(dim_out)
        
        self.acti = None
        if acti == True:
            self.acti = nn.Tanh()
        else:
            self.acti = nn.Identity()
    
    def forward(self, x):
        x = self.acti(self.bn1(self.up(x)))
        return self.acti(self.bn2(self.conv(x)))