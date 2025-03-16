import torch
from torch import nn

from einops.layers.torch import Rearrange

class DownSample(nn.Module):
    def __init__(self, dim_in, dim_out, acti=True, down:dict={}):
        super().__init__()
        
        self.conv = nn.Conv2d(dim_in, dim_out, 
                              down['conv1']['kernel'], 
                              down['conv1']['stride'], 
                              down['conv1']['padding'], bias=False)
        self.bn1 = nn.BatchNorm2d(dim_out)
        
        self.down = nn.Conv2d(dim_out, dim_out, 
                              down['conv2']['kernel'],
                              down['conv2']['stride'], 
                              down['conv2']['padding'], bias=False)
        self.bn2 = nn.BatchNorm2d(dim_out)

        self.acti = None
        if acti == True:
            self.acti = nn.SiLU()
        else:
            self.acti = nn.Identity()
        
    def forward(self, x):
        x = self.acti(self.bn1(self.conv(x)))
        return self.acti(self.bn2(self.down(x)))

class UpSample(nn.Module):
    def __init__(self, dim_in, dim_out, acti=True, up:dict = {}):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv = nn.Conv2d(dim_in, dim_out,
                              up['conv2']['kernel'],
                              up['conv2']['stride'], 
                              up['conv2']['padding'], bias=False)
        self.bn2 = nn.BatchNorm2d(dim_out)
        
        self.acti = None
        if acti == True:
            self.acti = nn.SiLU()
        else:
            self.acti = nn.Identity()
    
    def forward(self, x):
        x = self.up(x)
        x = self.acti(self.bn2(self.conv(x)))
        return x