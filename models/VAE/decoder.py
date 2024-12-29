import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self,
                 latent_size:int=10,
                 x_size:tuple=(1, 28, 28)):
        super().__init__()
        
        self.latent_size = latent_size
        
        # image size에 맞는 vector 크기로 Linear transform하고
        # forward에서 image size로 reshape
        self.output_size = 1
        for size in x_size:
            self.output_size *= size
        self.x_size = x_size
        
        self.li = nn.Linear(self.latent_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        x = self.li(z).reshape(-1, *self.x_size)
        x = self.sigmoid(x) # p(x|z)
    
        return x