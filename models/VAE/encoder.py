import torch
from torch import nn
from torch.distributions.normal import Normal

class Encoder(nn.Module):
    def __init__(self,
                 latent_size:int=10,
                 in_channels:int=1,
                 out_channels:int=16):
        '''
            latent_size(int): latent variable(z)의 크기
        '''
        
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3)
        self.relu = nn.ReLU()
        
        self.latent_size = latent_size
        
        self.mu = nn.Linear(in_features=out_channels, out_features=self.latent_size)
        self.log_var = nn.Linear(in_features=out_channels, out_features=self.latent_size)
        
        # Gaussian, N(0, 1)
        # loc = mean, scale = sigma
        self.gaussian = Normal(loc=torch.zeros(self.latent_size),
                               scale=torch.ones(self.latent_size))
        
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.flatten(2).mean(2) # Global Average Pooling, (B, C, W, H) -> (B, C)
        
        mu = self.mu(x) # (B, C) -> (B, latent_size)
        log_var = self.log_var(x) # (B, C) -> (B, latent_size)
        
        z, mu, std = self.reparameterization_trick(mu, log_var)
        
        return z, mu, std
    
    def reparameterization_trick(self, mu, log_var):
        eps = self.gaussian.sample()
        
        std = torch.exp(0.5 * log_var) # (1) 0.5 * log_var -> log_std, (2) torch.exp(log_std) -> std
        
        z = mu + eps * std # Reparameterization
        return z, mu, std