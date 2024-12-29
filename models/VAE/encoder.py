import torch
from torch import nn
from torch.distributions.normal import Normal

class Encoder(nn.Module):
    def __init__(self,
                 latent_size:int=200):
        '''
            latent_size(int): latent variable(z)의 크기
        '''
        
        super().__init__()
        
        self.latent_size = latent_size
        
        self.li1 = nn.Linear(784, self.latent_size)
        self.li2 = nn.Linear(self.latent_size, self.latent_size)
        self.relu = nn.ReLU()
        
        self.mu = nn.Linear(in_features=self.latent_size, out_features=self.latent_size)
        self.log_var = nn.Linear(in_features=self.latent_size, out_features=self.latent_size)
        
        # Gaussian, N(0, 1)
        # loc = mean, scale = sigma
        self.gaussian = Normal(loc=torch.zeros(self.latent_size),
                               scale=torch.ones(self.latent_size))
        
    def forward(self, x):
        x = self.relu(self.li1(x.reshape(x.shape[0], -1)))
        x = self.relu(self.li2(x))
        
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        z, mu, std = self.reparameterization_trick(mu, log_var)

        return z, mu, std
    
    def reparameterization_trick(self, mu, log_var):
        eps = self.gaussian.sample()
        
        std = torch.exp(0.5 * log_var) # (1) 0.5 * log_var -> log_std, (2) torch.exp(log_std) -> std
        
        z = mu + eps * std # Reparameterization
        return z, mu, std