import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F

class ELBO(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_prime, x, mu, log_var):
        # 1) Regularization
        first_term = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # 2) Reconstruction
        second_term = F.mse_loss(x_prime, x, reduction='none').sum(dim=(1,2,3)).mean()
        
        minus_elbo = first_term + second_term
        
        return minus_elbo, first_term, second_term