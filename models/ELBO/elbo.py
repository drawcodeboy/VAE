import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F

class ELBO(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_prime, x, mu, std):
        # 1. Regularization Term, KL(q(z|x)||p(z))
        prior_mu, prior_std = torch.zeros(mu.size()).to(mu.device), torch.ones(std.size()).to(mu.device)
        prior = dist.Normal(prior_mu, prior_std) # p(z)
        variational = dist.Normal(mu, std) # q(z|x)
        first_term = dist.kl_divergence(variational, prior).sum()
        
        # 2. Reconstruction Term
        second_term = F.mse_loss(x_prime, x, reduction='mean') # NLL이랑 같음
        
        
        minus_elbo = first_term + second_term # This is -ELBO to minimize // original goal is Maximize ELBO
        
        return minus_elbo