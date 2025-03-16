import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F

class ELBO(nn.Module):
    def __init__(self,
                 alpha:float = 0.01):
        self.alpha = alpha
        super().__init__()
    
    def forward(self, x_prime, x, mu, log_var):
        # Should Sum(intra sample) and Mean(inter sample)!!!!
        # If you set the reduction of each loss to 'mean' without doing this, 
        # it generates strange samples where the numbers all overlap.
        
        bz, c, w, h = x.shape
        
        # 1. Regularization Term, KL(q(z|x)||p(z))
        prior_mu, prior_std = torch.zeros_like(mu).to(mu.device), torch.ones_like(log_var).to(mu.device)
        prior = dist.Normal(prior_mu, prior_std) # p(z)
        
        variational = dist.Normal(mu, torch.exp(0.5 * log_var)) # q(z|x)

        first_term = dist.kl_divergence(variational, prior).sum(dim=1).mean() # first mean inner each sample, second mean inter sample
        
        
        # 2. Reconstruction Term
        second_term = self.alpha * F.mse_loss(x_prime, x, reduction='none').sum(dim=(1,2,3)).mean()
        
        # print(f"First term: {first_term.item()}, Second term: {second_term.item()}")
        
        minus_elbo = first_term + second_term # This is -ELBO to minimize // original goal is Maximize ELBO
        
        return minus_elbo, first_term, second_term