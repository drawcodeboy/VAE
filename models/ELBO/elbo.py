import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F

class ELBO(nn.Module):
    r'''
        Evidence Lower Bound
        - Evidence를 실제로 구하는 것은 어렵다. -> 고차원 벡터에 대한 적분 문제
        - Evidence의 Likelihood를 maximize하여, 근사한 분포를 찾는 것을 목적으로 한다.
        - 그래서, 수식을 다르게 전개하여 ELBO + KL을 얻는데, 이 때 KL은 구하지 못 한다.
        - 왜냐하면, 해당 KL에는 True Posterior(p(z|x))이 있어서 구하지 못 한다.
        - 하지만, KL이 항상 0이상의 값을 가진다는 것은 알기 때문에 이 점을 고려하여
        - ELBO를 최대화시키는 것을 목적으로 한다.
        
        (1st Term, Regularization Term)
        이 ELBO는 -KL(q(z|x)||p(z)) + E[log p(x|z)]이며, 첫 번째 term은 
        Approximate Posterior(q(z|x))를 update시켜 Prior(p(z))와 유사하게 만들겠다는 의미이다.
        Prior과 유사하게 만드는 이유는 Posterior이 특정 데이터 x에 과하게 의존하지 않도록 하기 위함이다.
        그래서, Regularization term이라고도 불린다.
        
        (2nd Term, Reconstruction Term)
        두 번째 term은 E[log p(x|z)]로 이 Expectation을 실제로 계산할 수는 없다.
        왜냐하면, z에 대한 기댓값이라 z에 대해 적분을 해야 한다는 의미인데, 이는 Evidence를 못 구하는 것과
        같이 기댓값을 구하면 안 된다. 그래서, Monte-Carlo estimation을 통해 기댓값(가중 평균)이 아닌 평균으로
        계산을 한다. 이는 큰 수의 법칙에 의해 그렇게 정리가 가능하다.
        -(1/N)\sum^{N}(log p(x|z))
        그렇다면, 이 term이 의미하는 바는 실제 데이터가 주어졌을 때, 해당 확률 분포일 확률을 의미하는
        log-likelihood가 되고, 이를 최대화시킨다는 것은 NLL을 구한다는 것과 같다.
        그래서, 실제 데이터와 예측 데이터(확률)의 차이를 구하게 되기 때문에 Reconstruction term이라 불린다.
        
    '''
    def __init__(self, latent_size=10):
        super().__init__()
        self.latent_size = latent_size
    
    def forward(self, x_prime, x, mu, std):
        '''
            x_prime: predict
            x: target
            mu: q(z|x)의 평균
            std: q(z|x)의 표준편차
        '''
        
        # 1. Regularization Term
        prior_mu, prior_std = torch.zeros(self.latent_size), torch.ones(self.latent_size)
        prior = dist.Normal(prior_mu, prior_std) # p(z)
        variational = dist.Normal(mu, std) # q(z|x)
        first_term = dist.kl_divergence(variational, prior).mean() # args 순서 중요
        
        # 2. Reconstruction Term
        second_term = F.binary_cross_entropy(x_prime, x, reduction='mean') # NLL이랑 같음
        
        elbo = -first_term + second_term
        print(first_term, second_term)
        
        return -elbo # ELBO를 최대화하는 게 목적이기 떄문에 Negative