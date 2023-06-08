import torch
import torch.nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
''' 
time t는 (0,1)의 범위에 존재함 
'''

def marginal_prob_std(t, sigma):
    '''Compute the std of p_{0t}(x(t) | x(0))'''
    t = torch.tensor(t, device=device)

    return torch.sqrt((sigma**(2*t) - 1.0) / 2.0 / np.log(sigma)) # (sigma^{2*{T=t}}-sigma^{2*{T=0})} / (2*log(sigma)) # VE SDE의 solution, 논문의 식(29)

# Brownian motion의 g(*)
def diffusion_coeff(t, sigma):
    '''Compute the diffusion coefficient of our SDE'''
    return torch.tensor(sigma**t, device=device) # sigma_t = {sigma=hyperparmeter}^t 로 정하나 봄. NCSN


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    '''The loss function for training score-based generative models

    Args:
        model: A time-dependent score-based generative model
        x: mini-batch data
        marginal_prob_std: A function that gives the std of the perturbation kernel
        eps: A tolerance value for numerical stability
    '''
    random_t = torch.rand(x.shape[0], device=x.device)*(1.0 - eps) + eps # time이 완전히 0 혹은 1이 되지 않도록 정의. t \in [e,1). 논문에 완전 0에서 할 수 없는 이유 나와있음.
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + std[:, None, None, None] * z # VE SDE. 논문의 식 (20)

    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score*std[:,None,None,None] + z)**2, dim=(1,2,3)))

    return loss



