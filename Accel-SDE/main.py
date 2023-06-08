import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

from model import ScoreNet, HalfScoreNet
from utils import loss_fn, marginal_prob_std, diffusion_coeff
from sampling import pc_sampler, ode_sampler, Euler_Maruyama_sampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default="true", help='whether train or test')
    parser.add_argument('--dataset', type=str, default="mnist", help='mnist | cifar10')
    parser.add_argument('--half', type=str, default="true", help='true: SBS | false: eDiff-I')

    cfg = parser.parse_args()
    return cfg


def main(config):
    '''Train'''
    if config.dataset == "mnist":
        dataset = MNIST('.',train=True, transform=transforms.ToTensor(), download=True)    
        in_channel = 1
    elif config.dataset == "cifar10": 
        dataset = CIFAR10('.',train=True, transform=transforms.ToTensor(), download=True)
        in_channel = 3
    else:
        assert(False, "Not proper dataset.")

    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma) # sigma 값만 고정시켜 정의함
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    score_model = nn.DataParallel(ScoreNet(marginal_prob_std_fn, in_channel=in_channel)).to(device)
    if config.half == "true":
        half_score_model = nn.DataParallel(HalfScoreNet(marginal_prob_std_fn, in_channel=in_channel)).to(device)
    else:
        half_score_model = nn.DataParallel(ScoreNet(marginal_prob_std_fn, in_channel=in_channel)).to(device)

    epochs = 50
    batch_size = 64
    lr = 1e-4
    eps = 1e-5 # for loss function. range of time-step "t" is [eps, 1]

    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

    optimizer = optim.Adam(score_model.parameters(), lr)
    optimizer_half = optim.Adam(half_score_model.parameters(), lr)

    if config.train == "true":
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs} ...")
            epoch_loss = 0.0
            epoch_loss_h = 0.0
            num_items = 0
            num_items_h = 0
            for x, y in tqdm(dataloader):
                x = x.to(device)
                
                optimizer.zero_grad()
                optimizer_half.zero_grad()

                random_t = torch.rand(x.shape[0], device=x.device)*(1.0 - eps) + eps # time이 완전히 0 혹은 1이 되지 않도록 정의. t \in [e,1). 논문에 완전 0에서 할 수 없는 이유 나와있음.
                random_t_0 = random_t[random_t<=0.5]
                random_t_1 = random_t[random_t>0.5]
                x_0_size = random_t_0.shape[0]

                # half to x_0
                loss = loss_fn(score_model, x[:x_0_size,], marginal_prob_std_fn, random_t = random_t_0)
                loss.backward()
                optimizer.step()

                # half from x_1
                loss_h = loss_fn(half_score_model, x[x_0_size:,], marginal_prob_std_fn, random_t = random_t_1, half=True)
                loss_h.backward()
                optimizer_half.step()

                epoch_loss += loss.item() * x_0_size
                epoch_loss_h += loss_h.item() * (batch_size - x_0_size)
                num_items += x_0_size
                num_items_h += (batch_size - x_0_size)

            print(f'Epoch Loss: {epoch_loss / num_items} | Half Loss: {epoch_loss_h / num_items_h}')
            torch.save(score_model.state_dict(), './model.pt')
            torch.save(half_score_model.state_dict(), './half_model.pt')
            torch.save(optimizer, "./optimizer.pt")
            torch.save(optimizer_half, "./optimizer_half.pt")

    '''Sampling'''
    checkpoint = torch.load('./model.pt', map_location = device)
    score_model.load_state_dict(checkpoint)
    checkpoint_h = torch.load('./half_model.pt', map_location = device)
    half_score_model.load_state_dict(checkpoint_h)

    sample_batch_size = 64
    sampler = pc_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

    start = time.time()
    samples = sampler(score_model, half_score_model, marginal_prob_std_fn, diffusion_coeff_fn, 
                    sample_batch_size, device=device, in_channel=in_channel)
    end = time.time()
    
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    print(f"Sampling Time: {end-start}")
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imsave("./sample.png", sample_grid.permute(1, 2, 0).cpu().numpy(), vmin=0.0, vmax=1.0)

    return


if __name__ == '__main__':
    cfg = argparser()
    main(cfg)