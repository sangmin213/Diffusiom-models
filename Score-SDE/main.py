import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from model import *
from utils import *
from sampling import *

import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default="true", help='whether train or test')
    parser.add_argument('--dataset', type=str, default="mnist", help='mnist | cifar10')

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

    epochs = 50
    batch_size = 32
    lr = 1e-4

    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

    optimizer = optim.Adam(score_model.parameters(), lr)

    if config.train == "true":
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs} ...")
            epoch_loss = 0.0
            num_items = 0
            for x, y in tqdm(dataloader):
                x = x.to(device)
                
                optimizer.zero_grad()

                loss = loss_fn(score_model, x, marginal_prob_std_fn)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]

            print(f'Epoch Loss: {epoch_loss / num_items}')
            torch.save(score_model.state_dict(), './model.pt')
            torch.save(optimizer, "./optimizer.pt")

    '''Sampling'''
    checkpoint = torch.load('model.pt', map_location = device)
    score_model.load_state_dict(checkpoint)

    sample_batch_size = 64
    sampler = pc_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

    start = time.time()
    samples = sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, 
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