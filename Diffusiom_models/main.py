import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.unet import Unet
from src.diffusion import GaussianDiffusion
from torchvision.utils import save_image

from tqdm import tqdm
import cv2
import pickle
from src.data import AAPMDataset
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default="true", help='whether train or test')

    cfg = parser.parse_args()
    return cfg

cfg = argparser()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 2000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)


if cfg.train == "true":
    lr = 2e-5
    batch_size = 16
    epochs = 100

    optimizer = optim.Adam(model.parameters(),lr)
    dataset = AAPMDataset("train", imgSize=128)
    dataloader = DataLoader(dataset, batch_size, True, num_workers=2)


    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}...")
        epoch_loss = 0.0

        for quad, full in tqdm(dataloader):
            full = full.to(device)

            optimizer.zero_grad()

            loss = diffusion(full)
            loss.backward()    

            optimizer.step()

            epoch_loss += loss * batch_size

        print(f"Loss: {epoch_loss / len(dataset)}")
        
        torch.save(model.state_dict(), "./model.pt")
        torch.save(optimizer.state_dict(), "./optimizer.pt")

ckpt = torch.load("./model.pt", map_location=device)
model.load_state_dict(ckpt)
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 2000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)

with open('data.pickle', 'wb') as f:
    pickle.dump(sampled_images, f, pickle.HIGHEST_PROTOCOL)

save_image(sampled_images, "torch_img2.png")


