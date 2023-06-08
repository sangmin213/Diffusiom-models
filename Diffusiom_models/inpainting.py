import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tqdm import tqdm
import pickle

from src.unet import Unet
from src.diffusion import InPainting_Projection, InPainting_DPS
from src.data import AAPMDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eps_model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    )
    ckpt = torch.load("./model.pt")
    eps_model.load_state_dict(ckpt)
    eps_model.to(device)
    eps_model.eval()

    diffusion = InPainting_DPS(
        eps_model,
        image_size = 128,
        timesteps = 2000,   # number of steps
        loss_type = 'l1',    # L1 or L2
        device=device
    ).to(device)

    batch_size = 1

    dataset = AAPMDataset("train", imgSize=128)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    for _, full in dataloader:
        full = full.to(device)
        
        result = diffusion.sample(full, eps_model)

        save_image(full, "./origin.png")
        save_image(result, "./inpaint.png")

        break # 예시로 데이터 하나만 뽑아내고 종료
    
    return


if __name__ == '__main__':
    main()