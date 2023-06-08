import torch
from torch.utils.data import DataLoader

from src.unet import Unet
from src.diffusion import CCDF
from torchvision.utils import save_image

from src.data import AAPMDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1
).to(device)

ckpt = torch.load("./model.pt", map_location=device)
model.load_state_dict(ckpt)
model.eval()

diffusion = CCDF(
    model,
    image_size = 128,
    timesteps = 2000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)

batch_size = 1
dataset = AAPMDataset("train", imgSize=128)
dataloader = DataLoader(dataset, batch_size, True, num_workers=2)

for quarter, full in dataloader:
    full = full.to(device)
    quarter = quarter.to(device)
    
    result = diffusion.sample(quarter)

    save_image(full, "./FDCT.png")
    save_image(quarter, "./LDCT.png")
    save_image(result, "./CCDF.png")

    break # 예시로 데이터 하나만 뽑아내고 종료