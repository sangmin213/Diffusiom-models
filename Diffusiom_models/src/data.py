from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

import os


class AAPMDataset(Dataset):
    def __init__(self, path="train", imgSize = 512):
        super(AAPMDataset, self).__init__()
        self.path = path
        self.imgSize = imgSize

        if self.path == 'train':
            self.plus = 500
        else:
            self.plus = 0

        if path == "train":
            self.length = 3839 # train data set cnt
        else:
            self.length = 421 # test data set cnt

        self.imgs_A = os.listdir(f'/home/math1/Diffusion/denoising-diffusion-pytorch/AAPM_data/{path}/full_dose')
        self.imgs_B = os.listdir(f'/home/math1/Diffusion/denoising-diffusion-pytorch/AAPM_data/{path}/quarter_dose')

    def __len__(self):
        return min(len(self.imgs_A), len(self.imgs_B))

    def __getitem__(self, idx): # length = self.__len__() : 매번 길이 계산하는 함수 불러오면 많이 느려질 것 같아서
        self.imgA = torch.tensor(np.load(f'/home/math1/Diffusion/denoising-diffusion-pytorch/AAPM_data/{self.path}/full_dose/{idx+1}.npy')).view(1,512,512)
        self.imgB = torch.tensor(np.load(f'/home/math1/Diffusion/denoising-diffusion-pytorch/AAPM_data/{self.path}/quarter_dose/{(idx+self.plus)%self.length+1}.npy')).view(1,512,512) # full & quarter not aligned (= not paired) <- Unsupervised

        self.imgA = self.minmax_normalization(self.imgA)
        self.imgB = self.minmax_normalization(self.imgB)

        self.imgA = self.train_transform(self.imgA)
        self.imgB = self.train_transform(self.imgB)

        return (self.imgA, self.imgB)
        
    def minmax_normalization(self, x):
        mini, maxi = torch.min(x), torch.max(x)
        x = (x - mini) / (maxi - mini) # min-max : 0 to 1

        return x

    def train_transform(self, x): # post-process ( 원상 복구 = 시각화 용 )
        transform = transforms.Compose([transforms.Resize(self.imgSize)]) 

        return transform(x)

    def test_transform(self, x): # post-process
        transform = transforms.Compose([transforms.Resize(self.imgSize),
                                        transforms.Normalize((-0.01,),(0.1,))]) # 데이터가 이미 정규화 됨. mean = 0.009 = std(?) 라서, mean은 -0.01, std = 0.1로 다시 정규화하면 원상 복구.
                                                                                # 보통 min = -0.3 max = 0.5 이정도? -> 분포를 봤을 때 generator out layer로 sigmoid() or tanh() 쓰기 적합하지 않아보임
                                                                                # out layer를 conv로 그냥 끝내버리자
        
        return transform(x)