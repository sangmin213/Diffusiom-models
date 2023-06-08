import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
    '''Gaussian random features for encoding time steps'''
    def __init__(self, embed_dim, scale=30.0):
        super(GaussianFourierProjection, self).__init__()
        # 초기화 시 weight을 random sampling.
        # 본 weight은 optimizing동안 fixed. not trained
        self.W = nn.Parameter(torch.randn(embed_dim // 2)* scale, requires_grad=False)
        # sample w ~ N(0, {scale^2}*I)
        # [sin(2pi*w*t);cos(2pi*w*t)]
        # t : timestep
    
    def forward(self, t):
        t_proj = t[:, None]*self.W[None, :]*2*np.pi
        
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class Dense(nn.Module):
    '''A fully connected layer that reshpaes outputs to feature maps'''
    def __init__(self, input_dim, output_dim):
        super(Dense,self).__init__()

        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)[..., None, None]


class ScoreNet(nn.Module):
    '''A time-dependent score-based model build upon U-Net architecture'''
    def __init__(self, marginal_prob_std, channels=[32,64,128,256], embed_dim=256, in_channel=1):
        '''Initialize a time-dependent score-based network.
        
        Args:
            - marginal_prob_std: A function that takes time "t" and gives the "std" of 
              the perturbation kernel p_{0t}(x(t)|x(0))
        '''
        super(ScoreNet, self).__init__()

        # Gaussian random feature embedding layer for time "t"
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        # swish activation function
        self.act = lambda x: x*torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

        # Encoding layers
        self.conv1 = nn.Conv2d(in_channel, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

        # Decoding layers
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], in_channel, 3, stride=1)

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for time step "t"
        embed = self.act(self.embed(t))

        # Encoding path
        h1 = self.conv1(x)    
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class HalfScoreNet(nn.Module):
    '''A time-dependent score-based model build upon U-Net architecture'''
    def __init__(self, marginal_prob_std, channels=[32,64,128,256], embed_dim=64, in_channel=1):
        '''Initialize a time-dependent score-based network.
        
        Args:
            - marginal_prob_std: A function that takes time "t" and gives the "std" of 
              the perturbation kernel p_{0t}(x(t)|x(0))
        '''
        super(HalfScoreNet, self).__init__()

        # Gaussian random feature embedding layer for time "t"
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        # swish activation function
        self.act = lambda x: x*torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

        # Encoding layers
        self.conv1 = nn.Conv2d(in_channel, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        # Decoding layers
        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense3 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], in_channel, 3, stride=1)

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for time step "t"
        embed = self.act(self.embed(t))

        # Encoding path
        h1 = self.conv1(x)    
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        # Decoding path
        h = self.tconv2(h2)
        ## Skip connection from the encoding path
        h += self.dense3(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
