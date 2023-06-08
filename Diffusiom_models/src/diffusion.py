import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import math
import numpy as np
from functools import partial
from collections import namedtuple
from tqdm import tqdm
from random import random
from einops import reduce

from .utils import *
from .measurement import InpaintingOperator
from .condition import PosteriorSampling

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.,
        auto_normalize = True
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start
        
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
    

class InPainting_Projection(GaussianDiffusion):
    def __init__(self,
                model,
                *,
                image_size,
                timesteps = 1000,
                sampling_timesteps = None,
                loss_type = 'l1',
                objective = 'pred_noise',
                beta_schedule = 'sigmoid',
                schedule_fn_kwargs = dict(),
                p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
                p2_loss_weight_k = 1,
                ddim_sampling_eta = 0.,
                auto_normalize = True):
        super().__init__(model,
                         image_size=image_size,
                         timesteps=timesteps,
                         sampling_timesteps=sampling_timesteps,
                         loss_type=loss_type,
                         objective=objective,
                         beta_schedule=beta_schedule,
                         schedule_fn_kwargs=schedule_fn_kwargs,
                         p2_loss_weight_gamma=p2_loss_weight_gamma,
                         p2_loss_weight_k=p2_loss_weight_k,
                         ddim_sampling_eta=ddim_sampling_eta,
                         auto_normalize=auto_normalize)
        self.loc = (50,80,20,20) # x, y, h, w

    def p_sample_loop(self, img, loc=None, return_all_timesteps=False, lambda_ = 0.5):
        '''
        Args:
            loc: masked area coordinates. (x,y,h,w) => (x,y) is left top
        '''
        if loc==None:
            loc = self.loc

        x, y, h, w = loc
        noise_img = torch.randn(img.shape, device = img.device)
        imgs  = [noise_img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None

            noise_img, x_start = self.p_sample(noise_img, t, self_cond)

            t_vec = torch.tensor([t for i in range(img.shape[0])], device=img.device).long()
            q_img = super().q_sample(img, t_vec)

            # 생성된 이미지에서 마스크 영역만 추출
            gen_mask = torch.zeros_like(q_img) 
            gen_mask[:, :, x:x+h, y:y+w] = noise_img[:, :, x:x+h, y:y+w]

            # 주어진 이미지에서 마스크 영역 외의 영역 추출
            bg_mask = q_img.clone() # background mask
            bg_mask[:, :, x:x+h, y:y+w] = 0 # background mask

            # inpainting 결합
            noise_img = gen_mask + bg_mask

            if t%100==0:
                save_image(bg_mask, f"./result_inpaint/bg_mask{t}.png") # except mask area of generated image
                save_image(gen_mask, f"./result_inpaint/perturbed_mask{t}.png") # mask area of perturbed original image
                save_image(noise_img, f"./result_inpaint/gen_img{t}.png") # generated image

            imgs.append(noise_img)

        ret = noise_img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # ret = self.unnormalize(ret)
        return ret
    
    def sample(self, img, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(img, return_all_timesteps = return_all_timesteps)


class InPainting_DPS(GaussianDiffusion):
    def __init__(self,
                model,
                *,
                image_size,
                timesteps = 1000,
                sampling_timesteps = None,
                loss_type = 'l1',
                objective = 'pred_noise',
                beta_schedule = 'sigmoid',
                schedule_fn_kwargs = dict(),
                p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
                p2_loss_weight_k = 1,
                ddim_sampling_eta = 0.,
                auto_normalize = True,
                device):
        super().__init__(model,
                         image_size=image_size,
                         timesteps=timesteps,
                         sampling_timesteps=sampling_timesteps,
                         loss_type=loss_type,
                         objective=objective,
                         beta_schedule=beta_schedule,
                         schedule_fn_kwargs=schedule_fn_kwargs,
                         p2_loss_weight_gamma=p2_loss_weight_gamma,
                         p2_loss_weight_k=p2_loss_weight_k,
                         ddim_sampling_eta=ddim_sampling_eta,
                         auto_normalize=auto_normalize)
        
        self.model = model
        self.loc = (50,80,20,20) # x, y, h, w
        self.scale = 1.0 # PS scaling
        self.operator = InpaintingOperator(device)
        self.measurement_cond_fn = PosteriorSampling(self.operator, None, scale = self.scale)

    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            # Apply DPS
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start
        
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)

        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    # @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype = torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def p_sample_loop(self, img, return_all_timesteps=False):
        '''
        Args:
            loc: masked area coordinates. (x,y,h,w) => (x,y) is left top
        '''
        img = self.normalize(img)

        loc = self.loc

        x, y, h, w = loc
        x_t = torch.randn(img.shape, device = img.device)

        # imgs  = [x_t]

        x_0_hat = None 

        # inpainting mask
        mask = torch.ones_like(img, device=img.device)
        mask[:, :, x:x+h, y:y+w] = 0
        # mask = torch.tensor(mask, requires_grad=True)
        
        # measurement
        measurement = img * mask
        
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_0_hat if self.self_condition else None
            
            x_t = x_t.requires_grad_()
            # DPS-Gaussian : Algorithm 1 in Paper
            x_tm1, x_0_hat = self.p_sample(x_t, t, self_cond)
            
            # Posterior Sampling Conditioning
            x_tm1, norm = self.measurement_cond_fn.conditioning(x_t, x_tm1, x_0_hat, measurement, mask)
            
            x_t = x_tm1.clone()
            x_t = x_t.detach() # free(gradient)

            if t%100==0:
                save_image(self.unnormalize(x_0_hat), f"./result_inpaint/x_0_hat{t}.png") # mask area of perturbed original image
                save_image(self.unnormalize(x_tm1), f"./result_inpaint/gen_img{t}.png") # generated image

            # imgs.append(x_tm1) # x_tm1을 imgs에 계속 저장하는 과정에서, 각 원소가 다 gradient를 가지므로 gpu 리소스를 필요로 해 gpu 터짐. 그래서 없앰

        # ret = x_tm1 if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = x_t

        ret = self.unnormalize(ret)
        return ret
    
    def sample(self, img, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn(img, return_all_timesteps = return_all_timesteps)
    

class CCDF(GaussianDiffusion):
    def __init__(self,
                model,
                *,
                image_size,
                timesteps = 1000,
                sampling_timesteps = None,
                loss_type = 'l1',
                objective = 'pred_noise',
                beta_schedule = 'sigmoid',
                schedule_fn_kwargs = dict(),
                p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
                p2_loss_weight_k = 1,
                ddim_sampling_eta = 0.,
                auto_normalize = True):
        super().__init__(model,
                         image_size=image_size,
                         timesteps=timesteps,
                         sampling_timesteps=sampling_timesteps,
                         loss_type=loss_type,
                         objective=objective,
                         beta_schedule=beta_schedule,
                         schedule_fn_kwargs=schedule_fn_kwargs,
                         p2_loss_weight_gamma=p2_loss_weight_gamma,
                         p2_loss_weight_k=p2_loss_weight_k,
                         ddim_sampling_eta=ddim_sampling_eta,
                         auto_normalize=auto_normalize)
        
    @torch.no_grad()
    def p_sample_loop(self, img, return_all_timesteps=False):
        N_prime = 200 # In paper, the new starting time for reverse diffusion

        img = self.normalize(img)

        time = torch.tensor([N_prime]*img.shape[0], dtype=torch.long, device=img.device)
        perturbed_img = self.q_sample(img, t=time)

        imgs = [perturbed_img]

        x_start = None

        for t in tqdm(reversed(range(0, N_prime)), desc = 'sampling loop time step', total = N_prime):
            self_cond = x_start if self.self_condition else None
            perturbed_img, x_start = self.p_sample(perturbed_img, t, self_cond)
            imgs.append(perturbed_img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret     

    def sample(self, img, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn(img, return_all_timesteps = return_all_timesteps)   