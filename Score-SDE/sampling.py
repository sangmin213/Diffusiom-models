import torch
import torch.nn
import numpy as np
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp

num_steps = 500

def Euler_Maruyama_sampler(model, marginal_prob_std, diffusion_coeff, batch_size=64, num_steps=num_steps, device='cuda', eps=1e-3, in_channel=1):
    '''Generate samples from score-based model with the Euler-Maruyama solver for SDE.'''
    t = torch.ones(batch_size, device=device) # 한 batch를 학습시킬 때, 각 input들의 time-step( 당연하게도 하나로 통일 ).
    init_x = torch.randn(batch_size, in_channel, 28, 28, device=device) * marginal_prob_std(t) # x_T which is on N(0,I). So, x_T = 0 + z*sigma (Note that, T=1)
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    step_size = time_steps[0]-time_steps[1] # 1.0 ~ eps 까지 sampling step.(t=0 : x_0 = what we want)
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    # Do not include any noise in the last sampling step -> So that return 'x_mean' not 'x'
    return mean_x


signal_to_noise_ratio = 0.16
# 논문 Alg.4
def pc_sampler(model, marginal_prob_std, diffusion_coeff, batch_size=64, num_steps=num_steps, snr = signal_to_noise_ratio, device='cuda', eps=1e-3, in_channel=1):
    '''Generate samples from score-based models with Predictor-Corrector method.'''
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, in_channel, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size).to(device) * time_step
            # Corrector step (Langevin MCMC)
            grad = model(x, batch_time_step) # score = gradient of p(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:])) # 왜 x.shape을 사용하지? z가 아니고? -> 적은 batch_size로 sampling 할 때에는 실험적으로 이런 방식이 더 좋더라.
            langevin_step_size = 2*(snr*noise_norm / grad_norm)**2 # Alg.4의 6th
            x = x + langevin_step_size*grad + torch.sqrt(2*langevin_step_size)*torch.randn_like(x)
            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * step_size * model(x, batch_time_step) # (g**2) * (step size) = sigma^t - sigma^(t-\delta{t})
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)
    # The last step does not include any noise -> So that return 'x_mean' not 'x'
    return x_mean


error_tolerance = 1e-5 # The error tolerance for the black-box ODE solver
def ode_sampler(model, marginal_prob_std, diffusion_coeff, batch_size=64, 
                atol=error_tolerance, rtol=error_tolerance, device='cuda', z=None, eps=1e-3, in_channel=1):
    '''Generate samples from score-based models with black-box ODE solvers.
    
    Args:
        atol: Tolerance of absolute errors. # Parameters for Runge-Kutta method (= representative ODE solver). 
        rtol: Tolerance of relative errors. # The solver is implemented in scipy.integrate.solve_ivp
        z: The latent code that governs the final sample. 
           If None, we start from p_1; otherwise, we start from given z. 
    '''
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = torch.randn(batch_size, in_channel, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        '''A wrapper of the score-based model for use by the ODE solver.'''
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
        with torch.no_grad():
            score = model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        '''The ODE function for use by the ODE solver.'''
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5*(g**2)*score_eval_wrapper(x, time_steps)
    
    # Run the black-box ODE solver.
    res = solve_ivp(ode_func, (1.0, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:,-1], device=device).reshape(shape)

    return x