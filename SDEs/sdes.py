import torch
import numpy as np

from data.data import grach_model

torch.manual_seed(4)

#create Lévy jump process - Tensor
def levy_solver(r, m, v, lam, sigma, T, steps, Npaths):
    dt = T / steps
    v = abs(v)
    rates = torch.rand(steps, Npaths)
    poisson = torch.poisson(rates)
    poi_rv = torch.mul(poisson, torch.normal(m, v).cumsum(dim=0))
    geo = torch.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt +
                        sigma * torch.sqrt(torch.tensor(dt)) * torch.normal(m, v)), dim=0)
    out = torch.exp(geo + poi_rv)
    return out


##create Lévy jump process - numpy
def merton_jump_paths(r, m, v, lam, sigma, T, steps, Npaths):
    size = (steps, Npaths)
    dt = T / steps
    poi_rv = np.multiply(np.random.poisson(lam * dt, size=size),
                         np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt + sigma * np.sqrt(dt) * \
                     np.random.normal(size=size)), axis=0)
    return np.exp(geo + poi_rv)

def generate_noise(noise_size, batch_size, noise_type, rs, params=None):
    noise = []
    if noise_type == 'gbm':
        noise = generate_gbm_paths(noise_size, batch_size)
    elif noise_type == 'normal':
        noise = rs.normal(0, 1, (batch_size, noise_size))
    elif noise_type == 'uniform':
        return torch.rand(batch_size, noise_size) * 2 - 1  # Uniform between -1 and 1
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    return torch.tensor(noise, dtype=torch.float32)


def generate_gbm_paths(noise_size, batch_size, T=1.0, mu=0.02, sigma=0.1):
    """
    Generate Geometric Brownian Motion paths.
    Parameters:
    batch_size (int): Number of paths to generate.
    noise_size (int): Number of steps in each path.
    T (float): Time period.
    mu (float): Drift.
    sigma (float): Volatility.
    Returns:
    torch.Tensor: Simulated GBM paths of shape (batch_size, noise_size).
    """
    dt = T / noise_size
    paths = np.zeros((batch_size, noise_size))

    for i in range(batch_size):
        path = [1]  # Start with an initial value of 1
        for _ in range(1, noise_size):
            z = np.random.normal(0, 1)  # Generate a random normal variable
            new_value = path[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
            path.append(new_value)
        paths[i] = path

    return torch.tensor(paths, dtype=torch.float)
