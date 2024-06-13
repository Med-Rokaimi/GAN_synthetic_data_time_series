import torch
import numpy as np
import torch.nn as nn

from data.data import grach_model


#Merton jump path with minor modification in
# poi_rv = torch.mul(poisson, torch.normal(m, v).cumsum(dim=0))
def levy_solver(r, m, v, lam, sigma, T, steps, Npaths):
    dt = T / steps
    rates = torch.rand(steps, Npaths)
    poisson = torch.poisson(rates)
    poi_rv = torch.mul(poisson, torch.normal(m, v).cumsum(dim=0))


    geo = torch.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt +
                        sigma * torch.sqrt(torch.tensor(dt)) * torch.normal(m, v)), dim=0)
    out = torch.exp(geo + poi_rv)

    return out


'''
def merton_jump_paths(T, r, sigma, lam, m, v, steps, Npaths):
    size = (steps, Npaths)
    dt = T / steps
    poi_rv = np.multiply(np.random.poisson(lam * dt, size=size),
                         np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt +
                     sigma * np.sqrt(dt) *
                     np.random.normal(size=size)), axis=0)

    return np.exp(geo + poi_rv)
'''

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



class OrnsteinUhlenbeckSDE(nn.Module):
        sde_type = 'ito'
        noise_type = 'scalar'

        def __init__(self, mu, theta, sigma):
            super().__init__()
            self.register_buffer('mu', torch.as_tensor(mu))
            self.register_buffer('theta', torch.as_tensor(theta))
            self.register_buffer('sigma', torch.as_tensor(sigma))

        def f(self, t, y):
            return self.mu * t - self.theta * y

        def g(self, t, y, t_size):
            return self.sigma.expand(y.size(0), 1, 1) * (2 * t / t_size)

#ou_sde = OrnsteinUhlenbeckSDE(mu=0.02, theta=0.1, sigma=0.4)



def levy_jump_path_gpt(T, N, mu=0.1, sigma=0.2, jump_lambda=0.5, jump_sigma=0.1, jump_mu=0.1):
    dt = T / N
    t = torch.linspace(0, T, N + 1)
    W = torch.zeros(N + 1)
    J = torch.zeros(N + 1)

    for i in range(1, N + 1):
        W[i] = W[i - 1] + torch.normal(mean=0.0, std=torch.sqrt(dt))
        if torch.rand(1).item() < jump_lambda * dt:
            J[i] = J[i - 1] + torch.normal(mean=jump_mu, std=jump_sigma)
        else:
            J[i] = J[i - 1]

    X = mu * t + sigma * W + J
    return t, X


def heston_model_gpt(T, N, S0=1.0, v0=0.1, kappa=2.0, theta=0.1, xi=0.1, rho=-0.7, r=0.0):
    dt = T / N
    t = torch.linspace(0, T, N + 1)
    S = torch.zeros(N + 1)
    v = torch.zeros(N + 1)
    S[0] = S0
    v[0] = v0

    for i in range(1, N + 1):
        dW1 = torch.normal(mean=0.0, std=torch.sqrt(dt))
        dW2 = rho * dW1 + torch.sqrt(1 - rho ** 2) * torch.normal(mean=0.0, std=torch.sqrt(dt))

        v[i] = v[i - 1] + kappa * (theta - v[i - 1]) * dt + xi * torch.sqrt(v[i - 1]) * dW2
        v[i] = torch.max(v[i], torch.tensor(0.0))  # Ensure variance is non-negative
        S[i] = S[i - 1] * torch.exp((r - 0.5 * v[i - 1]) * dt + torch.sqrt(v[i - 1]) * dW1)

    return t, S, v



def alpha_levy_jump_path_gpt(T, N, alpha=1.5, mu=0.1, sigma=0.2, jump_lambda=0.5, jump_sigma=0.1, jump_mu=0.1):
    dt = T / N
    t = torch.linspace(0, T, N + 1)
    W = torch.zeros(N + 1)
    J = torch.zeros(N + 1)

    for i in range(1, N + 1):
        W[i] = W[i - 1] + torch.normal(mean=0.0, std=torch.sqrt(dt))
        if torch.rand(1).item() < jump_lambda * dt:
            J[i] = J[i - 1] + torch.distributions.stable.Stable(alpha, 0, jump_sigma, jump_mu).sample()
        else:
            J[i] = J[i - 1]

    X = mu * t + sigma * W + J
    return t, X


def generate_noise(noise_size, batch_size, noise_type='gaussian', rs=None, params= None):
    noise = []
    """
    Generate noise for the GAN generator.

    Args:
        batch_size (int): Number of samples in the batch.
        noise_size (int): Dimension of the noise vector.
        noise_type (str): Type of noise ('gaussian', 'uniform', 'levy', 'poisson', 'garch').
        rs (nmupy seed): 

    Returns:
        torch.Tensor: Generated noise.
    """
    if noise_type == 'gaussian':
        return torch.randn(batch_size, noise_size)

    elif noise_type == 'uniform':
        return torch.rand(batch_size, noise_size) * 2 - 1  # Uniform between -1 and 1

    elif noise_type == 'poisson':
        rate = params.get('rate', 1.0)
        return torch.poisson(rate * torch.ones(batch_size, noise_size))

    elif noise_type == 'garch':
        conditional_volatility = params.get('conditional_volatility', torch.ones(batch_size, noise_size))
        return conditional_volatility * torch.randn(batch_size, noise_size)

    if noise_type == 'gbm':

        noise  = generate_gbm_paths(noise_size, batch_size)

    elif noise_type == 'normal':
        noise = rs.normal(0, 1, (batch_size, noise_size))

    elif noise_type == 'garch_me':
        _, noise, _ = grach_model(params.get('ts', 'target'), horizon=1)

    elif noise_type == 'Merton':
        noise = merton_jump_paths(T=1, r=0.02, sigma=0.02, lam=0.02, m=0.02, v=0.02, steps=noise_size, Npaths=1)

    elif noise_type == 'Merton_edited':
        pass
        #noise = levy_solver(r=0.02, m=0.02, v=0.02, lam=0.02, sigma=0.02, T=1, steps=noise_size, Npaths=1)


    elif noise_type == 'ou':
        pass

    elif noise_type == 'levy_stable':
        alpha_levy_jump_path_gpt(noise_size, 1, alpha=1.5, mu=0.1, sigma=0.2, jump_lambda=0.5, jump_sigma=0.1, jump_mu=0.1)


    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")


    return torch.tensor(noise, dtype=torch.float32)




'''

# Example usage
batch_size = 16
noise_dim = 32
gaussian_noise = generate_noise(batch_size, noise_dim, noise_type='gaussian')
uniform_noise = generate_noise(batch_size, noise_dim, noise_type='uniform')
levy_noise = generate_noise(batch_size, noise_dim, noise_type='levy',
                            params={'alpha': 1.5, 'beta': 0.0, 'scale': 1.0, 'loc': 0.0})
poisson_noise = generate_noise(batch_size, noise_dim, noise_type='poisson', params={'rate': 5.0})
# For GARCH, you need to have the conditional volatilities from your fitted GARCH model
# garch_noise = generate_noise(batch_size, noise_dim, noise_type='garch', params={'conditional_volatility': your_conditional_volatility})

'''
def print_merton_example(r = 0.02 , v = 0.02, m = 0.02, lam = 0.02, sigma= 0.02 ):
    import matplotlib.pyplot as plt
    merton_ou = levy_solver(r=torch.tensor(r), v=torch.tensor(v), m=torch.tensor(m),
                            lam=torch.tensor(lam), sigma=torch.tensor(sigma), T=1, steps=16,
                            Npaths=10)
    plt.plot(merton_ou)
    plt.xlabel('Days')
    plt.ylabel(' Price')
    plt.title(' Jump Diffusion Process')
    plt.show()
    #plt.savefig(saved_model_path + "merton:jump.jpg")


