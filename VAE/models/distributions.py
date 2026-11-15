import torch


def reparametrize(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mean + std * eps
