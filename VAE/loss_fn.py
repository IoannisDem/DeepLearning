import torch
import torch.nn as nn
import torch.nn.functional as F


def _kl_loss(z_mean, z_log_var):
    return -0.5 * torch.sum(1 + z_log_var - z_mean**2 - z_log_var.exp(), dim=1).mean()


class VAELossMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_reconstructed, z_mean, z_log_var, kl_weight=1.0):
        reconstruction_loss = F.mse_loss(x_reconstructed, x)
        num_pixels = x.shape[-1] * x.shape[-2]
        kl_loss = _kl_loss(z_mean, z_log_var) / num_pixels
        return reconstruction_loss + kl_weight * kl_loss, reconstruction_loss, kl_loss

