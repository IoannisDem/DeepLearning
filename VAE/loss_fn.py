import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_reconstructed, z_mean, z_log_var):
        reconstruction_loss = F.mse_loss(
            x_reconstructed, x
        )  # assuming constant varriance (non parametric)
        kl_loss = (
            -0.5 * torch.sum(1 + z_log_var - z_mean**2 - z_log_var.exp(), dim=1).mean()
        )
        return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss
