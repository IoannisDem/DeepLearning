import torch
import torch.nn as nn
from models import encoder, decoder, distributions


class VAE(nn.Module):
    def __init__(self, channels_dim, latent_dim=32, feature_size=7):
        super().__init__()
        self._encoder = encoder.SimpleEncoder(channels_dim)
        _reverse_channels_dim = channels_dim[::-1]
        self._decoder = decoder.SimpleDecoder(
            latent_dim, feature_size, _reverse_channels_dim
        )

        flat_dim = channels_dim[-1] * feature_size * feature_size
        self._x_mean_layer = nn.Linear(flat_dim, latent_dim)
        self._x_log_var_layer = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        x = self._encoder(x)
        z_mean = self._x_mean_layer(x)
        z_log_var = self._x_log_var_layer(x)
        z = distributions.reparametrize(z_mean, z_log_var)
        x_reconstructed = self._decoder(z)
        return x_reconstructed, z_mean, z_log_var
