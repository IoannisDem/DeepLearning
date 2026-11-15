import torch
import torch.nn as nn
from models import utils


class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, feature_size, channel_dims):
        super().__init__()
        self._feature_size = feature_size
        self._initial_channels = channel_dims[0]

        self._projection = nn.Linear(
            latent_dim, self._initial_channels * feature_size * feature_size
        )
        layers = []
        for i in range(1, len(channel_dims)):
            layers.append(utils.UpConvBlock(channel_dims[i - 1], channel_dims[i]))
        self._layers = nn.Sequential(*layers)

    def forward(self, z):
        x = self._projection(z)
        x = x.view(
            x.size(0), self._initial_channels, self._feature_size, self._feature_size
        )
        return self._layers(x)
