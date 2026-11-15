import torch
import torch.nn as nn
from models import utils


class SimpleEncoder(nn.Module):
    def __init__(self, channel_dims):
        super().__init__()
        layers = []
        for i in range(1, len(channel_dims)):
            layers.append(utils.ConvBlock(channel_dims[i - 1], channel_dims[i], stride=2))
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self._layers(x)
        return x.view(x.size(0), -1)
