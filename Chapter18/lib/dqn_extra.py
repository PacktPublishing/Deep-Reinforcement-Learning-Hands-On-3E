import math

import numpy as np
import torch
from torch import nn as nn
from torchrl.modules import NoisyLinear

# replay buffer params
BETA_START = 0.4
BETA_FRAMES = 100000

# distributional DQN params
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 256),
            NoisyLinear(256, n_actions)
        ]
        self.fc_adv = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]

    def reset_noise(self):
        for n in self.noisy_layers:
            n.reset_noise()


class BaselineDQN(nn.Module):
    """
    Dueling net
    """
    def __init__(self, input_shape, n_actions):
        super(BaselineDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)


class MountainCarBaseDQN(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 128):
        super(MountainCarBaseDQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class MountainCarNoisyNetDQN(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 128):
        super(MountainCarNoisyNetDQN, self).__init__()

        self.noisy_layers = [
            NoisyLinear(hid_size, n_actions),
        ]

        self.net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            self.noisy_layers[0]
        )

    def forward(self, x):
        return self.net(x)

    def reset_noise(self):
        for n in self.noisy_layers:
            n.reset_noise()
