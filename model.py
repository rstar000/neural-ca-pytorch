from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

Size = namedtuple('Size', ['width', 'height'])


def make_state_grid(size, num_channels):
    return torch.zeros(
        num_channels, size.height, size.width,
        dtype=torch.float32)


def make_initial_grid(size, num_channels):
    grid = make_state_grid(size, num_channels)
    grid[:, size.height // 2, size.width // 2] = 1.0
    return grid


def apply_flat_conv(x, conv):
    """Convolve each channel individually"""
    b, c, h, w = x.shape
    x = x.reshape([b*c, h, w]).unsqueeze(1)
    y = conv(x).squeeze(1).reshape([b,c,h,w])
    return y


def make_sobel_filters():
    def make_conv(weight):
        weight_param = nn.Parameter(
            data=torch.from_numpy(weight[None, None, ...]).float(),
            requires_grad=False)
        conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        conv.weight = weight_param
        return conv

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.transpose(sobel_x)
    return [make_conv(x) for x in [sobel_x, sobel_y]]


def stochastic_grid_update(grid, delta):
    grid_size = grid.shape[2], grid.shape[3]  # H x W
    dist = torch.distributions.Uniform(0.0, 1.0)
    mask = dist.sample(torch.Size(grid_size))
    mask = (mask < 0.5).float()
    return grid + delta * mask[None, None, :, :]


def alive_cells_mask(grid, alive_thresh=0.1):
    mask = F.max_pool2d(grid[:,3,:,:], 3, padding=1, stride=1)
    mask = (mask > alive_thresh).float()
    return grid * mask.unsqueeze(1)


class Perception(nn.Module):
    def __init__(self, phi):
        super().__init__()
        self._sobel_x, self._sobel_y = make_sobel_filters()

    def forward(self, grid):
        x, y = [
            apply_flat_conv(grid, sobel)
            for sobel in (self._sobel_x, self._sobel_y)
        ]

        return torch.cat([x, y, grid], dim=1)


class StateNetwork(nn.Module):
    def __init__(self, grid_channels, hidden_channels):
        super().__init__()
        self._conv1 = nn.Conv2d(
            grid_channels * 3, hidden_channels, kernel_size=1, padding=0)

        self._conv2 = nn.Conv2d(
            hidden_channels, grid_channels, kernel_size=1, padding=0)
        nn.init.zeros_(self._conv2.weight)
        nn.init.zeros_(self._conv2.bias)

    def forward(self, grid):
        x = self._conv1(grid)
        x = F.relu(x)
        x = self._conv2(x)
        return x


class CellNetwork(nn.Module):
    def __init__(self, grid_channels):
        super().__init__()
        self._perception = Perception(0.0)
        self._state_updater = StateNetwork(grid_channels, 128)

    def forward(self, grid):
        x = self._perception(grid)
        x = self._state_updater(x)
        x = stochastic_grid_update(grid, x)
        x = alive_cells_mask(x)
        return x
