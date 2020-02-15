import random

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F


def repeat(x, n_times):
    x = x.unsqueeze(0)
    x = x.repeat(n_times, 1, 1, 1)
    return x


class GrowingLoss(nn.Module):
    def __init__(self, pattern, min_iter, max_iter, net):
        super().__init__()
        self._min_iter = min_iter
        self._max_iter = max_iter
        self._pattern = pattern
        self._net = net
        self._loss = nn.MSELoss(reduction='mean')

    def forward(self, grid):
        num_iter = random.uniform(self._min_iter, self._max_iter)
        x = grid
        for i in range(int(num_iter)):
            x = self._net(x)

        gt_batch = repeat(self._pattern, x.shape[0])
        return x, self._loss(x[:,:4], gt_batch[:,:4])

