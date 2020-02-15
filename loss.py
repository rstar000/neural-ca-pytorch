import random

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, pattern, net, min_iter, max_iter):
        super().__init__()
        self._min_iter = min_iter
        self._max_iter = max_iter
        self._pattern = pattern
        self._net = net
        self._loss = nn.MSELoss(reduction='none')

    def forward(self, x):
        num_iter = random.uniform(self._min_iter, self._max_iter)
        for i in range(int(num_iter)):
            x = self._net(x)

        gt_batch = _repeat(self._pattern, x.shape[0])
        return x, self._loss(x[:,:4], gt_batch[:,:4])


def _repeat(x, n_times):
    x = x.unsqueeze(0)
    x = x.repeat(n_times, 1, 1, 1)
    return x
