import numpy as np
import torch

from model import make_initial_grid

class SamplePool:
    def __init__(self, pool_size, init):
        self._size = pool_size
        self._init = init

        self._items = [init] * pool_size

    def sample(self, batch_size):
        idxs = np.random.choice(self._size, batch_size)
        return idxs, [self._items[idx] for idx in idxs]

    def put_back(self, idxs, items):
        for idx, item in zip(idxs, items):
            self._items[idx] = item


def gen_batch(pool, batch_size, loss, init):
    idxs, samples = pool.sample(batch_size)
    sample_batch = torch.stack(samples)
    samples, loss_vals = loss.forward(sample_batch)
    loss_vals = loss_vals.detach().numpy().flatten()
    order = np.argsort(loss_vals)[::-1]

    batch = sample_batch
    batch[order[0]] = init


    return idxs, batch
