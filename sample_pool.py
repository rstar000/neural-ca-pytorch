import random
import numpy as np
import torch


class SamplePool:
    def __init__(self, pool_size, init):
        self._size = pool_size
        self._init = init

        self._items = [init] * pool_size

    def _sample(self, batch_size):
        idxs = np.random.choice(self._size, batch_size)
        return idxs, [self._items[idx] for idx in idxs]

    def put_back(self, idxs, items):
        for idx, item in zip(idxs, items):
            self._items[idx] = item

    def make_batch(self, batch_size, loss, damage=False):
        idxs, samples = self._sample(batch_size)
        batch = torch.stack(samples)
        samples, loss_vals = loss.forward(batch)
        loss_vals = loss_vals.detach().cpu().numpy()

        order = np.argsort(loss_vals)[::-1]
        replaced = order[0]

        if damage:
            for i, elem in enumerate(batch):
                batch[i] = damage_random(elem)

        batch[replaced] = self._init
        return idxs, batch

def damage_crop(grid):
    c, h, w = grid.shape
    damage_size = random.randint(10, min(h // 2, w // 2))
    x = random.randint(0, w - damage_size)
    y = random.randint(0, h - damage_size)

    grid[:, y:y+damage_size, x:x+damage_size] = 0
    return grid


def damage_quater(grid):
    c, h, w = grid.shape

    damage_size = min(h // 2, w // 2)

    offset_x = w // 2  if random.random() < 0.5 else 0
    offset_y = h // 2  if random.random() < 0.5 else 0

    grid[:, offset_y:offset_y + damage_size, offset_x:offset_x + damage_size] = 0
    return grid


def damage_random(grid):
    funcs = [
        lambda x: damage_quater(x),
        lambda x: damage_crop(x),
        lambda x: x
    ]

    func = random.choice(funcs)
    return func(grid)


