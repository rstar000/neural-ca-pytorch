import os
import argparse
import torch

from torch import nn
from torch.utils.tensorboard import SummaryWriter

import model
import pattern
import sample_pool

from loss import Loss

POOL_SIZE = 1024
MIN_ITER = 64
MAX_ITER = 96


class CheckpointCallback:
    def __init__(self, net, checkpoint_steps, train_dir):
        self._net = net
        self._checkpoint_steps = checkpoint_steps
        self._dir = os.path.join(train_dir, 'checkpoints')

        os.makedirs(self._dir, exist_ok=True)

    def __call__(self, step_num):
        if step_num and step_num % self._checkpoint_steps == 0:
            with open(
                os.path.join(self._dir, '{0:04d}.pt'.format(step_num)),
                'wb') as f:
                torch.save(self._net, f)


class SummaryCallback:
    def __init__(self, log_dir):
        self._writer = SummaryWriter(log_dir=log_dir)

    def __call__(self, step_num, scalars):
        for key, value in scalars.items():
            self._writer.add_scalar(key, value, step_num)


def train(args):
    os.mkdir(args.train_dir, exist_ok=True)

    grid_size = model.Size(args.grid_size, args.grid_size)
    initial_grid = model.make_initial_grid(grid_size, args.grid_channels)
    pool = sample_pool.SamplePool(
        pool_size=POOL_SIZE,
        init=initial_grid)

    gt = pattern.load_pattern(args.pattern, max_size=min(grid_size))
    net = model.CellNetwork(args.grid_channels)
    optimizer = torch.optim.Adam(params=net.parameters(), weight_decay=1e-4)
    loss = Loss(gt, net, MIN_ITER, MAX_ITER)

    checkpoint_callback = CheckpointCallback(
        net, args.checkpoint_steps, args.train_dir)

    summary_callback = SummaryCallback(args.train_dir)

    def train_basic():
        batch = initial_grid.unsqueeze(0).repeat(args.batch_size, 1, 1, 1)
        optimizer.zero_grad()
        samples, loss_values = loss.forward(batch)
        mean_loss = loss_values.mean()
        mean_loss.backward()
        optimizer.step()
        return mean_loss

    def train_with_pool(damage):
        idxs, batch = pool.make_batch(args.batch_size, loss, damage=damage)
        optimizer.zero_grad()
        samples, loss_values = loss.forward(batch)
        mean_loss = loss_values.mean()
        mean_loss.backward()
        optimizer.step()
        pool.put_back(idxs, samples.detach())
        return mean_loss


    train_funcs = {
        'growing': train_basic,
        'persistent': lambda: train_with_pool(False),
        'regenerative': lambda: train_with_pool(True)
    }

    train_step = train_funcs[args.model_type]

    for i in range(args.num_iter):
        mean_loss = train_step()
        checkpoint_callback(i)
        summary_callback(i, {
            'loss': float(mean_loss.detach().cpu().numpy())
        })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-dir',
        type=str,
        required=True)

    parser.add_argument(
        '--pattern',
        type=str,
        required=True)

    parser.add_argument(
        '--grid-size',
        type=int,
        default=64)

    parser.add_argument(
        '--grid-channels',
        type=int,
        default=16)

    parser.add_argument(
        '--num-iter',
        type=int,
        default=6000)

    parser.add_argument(
        '--checkpoint-steps',
        type=int,
        default=200)

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16)

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['growing', 'persistent', 'regenerative'],
        default='regenerative')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
