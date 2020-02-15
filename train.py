import argparse
import torch

from torch import nn
from model import CANetwork, Size


def train(args):
    grid_size = Size(args.grid_size, args.grid_size)
    model = CANetwork()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pattern')
    parser.add_argument('--grid-size', type=int)
    parser.add_argument('--num-iter', type=int)

    args = parser.parse_args()


