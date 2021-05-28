import argparse
import os

import numpy as np
import torch
from torchvision import datasets, transforms

from score.inception import InceptionV3
from score.fid import get_statistics

DIM = 2048
device = torch.device('cuda:0')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate states of CIFAR10/STL10")
    parser.add_argument("--output", type=str, default='cifar_10_fid_stats.npz',
                        help="stats output path (default=cifar_10_fid_stats.npz)")
    parser.add_argument("--inception_dir", type=str, default=None,
                        help='path to inception model dir (default=None)')
    parser.add_argument("--batch_size", type=int, default=50,
                        help="batch size (default=50)")
    parser.add_argument('--use_torch', action='store_true',
                        help='make torch be backend, or the numpy is used '
                             '(default=False)')
    parser.add_argument('--data_path', type=str, default='data/cifar10')
    args = parser.parse_args()

    dataset = datasets.CIFAR10(
        './data', train=False, download=True,
        transform=transforms.ToTensor())

    def image_generator(dataset):
        for x, _ in dataset:
            yield x.numpy()

    m, s = get_statistics(
        image_generator(dataset), num_images=len(dataset), batch_size=50,
        use_torch=args.use_torch, verbose=True, parallel=False, inception_dir=inception_dir)

    if args.use_torch:
        m = m.cpu().numpy()
        s = s.cpu().numpy()
    np.savez_compressed(args.output, m=m, s=s)
