import os
import json

import torch
from torch.utils.data import DataLoader

from nn.cifar10_loader import Cifar10


def train():
    pass


def test():
    pass


def main(args):
    # read configurations
    with open(os.path.join(args.config_dir, 'nn.json'), 'r') as f:
        configs = json.load(f)
        for k, v in configs.items():
            configs[k] = eval(v)
    print(configs)

    dataset_tr = Cifar10(args.data_dir, 'train')
    dataset_t = Cifar10(args.data_dir, 'test')
    data_loader_tr = DataLoader(dataset_tr, batch_size=configs['batch_size'],
                                pin_memory=True, num_workers=8, shuffle=True)
    data_loader_t = DataLoader(dataset_t, batch_size=configs['batch_size'],
                                pin_memory=True, num_workers=8, shuffle=True)

    for i in range(configs['epoch']):
        pass