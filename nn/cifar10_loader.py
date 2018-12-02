import numpy as np
import os

from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Cifar10(Dataset):
    def __init__(self, data_dir, mode):
        self.feat = []
        self.lbl = []
        self.data_dir = data_dir
        self.mode = mode

        self._read_raw()

        # transfer format
        self.feat = np.array(self.feat)
        self.lbl = np.array(self.lbl)

        print('Loaded dataset from %s with size %d' % (self.data_file, self.feat.shape[0]))

    # read train / test data
    def _read_raw(self):
        self.data_file = os.path.join(self.data_dir, '%s.csv' % self.mode)
        with open(self.data_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            items = line.rstrip().split(',')
            feat = items[1:-1]
            lbl = items[-1]

            self.feat.append(feat)
            self.lbl.append(lbl)

    def __getitem__(self, index):
        feat = transforms.ToTensor()(self.feat[index])
        lbl = transforms.ToTensor()(self.lbl[index])

        return feat, lbl

    def __len__(self):
        return self.feat.shape[0]