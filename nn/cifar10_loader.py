import os
import numpy as np
from sklearn.decomposition import PCA

import torch
from torch.utils.data.dataset import Dataset


class Cifar10(Dataset):
    def __init__(self, data_dir, mode, evaluate=False, normal=False, pca=False):
        self.feat = []
        self.lbl = []
        self.data_dir = data_dir
        self.mode = mode
        self.evaluate = evaluate

        self._read_raw()

        # transfer format
        self.feat = np.array(self.feat).astype(np.float32)
        self.lbl = np.array(self.lbl).astype(np.uint8)

        if normal:
            self.feat = (self.feat - self.feat.mean()) / self.feat.std()

        if pca:
            pca_estimator = PCA(n_components=512)
            self.feat = pca_estimator.fit_transform(self.feat)
            print(pca_estimator.explained_variance_ratio_)

        print('Loaded dataset from %s with size %d' % (self.data_file, self.feat.shape[0]))

    # read training / testing data
    def _read_raw(self):
        self.data_file = os.path.join(self.data_dir, '%s.csv' % self.mode)
        with open(self.data_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            items = line.rstrip().split(',')

            if not self.evaluate:   # for training and testing
                feat = items[1:-1]
                lbl = items[-1]
            else:                   # for evaluation
                feat = items[1:]
                lbl = 0

            self.feat.append(feat)
            self.lbl.append(lbl)

    def __getitem__(self, index):
        feat = torch.from_numpy(self.feat[index])
        lbl = torch.from_numpy(np.array(self.lbl[index])).long()

        return feat, lbl

    def __len__(self):
        return self.feat.shape[0]