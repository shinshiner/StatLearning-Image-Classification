import os
import random
import numpy as np
import json

import torch
from sklearn.preprocessing import StandardScaler


# split training and testing set, typically for pytorch
def split_train_test(source_file, target_dir, seed):
    print('======== spliting dataset =======')
    with open(source_file, 'r') as f:
        lines = f.readlines()[1:]

    train_id = []
    test_id = []
    random.seed(seed)

    # split train-test as 9-1
    for cls in range(12):
        sample_range = list(range(cls * 650, (cls + 1) * 650))
        random.shuffle(sample_range)
        test_id.extend(sample_range[:65])
        train_id.extend(sample_range[65:])

    # write the results
    with open(os.path.join(target_dir, 'train.csv'), 'w') as f:
        for i in train_id:
            f.write(lines[i])

    with open(os.path.join(target_dir, 'test.csv'), 'w') as f:
        for i in test_id:
            f.write(lines[i])


# read data for sklearn format (numpy-array)
def sk_read(data_path):
    if os.path.exists('origin_data/feats.npy') and \
        os.path.exists('origin_data/lbls.npy'):
        feats = np.load('origin_data/feats.npy').astype(np.float32)
        lbls = np.load('origin_data/lbls.npy').astype(np.uint8)
        print('======= numpy-array format data ready ! =======')

        return feats, lbls

    feats = []
    lbls = []

    with open(data_path, 'r') as f:
        lines = f.readlines()[1:]
        random.shuffle(lines)

    for line in lines:
        items = line.rstrip().split(',')
        feat = items[1:-1]
        lbl = items[-1]

        feats.append(feat)
        lbls.append(lbl)

    feats = np.array(feats).astype(np.float32)
    lbls = np.array(lbls).astype(np.uint8)

    scaler = StandardScaler()
    scaler.fit(feats)
    scaler.transform(feats)

    np.save('origin_data/feats.npy', feats)
    np.save('origin_data/lbls.npy', lbls)

    print('======= numpy-array format data ready ! =======')

    return feats, lbls


# read configurations from .json files
def config_parser(config_path):
    with open(config_path, 'r') as f:
        configs = json.load(f)
        for k, v in configs.items():
            if not v.isalpha():
                configs[k] = eval(v)

    if configs.get('gpu'):
        configs['gpu'] = configs['gpu'] and torch.cuda.is_available()
    print(configs)

    return configs


# trick
def compare(f1, f2):
    with open(f1, 'r') as f1:
        with open(f2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

    cnt = 0
    for l1, l2 in zip(lines1, lines2):
        if l1 != l2:
            cnt += 1
            print(l1[:-1], '  ', l2[:-1])
    print('Different result number: %d' % cnt)


if __name__ == '__main__':
    compare('results.csv', 'nn/results-98675.csv')