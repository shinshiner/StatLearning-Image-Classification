import os
import random
import numpy as np
import json
import pickle

import torch
from sklearn.preprocessing import StandardScaler


# split training and testing set
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
def sk_read(data_path, normal=True):
    if os.path.exists('origin_data/feats_%d.npy' % normal) and os.path.exists('origin_data/lbls_%d.npy' % normal):
        feats = np.load('origin_data/feats_%d.npy' % normal).astype(np.float32)
        lbls = np.load('origin_data/lbls_%d.npy' % normal).astype(np.uint8)
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

    if normal:
        scaler = StandardScaler()
        scaler.fit(feats)
        scaler.transform(feats)

    np.save('origin_data/feats_%d.npy' % normal, feats)
    np.save('origin_data/lbls_%d.npy' % normal, lbls)

    print('======= numpy-array format data ready ! =======')

    return feats, lbls


# read data for sklearn format (numpy-array)
def sk_read_eval(data_path, normal=False):
    if os.path.exists('origin_data/feats_eval_%d.npy' % normal):
        feats = np.load('origin_data/feats_eval_%d.npy' % normal).astype(np.float32)
        print('======= numpy-array format data ready ! =======')

        return feats

    feats = []

    with open(data_path, 'r') as f:
        lines = f.readlines()[1:]

    for line in lines:
        items = line.rstrip().split(',')
        feat = items[1:]

        feats.append(feat)

    feats = np.array(feats).astype(np.float32)

    if normal:
        # scaler = StandardScaler()
        # scaler.fit(feats)
        # scaler.transform(feats)
        feats = (feats - feats.mean()) / feats.std()

    np.save('origin_data/feats_eval_%d.npy' % normal, feats)

    print('======= numpy-array format data ready ! =======')

    return feats


# read configurations from .json files
def config_parser(config_path):
    if not os.path.exists(config_path): # for NULL configs
        return {}

    with open(config_path, 'r') as f:
        configs = json.load(f)
        for k, v in configs.items():
            if not v.isalpha():
                configs[k] = eval(v)

    if configs.get('gpu'):
        configs['gpu'] = configs['gpu'] and torch.cuda.is_available()

    print('======= Training with the following configurations =======')
    for k, v in configs.items():
        print('  ', k, ': ', v)

    return configs


def null_model():
    if not os.path.exists('demo.pkl'):
        with open('origin_data/train.csv', 'r') as f:
            lines = f.readlines()[1:]
            with open('demo.pkl', 'wb') as ff:
                pickle.dump(lines, ff)
    else:
        with open('demo.pkl', 'rb') as ff:
            lines = pickle.load(ff)

    cnt = 0
    for l in lines:
        pred = random.randint(0, 11)
        if pred == int(l.rstrip().split(',')[-1]):
            cnt += 1

    print(cnt / len(lines))


if __name__ == '__main__':
    null_model()