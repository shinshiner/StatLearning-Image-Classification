import os
import json

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from nn.cifar10_loader import Cifar10
from nn.model import *


def train(args, configs):
    torch.manual_seed(args.seed)

    # initialize logger
    if not os.path.isdir(os.path.join(args.log_dir, args.method)):
        os.makedirs(os.path.join(args.log_dir, args.method))
    log_tr = open(os.path.join(args.log_dir, args.method, 'train_log.txt'), 'w')
    log_t = open(os.path.join(args.log_dir, args.method, 'test_log.txt'), 'w')

    # initialize data loader
    dataset_tr = Cifar10(args.data_dir, 'train')
    dataset_t = Cifar10(args.data_dir, 'test')
    data_loader_tr = DataLoader(dataset_tr, batch_size=configs['batch_size'],
                            pin_memory=True, num_workers=configs['workers'], shuffle=True)
    data_loader_t = DataLoader(dataset_t, batch_size=1,
                            pin_memory=True, num_workers=configs['workers'], shuffle=True)

    # initialize model & optimizer & loss
    model = CNNCifarClassifer(num_classes=configs['num_classes'])
    optimizer = optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    if configs['gpu']:
        model = model.cuda()
        criterion = criterion.cuda()

    max_accuracy = 0

    for i in range(configs['epoch']):
        # training phase
        print('======= Training =======')
        model = model.train()

        for batch_i, (feat, lbl) in enumerate(data_loader_tr):
            if configs['gpu']:
                feat = feat.cuda()
                lbl = lbl.cuda()

            probs = model(feat)
            loss = criterion(probs, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_i % configs['loss_interval'] == 0:
                if configs['gpu']:
                    loss = loss.detach().cpu().numpy()
                else:
                    loss = loss.detach().numpy()
                log_info = '[Epoch: %d], [training loss: %0.8f]' % (i, loss)
                print(log_info)
                log_tr.write(log_info + '\n')
                log_tr.flush()

        # testing phase
        print('======= Testing =======')
        model = model.eval()

        correct = 0
        for batch_i, (feat, lbl) in enumerate(data_loader_t):
            if configs['gpu']:
                feat = feat.cuda()

            probs = model(feat)
            prob = F.softmax(probs, dim=-1)
            if configs['gpu']:
                pred = prob.max(1, keepdim=True)[1].cpu().numpy()
            else:
                pred = prob.max(1, keepdim=True)[1].numpy()
            correct += pred[0][0] == lbl.numpy()[0]

        accuracy = correct / len(data_loader_t)
        log_info = '[Epoch: %d], [test accuracy: %0.8f]' % (i, accuracy)
        print(log_info)
        log_t.write(log_info + '\n')
        log_t.flush()

        # save the model
        if i % configs['save_interval'] == 0:
            print('saving model')
            if not os.path.isdir(os.path.join(args.model_dir, args.method)):
                os.makedirs(os.path.join(args.model_dir, args.method))
            torch.save(model.state_dict(), os.path.join(args.model_dir, args.method, 'epoch_%d.pth' % i))

        if accuracy > max_accuracy:
            torch.save(model.state_dict(), os.path.join(args.model_dir, args.method, 'best_%0.4f.pth' % accuracy))
            max_accuracy = accuracy

    log_tr.close()
    log_t.close()


def evaluate(args, model_path, configs):
    result = open(os.path.join(args.method, 'results.csv'), 'w')
    result.write('id,categories\n')

    # initialize data loader
    dataset = Cifar10(args.eval_dir, 'test', evaluate=True)
    data_loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=configs['workers'],shuffle=False)

    # initialize model & optimizer & loss
    model = CifarClassifer(num_classes=configs['num_classes'])
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    if configs['gpu']:
        model = model.cuda()
    print('======= Loaded model from %s =======' % model_path)

    for batch_i, (feat, lbl) in enumerate(data_loader):
        if configs['gpu']:
            feat = feat.cuda()

        probs = model(feat)
        prob = F.softmax(probs, dim=-1)
        if configs['gpu']:
            pred = prob.max(1, keepdim=True)[1].cpu().numpy()
        else:
            pred = prob.max(1, keepdim=True)[1].numpy()

        result.write('%s,%s\n' % (str(batch_i), str(pred[0][0])))

    result.close()


def train_cv(args, configs):
    from utils import sk_read
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    feats, lbls = sk_read('origin_data/train.csv')

    # initialize logger
    if not os.path.isdir(os.path.join(args.log_dir, args.method)):
        os.makedirs(os.path.join(args.log_dir, args.method))
    log_tr = open(os.path.join(args.log_dir, args.method, 'train_log.txt'), 'w')
    log_t = open(os.path.join(args.log_dir, args.method, 'test_log.txt'), 'w')

    # initialize model & optimizer & loss
    model = CifarClassifer(num_classes=configs['num_classes'])
    optimizer = optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    if configs['gpu']:
        model = model.cuda()
        criterion = criterion.cuda()

    max_accuracy = 0

    for i in range(configs['epoch']):
        cv_i = 0

        # using k-folds cv
        kfolds = StratifiedKFold(n_splits=configs['n_folds'], random_state=args.seed+i, shuffle=True)
        for tr, t in kfolds.split(feats, lbls):
            cv_i += 1
            print('======= training fold %d of epoch %d ========' % (cv_i, i))

            # training phase
            loss_sum = 0
            for batch_i, (feat, lbl) in enumerate(list(zip(feats[tr], lbls[tr]))):
                feat = torch.from_numpy(feat).unsqueeze(0)
                lbl = torch.from_numpy(np.array(lbl)).unsqueeze(0).long()

                if configs['gpu']:
                    feat = feat.cuda()
                    lbl = lbl.cuda()

                # forward & backward
                probs = model(feat)
                loss = criterion(probs, lbl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # summerize loss
                if configs['gpu']:
                    loss = loss.detach().cpu().numpy()
                else:
                    loss = loss.detach().numpy()
                loss_sum += loss

                if batch_i % configs['loss_interval'] == 0:
                    log_info = '[Epoch: %d], [training loss: %0.8f]' % (i, loss_sum)
                    print(log_info)
                    log_tr.write(log_info + '\n')
                    log_tr.flush()

                    loss_sum = 0

            # testing phase
            correct = 0
            for feat, lbl in zip(feats[t], lbls[t]):
                feat = torch.from_numpy(feat).unsqueeze(0)

                if configs['gpu']:
                    feat = feat.cuda()

                probs = model(feat)
                prob = F.softmax(probs, dim=-1)
                if configs['gpu']:
                    pred = prob.max(1, keepdim=True)[1].cpu().numpy()
                else:
                    pred = prob.max(1, keepdim=True)[1].numpy()
                correct += pred[0][0] == lbl

            accuracy = correct / (feats.shape[0] / configs['n_folds'])
            log_info = '[Epoch: %d], [test accuracy: %0.8f]' % (i, accuracy)
            print(log_info)
            log_t.write(log_info + '\n')
            log_t.flush()

            # save the model
            if i % configs['save_interval'] == 0:
                print('======= saving model ========')
                if not os.path.isdir(os.path.join(args.model_dir, args.method)):
                    os.makedirs(os.path.join(args.model_dir, args.method))
                torch.save(model.state_dict(), os.path.join(args.model_dir, args.method, 'epoch_%d.pth' % i))

            if accuracy > max_accuracy:
                torch.save(model.state_dict(), os.path.join(args.model_dir, args.method, 'best_%0.4f.pth' % accuracy))
                max_accuracy = accuracy


def main(args, configs):
    if args.mode == 'train':
        train_cv(args, configs)
    elif args.mode == 'test':
        evaluate(args, model_path=os.path.join(args.model_dir, args.method, 'best.pth'), configs=configs)
    else:
        raise NotImplemented