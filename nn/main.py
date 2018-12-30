import os
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from nn.cifar10_loader import Cifar10
from nn.model import *
from utils import sk_read, sk_read_eval, split_train_test


def train(args, configs):
    torch.manual_seed(args.seed)
    if configs['separate']:
        split_train_test('origin_data/train.csv', 'splited_data', args.seed)

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
    model = CifarClassifer2(num_classes=configs['num_classes'])
    optimizer = optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    if configs['gpu']:
        model = model.cuda()
        criterion = criterion.cuda()

    max_accuracy = 0
    loss_sum = 0

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

    torch.save(model.state_dict(), os.path.join(args.model_dir, args.method, 'final.pth'))
    log_tr.close()
    log_t.close()


def evaluate(args, model_path, configs):
    result = open(os.path.join(args.method, 'results.csv'), 'w')
    result.write('id,categories\n')

    # initialize data
    feats = sk_read_eval('origin_data/test.csv', normal=False)

    # initialize model
    model = CifarClassifer2(num_classes=configs['num_classes'])
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    if configs['gpu']:
        model = model.cuda()
    print('======= Loaded model from %s =======' % model_path)

    for batch_i, feat in enumerate(feats):
        feat = torch.from_numpy(feat).unsqueeze(0)

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
    feats, lbls = sk_read('origin_data/train.csv', normal=False)

    # initialize logger
    if not os.path.isdir(os.path.join(args.log_dir, args.method)):
        os.makedirs(os.path.join(args.log_dir, args.method))
    log_tr = open(os.path.join(args.log_dir, args.method, 'train_log.txt'), 'w')
    log_t = open(os.path.join(args.log_dir, args.method, 'test_log.txt'), 'w')

    # initialize model & optimizer & loss
    model = CNNCifarClassifer(num_classes=configs['num_classes'])
    optimizer = optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    if configs['gpu']:
        model = model.cuda()
        criterion = criterion.cuda()

    max_accuracy = 0

    # using k-folds cv
    kfolds = StratifiedKFold(n_splits=configs['n_folds'], random_state=args.seed, shuffle=True)
    cv_i = 0
    for tr, t in kfolds.split(feats, lbls):
        cv_i += 1
        feat_tr, feat_t, lbl_tr, lbl_t = feats[tr], feats[t], lbls[tr], lbls[t]
        for i in range(configs['epoch']):
            print('======= training fold %d of epoch %d ========' % (cv_i, i))

            # shuffle training data
            feat_shuffle, lbl_shuffle = shuffle(feat_tr, lbl_tr, random_state=args.seed + i)

            # training phase
            loss_sum = 0
            model = model.train()
            for batch_i in range(0, feat_shuffle.shape[0], configs['batch_size']):
                feat = torch.from_numpy(feat_shuffle[batch_i: min(feat_shuffle.shape[0], batch_i + configs['batch_size'])])
                lbl = torch.from_numpy(lbl_shuffle[batch_i: min(lbl_shuffle.shape[0], batch_i + configs['batch_size'])]).long()

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
            model = model.eval()
            for feat, lbl in zip(feat_t, lbl_t):
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
                torch.save(model.state_dict(), os.path.join(args.model_dir, args.method, 'epoch_%d_%d.pth' % (cv_i, i)))

            if accuracy > max_accuracy:
                torch.save(model.state_dict(), os.path.join(args.model_dir, args.method, 'best_%0.4f.pth' % accuracy))
                max_accuracy = accuracy

    torch.save(model.state_dict(), os.path.join(args.model_dir, args.method, 'final.pth'))


def bagging(args, model_path_list, configs):
    result = open(os.path.join(args.method, 'results.csv'), 'w')
    result.write('id,categories\n')

    # initialize data
    feats = sk_read_eval('origin_data/test.csv', normal=False)

    # initialize model
    model_list = []
    for sub_model_path, sub_model_type in model_path_list:
        if sub_model_type == 'nn':
            model = CifarClassifer(num_classes=configs['num_classes'])
        elif sub_model_type == 'cnn':
            model = CNNCifarClassifer(num_classes=configs['num_classes'])
        elif sub_model_type == 'pcd':
            model = PointNetfeat(num_classes=configs['num_classes'])
        elif sub_model_type == 'simplenn':
            model = CifarClassifer2(num_classes=configs['num_classes'])

        model.load_state_dict(torch.load(sub_model_path))
        if configs['gpu']:
            model = model.cuda()
        model = model.eval()
        model_list.append(model)

        print('======= Loaded model from %s =======' % sub_model_path)

    # start to inference
    for batch_i, feat in enumerate(feats):
        feat = torch.from_numpy(feat).unsqueeze(0)

        if configs['gpu']:
            feat = feat.cuda()

        max_probs_list = []
        prob_list = []
        pred_list = []

        for model in model_list:
            probs = model(feat)
            prob = F.softmax(probs, dim=-1)

            if configs['gpu']:
                pred = prob.max(1, keepdim=True)[1].cpu().numpy()

                prob_list.append(prob.cpu().detach().numpy()[0])
                max_probs_list.append(prob.max(1, keepdim=True)[0].cpu().detach().numpy()[0][0])
            else:
                pred = prob.max(1, keepdim=True)[1].numpy()

                prob_list.append(prob.detach().numpy()[0])
                max_probs_list.append(prob.max(1, keepdim=True)[0].detach().numpy()[0][0])

            pred_list.append(pred)

        # choose the prediction
        max_probs_list = np.array(max_probs_list)
        if configs['bagging_mode'] == 'hard':
            max_pred = pred_list[np.argmax(max_probs_list)][0][0]
        elif configs['bagging_mode'] == 'soft':
            prob_sum = np.zeros((configs['num_classes']))
            for sub_prob in prob_list:
                prob_sum += sub_prob
            max_pred = np.argmax(prob_sum)

        result.write('%s,%s\n' % (str(batch_i), str(max_pred)))

    result.close()


def main(args, configs):
    if args.mode == 'train':
        train(args, configs)
    elif args.mode == 'test':
        # evaluate(args, model_path=os.path.join(args.model_dir, args.method, 'epoch_2650.pth'), configs=configs)
        bagging(args, model_path_list=[
            (os.path.join(args.model_dir, args.method, 'nn_92692.pth'), 'nn'),
            (os.path.join(args.model_dir, args.method, 'cnn_92329.pth'), 'cnn'),
            # (os.path.join(args.model_dir, args.method, 'pcd_91623.pth'), 'pcd'),
            (os.path.join(args.model_dir, args.method, 'simplenn_92948.pth'), 'simplenn'),
        ], configs=configs)
    else:
        raise NotImplemented