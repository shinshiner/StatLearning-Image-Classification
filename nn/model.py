from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1: 4])
        fan_out = np.prod(weight_shape[2: 4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)
    elif classname.find('LSTMCell') != -1:
        m.bias_ih.data.fill_(0.0)
        m.bias_hh.data.fill_(0.0)


class CifarClassifer(nn.Module):
    def __init__(self, num_classes):
        super(CifarClassifer, self).__init__()

        # self.fc_block = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(4096, 1024)),
        #     ('relu1', nn.ReLU()),
        #     ('fc2', nn.Linear(1024, 512)),
        #     ('relu2', nn.ReLU()),
        #     ('fc3', nn.Linear(512, 256)),
        #     ('relu3', nn.ReLU()),
        #     ('fc4', nn.Linear(256, 64)),
        #     ('relu4', nn.ReLU()),
        #     ('fc5', nn.Linear(64, num_classes))
        # ]))

        # self.fc_bn_block = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(4096, 1024)),
        #     ('bn1', nn.BatchNorm1d(1024)),
        #     ('relu1', nn.ReLU()),
        #     ('fc2', nn.Linear(1024, 512)),
        #     ('bn2', nn.BatchNorm1d(512)),
        #     ('relu2', nn.ReLU()),
        #     ('fc3', nn.Linear(512, 256)),
        #     ('bn3', nn.BatchNorm1d(256)),
        #     ('relu3', nn.ReLU()),
        #     ('fc4', nn.Linear(256, 64)),
        #     ('bn4', nn.BatchNorm1d(64)),
        #     ('relu4', nn.ReLU()),
        #     ('fc5', nn.Linear(64, num_classes))
        # ]))

        self.fc_bn_drop_block = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4096, 1024)),
            ('bn1', nn.BatchNorm1d(1024)),
            ('d1', nn.Dropout(0.5)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(1024, 512)),
            ('bn2', nn.BatchNorm1d(512)),
            ('d2', nn.Dropout(0.5)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(512, 256)),
            ('bn3', nn.BatchNorm1d(256)),
            ('d3', nn.Dropout(0.5)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(256, 64)),
            ('bn4', nn.BatchNorm1d(64)),
            ('d4', nn.Dropout(0.5)),
            ('relu4', nn.ReLU()),
            ('fc5', nn.Linear(64, num_classes))
        ]))

        self.apply(weights_init)

    def forward(self, x):
        return self.fc_bn_drop_block(x)


class CNNCifarClassifer(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifarClassifer, self).__init__()

        self.cnn_block = nn.Sequential(OrderedDict([
            ('cnn1', nn.Conv2d(1, 32, kernel_size=5, stride=3)),
            ('bn1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU()),
            ('cnn2', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('cnn3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU()),
            ('cnn4', nn.Conv2d(64, 32, kernel_size=3, stride=1)),
            ('bn4', nn.BatchNorm2d(32)),
            ('relu4', nn.ReLU()),
        ]))

        self.fc_block = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(32 * 14 * 14, 2048)),
            ('d1', nn.Dropout(0.5)),
            ('relu4', nn.ReLU()),
            ('fc2', nn.Linear(2048, 512)),
            ('d2', nn.Dropout(0.5)),
            ('relu5', nn.ReLU()),
            ('fc3', nn.Linear(512, 64)),
            ('d3', nn.Dropout(0.5)),
            ('relu6', nn.ReLU()),
            ('fc5', nn.Linear(64, num_classes))
        ]))

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 64, 64)

        x = self.cnn_block(x)
        x = x.view(-1, 32 * 14 * 14)

        return self.fc_block(x)


class PointNetfeat(nn.Module):
    def __init__(self, num_classes, num_feats=4096):
        super(PointNetfeat, self).__init__()
        self.num_feats = num_feats

        self.conv1 = torch.nn.Conv1d(1, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_feats)

        self.fc1 = nn.Linear(1024, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn22 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 1, self.num_feats)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn11(self.fc1(x)))
        x = F.relu(self.bn22(self.fc2(x)))

        return self.fc3(x)