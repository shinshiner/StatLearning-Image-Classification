from collections import OrderedDict

import torch.nn as nn


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
            ('d1', nn.Dropout(0.2)),
            ('relu4', nn.ReLU()),
            ('fc2', nn.Linear(2048, 512)),
            ('d2', nn.Dropout(0.2)),
            ('relu5', nn.ReLU()),
            ('fc3', nn.Linear(512, 64)),
            ('d3', nn.Dropout(0.2)),
            ('relu6', nn.ReLU()),
            ('fc5', nn.Linear(64, num_classes))
        ]))

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 64, 64)

        x = self.cnn_block(x)
        x = x.view(-1, 32 * 14 * 14)

        return self.fc_block(x)