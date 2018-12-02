from collections import OrderedDict

import torch.nn as nn


class CifarClassifer(nn.Module):
    def __init__(self, num_classes):
        super(CifarClassifer, self).__init__()

        self.fc_block = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4096, 1024)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(1024, 512)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(512, 256)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(256, 64)),
            ('relu4', nn.ReLU()),
            ('fc5', nn.Linear(64, num_classes))
        ]))

        self.fc_bn_block = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4096, 1024)),
            ('bn1', nn.BatchNorm1d(1024)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(1024, 512)),
            ('bn2', nn.BatchNorm1d(512)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(512, 256)),
            ('bn3', nn.BatchNorm1d(256)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(256, 64)),
            ('bn4', nn.BatchNorm1d(64)),
            ('relu4', nn.ReLU()),
            ('fc5', nn.Linear(64, num_classes))
        ]))

    def forward(self, x):
        return self.fc_bn_block(x)