# -*- coding: utf-8 -*-
# @author: nosaka

# python2 torch==1.4.0で動作確認

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class NonLinear(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NonLinear, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.linear = nn.Linear(self.input_channels,
                                self.output_channels)
        self.relu = nn.ReLU(inplace = True)
        self.bn1d = nn.BatchNorm1d(self.output_channels)

    def forward(self, input_data):
        out = self.linear(input_data)
        out = self.relu(out)
        out = self.bn1d(out)
        return out

class MaxPool(nn.Module):
    def __init__(self, num_channels, num_points):
        super(MaxPool, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points

        self.main = nn.MaxPool1d(self.num_points)

    def forward(self, input_data):
        out = input_data.view(-1, self.num_channels,
                                  self.num_points)
        out = self.main(out)
        out = out.view(-1, self.num_channels)
        return out

class PointNet(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.linear_3_64 = NonLinear(3, 64)
        self.linear_64_64 = NonLinear(64, 64)
        self.linear_64_128 = NonLinear(64, 128)
        self.linear_128_1024 = NonLinear(128, 1024)
        self.mp = MaxPool(1024, self.num_points)
        self.linear_1024_512 = NonLinear(1024, 512)
        self.dp = nn.Dropout(p = 0.3)
        self.linear_512_256 = NonLinear(512, 256)
        self.linear_256_labels = NonLinear(256, self.num_labels)

    def forward(self, input_data):
        out = self.linear_3_64(input_data)
        out = self.linear_64_64(out)
        out = self.linear_64_64(out)
        out = self.linear_64_128(out)
        out = self.linear_128_1024(out)
        out = self.mp(out)
        out = self.linear_1024_512(out)
        out = self.dp(out)
        out = self.linear_512_256(out)
        out = self.dp(out)
        out = self.linear_256_labels(out)
        return out

