# -*- coding: utf-8 -*-
# @author: nosaka

# confirmed to work with python2, torch==1.4
# confirmed to work with python3, torch==1.11

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from model import PointNet


######## main
NUM_POINTS = 512
NUM_LABELS = 3

pred_net = PointNet(NUM_POINTS, NUM_LABELS)
pred_net.load_state_dict(torch.load("./weight/net_test.prm"))
pred_net.eval()

_in = np.load("./dataset/pointcloud/pcd_8.npy")
_in = torch.from_numpy(_in)
_in = torch.tensor(_in).float()

_out = pred_net(_in)
print("OutPut:  ", _out)
