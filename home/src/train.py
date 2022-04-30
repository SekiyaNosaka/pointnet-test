# -*- coding: utf-8 -*-
# @author: nosaka

# confirmed to work with python2, torch==1.4
# confirmed to work with python3, torch==1.11

import os import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from model import PointNet
from dataset import MyDatasets

def train_net(net, criterion, t_loader, device):
  t_losses = []
  #e_losses = []
  
  for epoch in range(EPOCH+1):
    running_loss = 0.0
    net.train()
    for i, (batch_pcd, batch_pose) in enumerate(t_loader):
      batch_pcd = batch_pcd.view(-1, 3)
      
      batch_pcd = torch.tensor(batch_pcd).float()
      batch_pose = torch.tensor(batch_pose).float()
  
      batch_pcd = batch_pcd.to(device)
      batch_pose = batch_pose.to(device)
  
      output = net(batch_pcd)
      loss = criterion(output, batch_pose)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
  
    t_loss = running_loss / i
    #e_loss = eval_net(net, criterion, e_loader, device)
    
    t_losses.append(t_loss)
    #e_losses.append(e_loss)

    if epoch % 10 == 0: # 10エポックごと
      print("Epoch: {}  Train Loss: {}".format(epoch, t_loss))
      #print("Epoch: {}  Eval  Loss: {}".format(epoch, e_loss))
  
  # lossの推移を描画
  plt.plot(t_losses, label = "train")
  plt.legend()
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.xlim(0, len(t_losses))
  plt.show()

def eval_net(net, criterion, e_loader, device):
  running_loss = 0.0
  net.eval()
  with torch.no_grad():
    for i, (batch_pcd, batch_pose) in enumerate(e_loader):
      batch_pcd = batch_pcd.view(-1, 3)
      
      batch_pcd = batch_pcd.to(device)
      batch_pose = batch_pose.to(device)
      
      output = net(batch_pcd)
      loss = criterion(output, batch_pose)

      running_loss += loss.item()
  return (running_loss / i)


############### main
EPOCH = 30
BATCH_SIZE = 2

NUM_POINTS = 512
NUM_LABELS = 3

device = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")

net = PointNet(NUM_POINTS, NUM_LABELS)
net = net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),
                       lr = 0.001)

# loading datasets for train and eval
t_dataset = MyDatasets("./dataset")
#e_dataset = MyDatasets("./dataset")

# create data_loaders for train and eval
t_loader = DataLoader(t_dataset,
                      batch_size = BATCH_SIZE,
                      shuffle = True)
#e_loader = DataLoader(e_dataset,
#                      batch_size = BATCH_SIZE,
#                      shuffle = False)

# model training
train_net(net, criterion, t_loader, device)

# save training weights
torch.save(net.state_dict(), "./weight/net.prm")
