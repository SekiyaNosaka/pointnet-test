# -*- coding: utf-8 -*-
# @author: nosaka

import os
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class MyDatasets(Dataset):
    def __init__(self, dataset_dir):
      self.dataset_dir = dataset_dir
      self.pcd_paths = self._get_pcd_paths()
      self.pose_paths = self._get_pose_paths()

    def _get_pcd_paths(self):
      pcd_paths = os.listdir(self.dataset_dir + "/pointcloud")
      return pcd_paths
    
    def _get_pose_paths(self):
      pose_paths = os.listdir(self.dataset_dir + "/pose")
      return pose_paths

    def __getitem__(self, idx):
      pcd_path = self.pcd_paths[idx]
      pose_path = self.pose_paths[idx]
      
      pcd_data = np.load(self.dataset_dir + "/pointcloud/" + pcd_path)
      pose_data = np.load(self.dataset_dir + "/pose/" + pose_path)
      return pcd_data, pose_data

    def __len__(self):
      return len(self.pcd_paths)

