# This code belongs to the paper
#
# Sebastian Neumayer and Fabian Altekrüger (2024)
# Stability of Data-Dependent Ridge-Regularization for Inverse Problems
#
# Please cite the paper, if you use the code.
# The file is a modified version from 
# 
# A. Goujon, S. Neumayer, P. Bohra, S. Ducotterd, and M. Unser
# A Neural-Network-based Convex Regularizer for Inverse Problems
# IEEE Transactions on Computational Imaging, 9:781–795, 2023
# (https://github.com/axgoujon/convex_ridge_regularizer)

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
import os

random.seed(1)

class BSD500(Dataset):
    '''
    defines dataset of BSD500 
    input: coil (single or multi), acceleration factor, center_fraction,
            standard deviation of the noise, data type(pd or pdfs),
            data directory of the fastMRI data
    ''' 
    def __init__(self, data_file, shuffle=True):
        super(Dataset, self).__init__()
        self.data_file = data_file
        self.dataset = None
        with h5py.File(self.data_file, 'r') as file:
            self.keys_list = list(file.keys())
            if shuffle:
                random.shuffle(self.keys_list)

    def __len__(self):
        return len(self.keys_list)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.data_file, 'r')
        data = torch.Tensor(np.array(self.dataset[self.keys_list[idx]]))
        return data

class fastMRI(Dataset):
    '''
    defines dataset of fastMRI data 
    input: coil (single or multi), data directory of the fastMRI data
    ''' 
    def __init__(self, coil, data_dir):
        super(Dataset, self).__init__()
        self.coil = coil
        self.data_dir = data_dir
        self.data_files = sorted(os.listdir(self.data_dir))

    def __getitem__(self, index):
        curr_file_dir = f'{self.data_dir}/{self.data_files[index]}'
        mask = torch.load(f'{curr_file_dir}/mask.pt',map_location='cpu')[0,:,:,:]
        if self.coil == 'multi':
            smaps = torch.load(f'{curr_file_dir}/smaps.pt',map_location='cpu')[0,:,:,:]
        else:
            smaps = torch.tensor([],device=mask.device)
        y = torch.load(f'{curr_file_dir}/y.pt',map_location='cpu')[0,:,:,:]
        x = torch.load(f'{curr_file_dir}/x_crop.pt',map_location='cpu')[0,:,:,:]
        return {'mask': mask, 'smaps': smaps, 'y': y, 'x': x}

    def __len__(self):
        return len(self.data_files)
        
