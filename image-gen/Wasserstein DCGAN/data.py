import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.functional as F
import numpy as np
import os
# pytorch dataset for features generated from gamma data
class Dataset(Dataset):
    @staticmethod
    def load_to_dict(file, labels): 
        return {'x': np.load(file), 'y': np.load(labels)}
    @staticmethod
    def load(path, train, dims=1):
        assert dims == 1 or dims == 2
        suffix = '.npy' if dims==1 else '_2.npy'
        if train:
            train_data = [Dataset.load_to_dict(os.path.join(path, 'gamma_image_mapping_data' + suffix), os.path.join(path, 'gamma_labels' + suffix))]
            trainx = np.concatenate([d['x'] for d in train_data], axis=0)
            trainy = np.concatenate([d['y'] for d in train_data], axis=0)
            return trainx, trainy
        else:
            test_data = [Dataset.load_to_dict(os.path.join(path, 'gamma_test_image_mapping_data'+  suffix), os.path.join(path, 'gamma_test_labels' + suffix))]
            trainx = np.concatenate([d['x'] for d in test_data], axis=0)
            trainy = np.concatenate([d['y'] for d in test_data], axis=0)
            return trainx, trainy


    def __init__(self, path, train=True, split_size=0.8):   # Might need to deal with image normalization or log at some point
        assert 0 < split_size < 1
        self.data = Dataset.load(path, train)
        self.len = self.data[0].shape[0]             # Count the number of images
        if train:
            self.len = int(self.len * split_size)
            self.data, self.labels = self.data[0][:self.len], self.data[1][:self.len]
        else:
            self.len = self.len - int(self.len * split_size)
            self.data, self.labels = self.data[0][self.len:], self.data[1][self.len:]
    def __getitem__(self, item):
        assert self.len > item
        return self.data[item], self.labels[item]
    def __len__(self):
        return self.len

