import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.functional as F

# pytorch dataset for features generated from gamma data
class Data(Dataset):
	def __init__(self, train, split_size=0.8, normalize=False):
		assert 0 < split_size < 1
		self.dataset = np.load("../../gamma_data.npy")
		print self.dataset.shape
		self.len = self.dataset.shape[0]
		if train:
			self.len = int(self.len * split_size)
			self.dataset = self.dataset[:self.len]
		else:
			self.len = self.len - int(self.len * split_size)

			self.dataset = self.dataset[self.len:]
		if normalize:
			self.dataset -= [ 2.6428657e+00,  4.6778517e+00,  7.7909584e+01,  4.1665417e-01,
       -1.0895852e-02,  6.5963097e+01,  2.5333636e+02,  2.3245733e+00,
        7.7152114e+00,  2.3917170e+01,  1.8024490e+02]
			self.dataset /= [  0.5619153 ,   0.801478  ,  61.452538  ,   0.20918544,
         1.8056167 ,  94.12724   ,  94.01841   ,   0.99879366,
         1.7095443 ,  11.550515  , 103.78041   ]
	def __getitem__(self, item):
		assert self.len > item
		vec = self.dataset[item]
		return (vec[:-3],vec[-3:])	# Last 3 elements are the coditional elements

	def __len__(self):
		return self.len

	def get_mean_std(self):
		return np.mean(self.dataset, axis=0), np.std(self.dataset, axis=0)


if __name__ == "__main__":
	print(Data(True, normalize=False).get_mean_std())
	print(Data(True, normalize=True).get_mean_std())

