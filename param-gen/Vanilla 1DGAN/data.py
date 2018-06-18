import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.functional as F

# pytorch dataset for features generated from gamma data
class Data(Dataset):
	def __init__(self, train, split_size=0.8, normalize=False, conditional=False):
		assert 0 < split_size < 1
		# self.dataset = np.load("gamma_data_nosparse.npy")
		self.dataset = np.load("../gamma_data.npy")
		# self.dataset[:, 0] = np.log(self.dataset[:, 0]) # Just for SimEvent.nEMParticles
		# self.dataset[:, 4] = np.log(self.dataset[:, 4]) # Just for SimEvent.nHit
		self.len = self.dataset.shape[0]
		if train:
			self.len = int(self.len * split_size)
			self.dataset = self.dataset[:self.len]
		else:
			self.len = self.len - int(self.len * split_size)
			self.dataset = self.dataset[self.len:]
		if normalize:
			self.dataset -= [ 2.6437490e+00,  4.6791139e+00,  7.8025818e+01,  4.1586936e-01,
	   -5.8107176e-03,  6.5650093e+01,  2.5367000e+02,  5.3349188e+02]
			self.dataset /= [5.6474817e-01, 8.0267268e-01, 6.1667343e+01, 2.0911543e-01,
	   1.8022212e+00, 9.4777672e+01, 9.4920013e+01, 3.2656406e+02]
	def __getitem__(self, item):
		assert self.len > item
		vec = self.dataset[item]
		return (vec,0.)

	def __len__(self):
		return self.len

	def get_mean_std(self):
		return np.mean(self.dataset, axis=0), np.std(self.dataset, axis=0)


if __name__ == "__main__":
	print(Data(True, normalize=False).get_mean_std())
	print(Data(True, normalize=True).get_mean_std())

