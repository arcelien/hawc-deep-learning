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
			self.dataset -= [ 2.6437490e+00,  4.6791139e+00,  7.8025818e+01,  4.1586936e-01,
	   -5.8107176e-03,  6.5650093e+01,  2.5367000e+02,  5.3349188e+02,
		7.7266512e+00,  2.3883596e+01,  1.7998697e+02]
			self.dataset /= [5.6474817e-01, 8.0267268e-01, 6.1667343e+01, 2.0911543e-01,
	   1.8022212e+00, 9.4777672e+01, 9.4920013e+01, 3.2656406e+02,
	   1.7147831e+00, 1.1555794e+01, 1.0398858e+02]
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

