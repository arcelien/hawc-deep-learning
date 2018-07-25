import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.functional as F

# pytorch dataset for features generated from gamma data
class Data(Dataset):
	def __init__(self, train, split_size=0.8, normalize=False, conditional=False):
		assert 0 < split_size < 1
		self.dataset = np.load("../../gamma_data.npy")[:,:-3]
		# print self.dataset.shape
		self.len = self.dataset.shape[0]
		if train:
			self.len = int(self.len * split_size)
			self.dataset = self.dataset[:self.len]
		else:
			self.len = self.len - int(self.len * split_size)
			self.dataset = self.dataset[self.len:]
		if normalize:
			self.dataset -= [2.6437070e+00, 4.6785073e+00, 7.7988869e+01, 4.1661051e-01, -1.4959634e-02, 6.5947525e+01, 2.5352551e+02, 2.3246317e+00]
			self.dataset /= [0.5642389, 0.80296326, 61.57711, 0.2091742, 1.8062998, 94.742744, 94.72378, 0.9968804]
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

