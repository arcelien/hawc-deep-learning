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
		print(self.dataset.shape)
		self.len = self.dataset.shape[0]
		if train:
			self.len = int(self.len * split_size)
			self.dataset = self.dataset[:self.len]
		else:
			self.len = self.len - int(self.len * split_size)

			self.dataset = self.dataset[self.len:]
		if normalize:
			self.dataset -= [2.6447110e+00, 4.6803799e+00, 7.8092407e+01, 4.1693807e-01, -8.8998480e-03, 6.5917976e+01, 2.5366927e+02,  2.3263862e+00, 7.7275338e+00, 2.3912493e+01, 1.7977229e+02]
			self.dataset /= [0.5619153, 0.801478, 61.452538, 0.20918544, 1.8056167, 94.12724, 94.01841, 0.99879366, 1.7095443, 11.550515, 103.78041]
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

