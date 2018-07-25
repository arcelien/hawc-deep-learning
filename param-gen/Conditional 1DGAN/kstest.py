## Imports ##
#------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
import scipy.stats
import time
import math
import os
from termcolor import colored
import sys
#------------------------------------------------#
from data import Data
#------------------------------------------------#
## Models ##
class Discriminator(nn.Module):
	def __init__(self, class_dim, data_dim, nh):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(nn.Linear(class_dim+data_dim, nh),
								 nn.ReLU(),
								 nn.Linear(nh, nh//2),
								 nn.ReLU(),		
								 nn.Linear(nh//2, nh//2),
								 nn.ReLU(),
								 nn.Linear(nh//2, 1),
								 nn.Sigmoid())
	def forward(self, x, y):
		x = torch.cat([x, y], 1)
		return self.net(x)

class Generator(nn.Module):
	def __init__(self, z_dim, class_num, data_dim, nh):
		super(Generator, self).__init__()
		self.net = nn.Sequential(nn.Linear(z_dim+class_dim, nh),
								 nn.LeakyReLU(),
								 nn.Linear(nh, nh//2),
								 nn.LeakyReLU(),
								 nn.Linear(nh//2, nh//2),
								 nn.LeakyReLU(),
								 nn.Linear(nh//2, data_dim))
	def forward(self, x, y):
		x = torch.cat([x, y], 1)
		return self.net(x)
## Declare Hyperparameters  ##
Dimension  = 8      			# Number of gaussians to replicate
z_dim      = 50					# Size of latent layer
batch_size = 15000				# Number of points in dataset to sample (60112 max size of the current dataset)
sample_size= batch_size  	    # Number of points to sample from generator (unbounded)
alpha      = .01				# Confidence interval for the ks-test
# Training data
testset    = Data(train=False, normalize=True)
testloader = torch.utils.data.DataLoader(dataset=testset, shuffle=True, batch_size=batch_size)
# Create Models
G = torch.load('saved/Gepoch2000')

real = next(iter(testloader))
real_samples, real_labels = real[0], real[1]
z_ 			 = torch.randn(sample_size, z_dim).cuda()
fake_samples = G(z_, real_labels.cuda())

real_samples, fake_samples = real_samples.numpy(), fake_samples.cpu().detach().numpy()


# Calculations
c_a = np.sqrt(-0.5 * np.log(alpha / 2.0))		# confidence interval c(a)
print(c_a)
compare = c_a * np.sqrt((sample_size + batch_size)/(sample_size * batch_size))
print(compare)

print("___________________")
for i in range(real_samples.shape[1]):
	Distance = ks_2samp(fake_samples[:, i], real_samples[:,i])[0]
	print("{0:.3f}".format(round(Distance, 5)), end="   ")
	if Distance > compare:
		print(colored("✔", "green"))
	else:
		print(colored("✘", "red"))
		