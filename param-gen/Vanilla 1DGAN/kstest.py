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
	def __init__(self, Dimension, nh):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(nn.Linear(Dimension, nh),
								 nn.ReLU(),
								 nn.Linear(nh, nh),
								 nn.ReLU(),
								 nn.Linear(nh, 1),
								 nn.Sigmoid())
	def forward(self, x):
		return self.net(x)

class Generator(nn.Module):
	def __init__(self, z_dim, Dimension, nh):
		super(Generator, self).__init__()
		self.net = nn.Sequential(nn.Linear(z_dim, nh),
								 nn.LeakyReLU(),
								 nn.Linear(nh, nh),
								 nn.LeakyReLU(),
								 nn.Linear(nh, Dimension))
	def forward(self, x):
		return self.net(x)
## Declare Hyperparameters  ##
Dimension  = 8      			# Number of gaussians to replicate
z_dim      = 50					# Size of latent layer
batch_size = 60000				# Number of points in dataset to sample (60112+15029 max size of the current dataset)
sample_size= batch_size * 15 	   # Number of points to sample from generator (unbounded)
alpha      = .9999999999				# Confidence interval for the ks-test
means   = [2.6437070e+00, 4.6785073e+00, 7.7988869e+01, 4.1661051e-01, -1.4959634e-02, 6.5947525e+01, 2.5352551e+02, 2.3246317e+00]  # List of means for the loaded data
stddevs = [0.5642389, 0.80296326, 61.57711, 0.2091742, 1.8062998, 94.742744, 94.72378, 0.9968804]									 # List of standard deviations for the loaded data
logs    = [0, 1, 0, 0, 0, 0, 0, 1]	
# Training data
testset    = Data(train=True, normalize=True)
testloader = torch.utils.data.DataLoader(dataset=testset, shuffle=True, batch_size=batch_size)
print(len(testset))
# Create Models
G = torch.load('saved/Gepoch10000')

real_samples = next(iter(testloader))[0]
z_ 			 = torch.randn(sample_size, z_dim).cuda()
fake_samples = G(z_)

real_samples, fake_samples = real_samples.numpy(), fake_samples.cpu().detach().numpy()

# Calculations
c_a = np.sqrt(-0.5 * np.log(alpha / 2.0))		# confidence interval c(a)
print(c_a)
compare = c_a * np.sqrt((sample_size + batch_size)/(sample_size * batch_size))
print(compare)

print("___________________")
for i in range(real_samples.shape[1]):
	realhist = (real_samples[:,i]*stddevs[i])+means[i]
	fakehist = (fake_samples[:,i]*stddevs[i])+means[i]
	if (logs[i] == 1):
		realhist = np.exp(realhist)
		fakehist = np.exp(fakehist)
	Distance, p = ks_2samp(realhist, fakehist)
	print("Distribution", str(i), "{0:.4f}, {1:.4f}".format(round(Distance, 5), round(compare, 5)))