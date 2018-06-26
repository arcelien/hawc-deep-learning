## Imports ##
#------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import os
import sys
#------------------------------------------------#
from data import Data
#------------------------------------------------#
## Model ##
class Generator(nn.Module):
	def __init__(self, z_dim, class_num, data_dim, nh):
		super(Generator, self).__init__()
		self.net = nn.Sequential(nn.Linear(z_dim+class_dim, nh),
								 nn.LeakyReLU(),
								 nn.Linear(nh, nh),
								 nn.LeakyReLU(),
								 nn.Linear(nh, data_dim))
	def forward(self, x, y):
		x = torch.cat([x, y], 1)
		return self.net(x)
## Varibles ##
test_size = 100
num_dists = 8
num_conds = 3
## Testing ##
model = torch.load('./saved/Gepoch4000')  # Path to where saved generator model is 
labels = {0: "rec.logNPE", 1: "rec.nHit", 2: "rec.nTankHit", 3: "rec.zenith", 
          4: "rec.azimuth", 5: "rec.coreX", 6: "rec.coreY", 7: "rec.CxPE40",
          8: "SimEvent.energyTrue", 9: "SimEvent.thetaTrue", 10:"SimEvent.phiTrue"}
means   = [2.6447110e+00, 4.6803799e+00, 7.8092407e+01, 4.1693807e-01, -8.8998480e-03, 6.5917976e+01, 2.5366927e+02,  2.3263862e+00, 7.7275338e+00, 2.3912493e+01, 1.7977229e+02]  # List of means for the loaded data
stddevs = [0.5619153, 0.801478, 61.452538, 0.20918544, 1.8056167, 94.12724, 94.01841, 0.99879366, 1.7095443, 11.550515, 103.78041]			   									  # List of standard deviations for the loaded data
logs    = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]													 																				  # Keep track of variables where we applied log
## Loading Data ##
testset = Data(train=False, normalize=True)
testloader = torch.utils.data.DataLoader(dataset=testset, shuffle=True, batch_size=1)

cond_range = [(0, 1), (-.1, 10), (-.1, 1)]  # Choose sample ranges for all conditional variables
valid_data = []
# Only keep data that meets conditions in cond_range
for dist, cond in testloader:
	will_insert = all([cond_range[i][0] < cond[0][i] < cond_range[i][1] for i in range(cond.shape[1])])
	output = dist.tolist()[0] + cond.tolist()[0]
	if will_insert: valid_data.append(output)
valid_data = np.array(valid_data)
print valid_data.shape

# Use the commented code to visualize labels sampled from a uniform distribution instead
z_1 = torch.empty(valid_data.shape[0], 1).uniform_(cond_range[0][0], cond_range[0][1])
z_2 = torch.empty(valid_data.shape[0], 1).uniform_(cond_range[1][0], cond_range[1][1])
z_3 = torch.empty(valid_data.shape[0], 1).uniform_(cond_range[2][0], cond_range[2][1])
z_label = torch.cat([z_1, z_2, z_3], dim=1).cuda()
# z_label = torch.FloatTensor(valid_data[:,-3:]).cuda()
z_noise = torch.empty(valid_data.shape[0], 50).normal_(0, 1).cuda()  				# z_dim latent size

print z_label.shape
with torch.no_grad():	
	generated = model(z_noise, z_label)

plt.figure(figsize=(15,15))
for i in range(num_dists):
	plt.subplot(3, 3, i+1)
	plt.title(labels[i])
	# realhist = valid_data[:,i]
	# fakehist = generated[:,i]
	realhist = (valid_data[:,i]*stddevs[i])+means[i]
	fakehist = (generated[:,i]*stddevs[i])+means[i]
	if (logs[i] == 1):
		realhist = np.exp(realhist)
		fakehist = np.exp(fakehist)
	bins = np.histogram(np.hstack((realhist, fakehist)), bins=30)[1]
	plt.hist(realhist, bins, alpha=1, label='real')
	plt.hist(fakehist, bins, alpha=.5, label='fake')
	plt.legend()
plt.show()