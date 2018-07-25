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
data_dim   = 8      			# Number of gaussians to replicate
class_dim  = 3					# Number of conditions ({Right ascention, declination})
z_dim      = 50					# Size of latent layer
batch_size = 2048				# Number of points to compute with
nh         = 512				# Nmber of hidden nodes per layer
lr         = .03				# Lerning rate
lr_decay   = .9999				# Learning rate decay
epochs     = 4000				# Number of training iterations
use_gpu    = True				# Options (True: Single GPU, False: CPU)
## Choose CPU or GPU ##
use_cuda = torch.cuda.is_available() and use_gpu
print('gpu mode:', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
## Helper Functions ##
find_plot_dim = lambda x: math.ceil(math.sqrt(data_dim))
normal        = lambda mu, sigma: lambda batch_size: torch.normal(mean=torch.ones(batch_size, 1)*mu, std=sigma)
uniform       = lambda a, b: lambda batch_size: torch.rand((batch_size, 1)) * (b-a) + a
exponential   = lambda rate=1: lambda batch_size: torch.from_numpy(np.random.exponential(rate, batch_size).astype(np.float32)).view(-1, 1)
def sample_z(batch_size, z_dim):		# Generate latent space vector
	z_ = latent[0](batch_size)       	
	for i in range(z_dim)[1:]:
		z_i = latent[i](batch_size)
		z_ = torch.cat((z_, z_i), 1)
	return z_
## Setup Information ##
dd = {'u': uniform, 'n': normal, 'e': exponential} 	# Dictionary of different distributions
# latent   = [dd['e'](.8), dd['n'](0, 1), dd['n'](1, 2), dd['u'](0, 10)]
# latent   = [dd['n'](0, 1)] * 10 + [dd['n'](0, 1)] * 6 + [dd['e'](1)] * 4
latent = [dd['n'](0, 1)] * z_dim			# Creates a latent space entirely sampled from N(0, 1)
labels = {0: "rec.logNPE", 1: "rec.nHit", 2: "rec.nTankHit", 3: "rec.zenith", 
          4: "rec.azimuth", 5: "rec.coreX", 6: "rec.coreY", 7: "rec.CxPE40",
          8: "SimEvent.energyTrue", 9: "SimEvent.thetaTrue", 10:"SimEvent.phiTrue"}
means   = [2.6447110e+00, 4.6803799e+00, 7.8092407e+01, 4.1693807e-01, -8.8998480e-03, 6.5917976e+01, 2.5366927e+02,  2.3263862e+00, 7.7275338e+00, 2.3912493e+01, 1.7977229e+02] # List of means for the loaded data
stddevs = [0.5619153, 0.801478, 61.452538, 0.20918544, 1.8056167, 94.12724, 94.01841, 0.99879366, 1.7095443, 11.550515, 103.78041]			   									  # List of standard deviations for the loaded data
logs    = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]													 																				  # Keep track of variables where we applied log	
## Sanity Checks ##
assert z_dim >= data_dim
assert z_dim == len(latent)
# Training data
trainset = Data(train=True, normalize=True)
trainloader = torch.utils.data.DataLoader(dataset=trainset, shuffle=True, batch_size=batch_size)
# Create Models
D = Discriminator(class_dim, data_dim, nh).to(device)
G = Generator(z_dim, class_dim, data_dim, nh).to(device)
# Create Optimizers
total_loss = 0
D_optim = torch.optim.SGD(D.parameters(), lr=lr)
G_optim =torch.optim.SGD(G.parameters(), lr=lr)
bce = nn.BCELoss()
# Training Iteration
epochs += 1
for epoch in range(epochs):
	G_losses, D_losses = [], []
	Gloss, total_loss = 0, 0
	train_t = time.time()
	test_inp, test_label = None, None  			# Using this variable for the plots to ensure that its size is `batch_size`
	for real_data, real_label in trainloader:
		real_data, real_label = real_data.to(device), real_label.to(device)
		if test_inp is None:
			test_inp = real_data
			test_label = real_label

		D.train()
		G.train()
		
		D_optim.zero_grad()
		G_optim.zero_grad()
		if epoch % 1 == 0: # Train Generator
			z_ = sample_z(real_data.shape[0], z_dim).to(device)
			fake = G(z_, real_label)
			Dfake = D(fake, real_label)
			y_real = torch.ones_like(Dfake).to(device)
			Gloss = bce(Dfake, y_real)
			Gloss.backward()
			G_optim.step()
			G_losses.append(Gloss.item())
		# Train Discriminator
		real = real_data

		z_ = sample_z(real_data.shape[0], z_dim).to(device)

		fake = G(z_, real_label)

		Dreal = D(real, real_label)
		y_real = torch.ones_like(Dreal).to(device)
		real_loss = bce(Dreal, y_real)

		Dfake = D(fake, real_label)
		y_fake = torch.zeros_like(Dfake).to(device)
		fake_loss = bce(Dfake, y_fake)

		total_loss = real_loss + fake_loss
		total_loss.backward()
		D_optim.step()
		D_losses.append(total_loss.item())

	print("Epoch %s, G_loss: %f, D_loss: %f, time: %.3f, lr: %.3f" %
		  (epoch, Gloss, total_loss, time.time() - train_t, lr))
	# Just in case overfitting causes mode collapse
	lr *= lr_decay
	# Visualize the output every 10 epochs
	if epoch % 10 == 0:
		with torch.no_grad():
			D.eval()
			G.eval()
			samples = []
			real = test_inp
			z_ = sample_z(real.shape[0], z_dim).to(device)
			generated = G(z_, test_label)
			# print(generated.size())

			samples.extend(generated.cpu().numpy())
			samples = np.array(samples)
			real = real.cpu().numpy()

			print('Real Means = ', [np.mean(real[:,i]) for i in range(data_dim)])
			print('Fake Means = ', [np.mean(samples[:,i]) for i in range(data_dim)])


			x = find_plot_dim(data_dim)
			y = find_plot_dim(data_dim)
			# y = max(data_dim - x, 1)
			fig = plt.figure(figsize=(16, 16))
			for i in range(data_dim):
				plt.subplot(x, y, i+1)
				plt.title(labels[i])
				realhist = (real[:,i]*stddevs[i])+means[i]
				fakehist = (samples[:,i]*stddevs[i])+means[i]
				if (logs[i] == 1):
					realhist = np.exp(realhist)
					fakehist = np.exp(fakehist)
				bins = np.histogram(np.hstack((realhist, fakehist)), bins=30)[1]
				plt.hist(realhist, bins, alpha=1, label='real')
				plt.hist(fakehist, bins, alpha=.5, label='fake')
			plt.legend(loc='upper right')
			plt.savefig('paramGANplots/hist'+str(epoch)+".png")
			plt.close(fig)
	test_inp = None
	test_label = None
	# Visualize the output every 100 epochs
	if epoch % 100 == 0:
		with torch.no_grad():
			D.eval()
			G.eval()
			samples = []
			test_size = 6000
			testset = Data(train=False, normalize=True)				# Load the test set
			testloader = torch.utils.data.DataLoader(dataset=testset, shuffle=True, batch_size=test_size)
			for real_data, real_label in testloader:   # Only load first batch
				real = real_data
				break
			z_ = sample_z(test_size, z_dim).to(device)
			# print(z_.size())
			real_label = real_label.to(device)
			generated = G(z_, real_label)
			print(generated[0, 3:5].cpu().numpy(), real_label[0:1].cpu().numpy())

			samples.extend(generated.cpu().numpy())
			samples = np.array(samples)
			real = real.cpu().numpy()

			# print('Real Means = ', [np.mean(real[:,i]) for i in range(data_dim)])
			# print('Fake Means = ', [np.mean(samples[:,i]) for i in range(data_dim)])

			x = find_plot_dim(data_dim)
			y = find_plot_dim(data_dim)
			fig = plt.figure(figsize=(16, 16))
			for i in range(data_dim):
				plt.subplot(x, y, i+1)
				plt.title(labels[i])
				realhist = (real[:,i]*stddevs[i])+means[i]
				fakehist = (samples[:,i]*stddevs[i])+means[i]
				if (logs[i] == 1):
					realhist = np.exp(realhist)
					fakehist = np.exp(fakehist)
				bins = np.histogram(np.hstack((realhist, fakehist)), bins=30)[1]
				plt.hist(realhist, bins, alpha=1, label='real')
				plt.hist(fakehist, bins, alpha=.5, label='fake')
			plt.legend(loc='upper right')
			plt.savefig('paramGANplots/hist'+str(epoch)+".png")
			plt.close(fig)
	if epoch % 1000 == 0:
		torch.save(G, "./saved/Gepoch"+str(epoch))
		torch.save(D, "./saved/Depoch"+str(epoch))
exit()
