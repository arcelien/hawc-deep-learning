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
	def __init__(self, Dimension, nh):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(nn.Linear(Dimension, nh),
								 nn.ReLU(),
								 nn.Linear(nh, nh),
								 nn.Tanh(),		# Was previously Tanh
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
batch_size = 2048				# Number of points to compute with
nh         = 512				# Nmber of hidden nodes per layer
lr         = .03				# Lerning rate
lr_decay   = .9999				# Learning rate decay
epochs     = 10000				# Number of training iterations (takes at least 200 epochs to start seeing progress)
use_gpu    = True				# Options (True: Single GPU, False: CPU)
## Choose CPU or GPU ##
use_cuda = torch.cuda.is_available() and use_gpu
print('gpu mode:', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
## Helper Functions ##
find_plot_dim = lambda x: math.ceil(math.sqrt(Dimension))
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
labels = {0: "rec.logNPE", 1: "log rec.nHit", 2: "rec.nTankHit", 3: "rec.zenith", 
          4: "red.azimuth", 5: "rec.coreX", 6: "rec.coreY", 7: "log rec.CxPE40"}
## Sanity Checks ##
assert z_dim >= Dimension
assert z_dim == len(latent)
# Training data
trainset = Data(train=True, normalize=True)
trainloader = torch.utils.data.DataLoader(dataset=trainset, shuffle=True, batch_size=batch_size)
# Create Models
D = Discriminator(Dimension, nh).to(device)
G = Generator(z_dim, Dimension, nh).to(device)
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
	test_inp = None  			# Using this variable for the plots to ensure that its size is `batch_size`
	for inp, _ in trainloader:
		inp = inp.to(device)
		if test_inp is None:
			test_inp = inp

		D.train()
		G.train()
		
		D_optim.zero_grad()
		G_optim.zero_grad()
		if epoch % 1 == 0: # Train Generator
			z_ = sample_z(inp.shape[0], z_dim).to(device)
			fake = G(z_)
			Dfake = D(fake)
			y_real = torch.ones_like(Dfake).to(device)
			Gloss = bce(Dfake, y_real)
			Gloss.backward()
			G_optim.step()
			G_losses.append(Gloss.item())
		# Train Discriminator
		real = inp

		z_ = sample_z(inp.shape[0], z_dim).to(device)

		fake = G(z_)

		Dreal = D(real)
		y_real = torch.ones_like(Dreal).to(device)
		real_loss = bce(Dreal, y_real)

		Dfake = D(fake)
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
			z_ = sample_z(batch_size, z_dim).to(device)
			generated = G(z_)

			samples.extend(generated.cpu().numpy())
			samples = np.array(samples)
			real = real.cpu().numpy()

			print('Real Means = ', [np.mean(real[:,i]) for i in range(Dimension)])
			print('Fake Means = ', [np.mean(samples[:,i]) for i in range(Dimension)])


			x = find_plot_dim(Dimension)
			y = find_plot_dim(Dimension)
			fig = plt.figure(figsize=(16, 16))
			for i in range(Dimension):
				plt.subplot(x, y, i+1)
				plt.title(labels[i])
				plt.hist(real[:,i]   , alpha=1, bins=30, range=(-6,6), label='real')
				plt.hist(samples[:,i], alpha=.5, bins=30, range=(-6,6), label='fake')
			plt.legend(loc='upper right')
			plt.savefig('paramGANplots/hist'+str(epoch)+".png")
			plt.close(fig)
	test_inp = None
	# Visualize the output every 100 epochs
	if epoch % 100 == 0:
		with torch.no_grad():
			D.eval()
			G.eval()
			samples = []
			test_size = 6000
			testset = Data(train=False, normalize=True)				# Load the test set
			testloader = torch.utils.data.DataLoader(dataset=testset, shuffle=True, batch_size=test_size)
			for inp, _ in testloader:   # Only load first batch
				real = inp
				break
			z_ = sample_z(test_size, z_dim).to(device)
			generated = G(z_)

			samples.extend(generated.cpu().numpy())
			samples = np.array(samples)
			real = real.cpu().numpy()

			print('Real Means = ', [np.mean(real[:,i]) for i in range(Dimension)])
			print('Fake Means = ', [np.mean(samples[:,i]) for i in range(Dimension)])

			x = find_plot_dim(Dimension)
			y = find_plot_dim(Dimension)
			fig = plt.figure(figsize=(16, 16))
			for i in range(Dimension):
				plt.subplot(x, y, i+1)
				plt.title(labels[i])
				plt.hist(real[:,i]   , alpha=1, bins=30, range=(-6,6), label='real')
				plt.hist(samples[:,i], alpha=.5, bins=30, range=(-6,6), label='fake')
			plt.legend(loc='upper right')
			plt.savefig('paramGANplots/hist'+str(epoch)+".png")
			plt.close(fig)