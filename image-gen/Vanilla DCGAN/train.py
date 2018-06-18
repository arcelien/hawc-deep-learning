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
from data import Dataset
from model import Discriminator, Generator
#------------------------------------------------#

## Declare Hyperparameters ##
batch_size = 100
imgdim     = 40
z_dim      = 20
lr         = .005
lr_decay   = .9999
epochs     = 1000
train_gpu  = True
## Setup Information ##
labels 	   = {0: 'zenith', 1: 'azimuth'}
save_path  = './saved/'
## Saniy Checks ## 
# assert z_dim >= imgdim
# assert lr 	 <= .03
## Helper Functions ##
normal     = lambda mu, sigma: lambda z_dim: torch.normal(mean=torch.ones(z_dim, batch_size)*mu, std=sigma)
def sample_z(batch_size, z_dim):		# Generate latent space input
	return torch.randn((batch_size, z_dim)).view(-1, z_dim, 1, 1)
## Choose Device to run on (CPU or GPU) ##
use_cuda = torch.cuda.is_available() and train_gpu
device = torch.device("cuda" if use_cuda else "cpu")
print('Device mode: ', device)
## Training Data ##
""" 
Dims is the number of channels you want 
References the ../data/ folder  
""" 
trainset = Dataset('../../data', train=True, split_size=.8)  			# X is images, y represents [zenith, azimuth] labels
trainloader = DataLoader(dataset=trainset, shuffle=True, batch_size=batch_size)
## Create Models ##
D = Discriminator(imgdim, 100).to(device)
G = Generator(z_dim, imgdim, 200).to(device)
## Create Optimizers ##
total_loss = 0
D_optim = torch.optim.SGD(D.parameters(), lr=lr)
G_optim =torch.optim.SGD(G.parameters(), lr=lr)
bce = nn.BCELoss()
## Training Iterations ##
print('Started Training', '| Train Size = ', len(trainset))
epochs += 1
last_gen_loss = 0
for epoch in range(epochs):
	G_losses, D_losses = [], []
	Gloss, total_loss = 0, 0
	train_t = time.time()
	for realdata, reallabel in trainloader:
		D.train()
		G.train()
		realdata, reallabel = realdata.to(device), reallabel.to(device)
		D_optim.zero_grad()
		G_optim.zero_grad() 
		# We set threshold for both models so neither become too powerful
		if last_gen_loss < .85: # Train Generator
			z_ = sample_z(batch_size, z_dim).to(device).view(-1, z_dim, 1, 1)
			fake = G(z_)
			Dfake = D(fake).squeeze()
			y_real = torch.ones_like(Dfake).to(device)
			Gloss = bce(Dfake, y_real)
			Gloss.backward()
			G_optim.step()
			G_losses.append(Gloss.item())
			last_gen_loss = torch.mean(Dfake)
		if last_gen_loss > .15:
			z_ = sample_z(batch_size, z_dim).to(device).view(-1, z_dim, 1, 1)
			fake = G(z_)

			realdata = realdata.view(-1, 1, 40, 40)
			Dreal = D(realdata)
			y_real = torch.ones_like(Dreal).to(device)
			real_loss = bce(Dreal, y_real)

			Dfake = D(fake)
			y_fake = torch.zeros_like(Dfake).to(device)
			fake_loss = bce(Dfake, y_fake)
			last_gen_loss = torch.mean(Dfake)

			total_loss = real_loss + fake_loss
			total_loss.backward()
			D_optim.step()
			D_losses.append(total_loss.item())
	print("Epoch %s, G_loss: %f, D_loss: %f, time: %.3f, lr: %.3f" %
		  (epoch, Gloss, total_loss, time.time() - train_t, lr))
	if epoch % 10 == 0:
		with torch.no_grad():
			D.eval()
			G.eval()
			plt.figure()
			z_ = sample_z(batch_size, z_dim).to(device).view(-1, z_dim, 1, 1)
			fake = G(z_).squeeze()
			fake = fake.cpu().numpy()
			fake = fake[0, :, :]
			fake[fake < 0] = 0
			plt.imshow(fake)
			plt.colorbar()
			plt.savefig('./plots/'+str(epoch)+".png")
			plt.close()
	if (epoch + 1) % 50 == 0:
		torch.save(G, save_path+'G_epoch'+str(epoch))
		torch.save(D, save_path+'D_epoch'+str(epoch))
		print('Saved Model')
exit()