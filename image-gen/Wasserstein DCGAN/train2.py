"""
Implementation of Wasserstein GANS
Data: HAWC dataset
	Labels: Event parameters
	Images: Event observations (channels: time and charge)
"""
## Imports ##
#------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.autograd as autograd
#------------------------------------------------#
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import numpy as np
from time import time
import math
import os
import sys
sys.path.append(os.getcwd())
import sklearn.datasets
#------------------------------------------------#
# import tflib as lib
# import tflib.save_images
# import tflib.mnist
# import tflib.plot
#------------------------------------------------#
from data import Dataset
# from model import Discriminator, Generator
#------------------------------------------------#
## Hyperparamters ##
batch_size  = 50			# Number of images to load at a time
epochs 		= 2000			# NUmber of interations throough training data
dim 		= 64			# Model Dimensions
z_dim       = 2000			# Latent space size
d_iter 		= 5				# Number of discriminator iterations per epoch
LAMBDA 		= 10			# Gradient penalty hyperparamter
output_dim  = 1600		
train_gpu 	= True
## Helper functions ##
normal      = lambda mu, sigma: lambda batch_size: torch.normal(mean=torch.ones(batch_size, 1)*mu, std=sigma)
exponential = lambda rate=1: lambda batch_size: torch.from_numpy(np.random.exponential(rate, batch_size).astype(np.float32)).view(-1, 1)
def sample_z(batch_size, z_dim):
	z_ = latent[0](batch_size)       	# Generate latent space vector
	for i in range(z_dim)[1:]:
		z_i = latent[i](batch_size)
		z_ = torch.cat((z_, z_i), 1)
	return z_
## Setting up ##
torch.manual_seed(1)
use_cuda = torch.cuda.is_available() and train_gpu
device = torch.device("cuda" if use_cuda else "cpu")
print('Device mode: ', device)
dd    = {'n': normal, 'e': exponential} 	# Dictionary of different distributions
latent   = [dd['n'](0, 1)] * z_dim
# ==================Definition Start======================
## Things to try: leakyrelu, batchnorm
class Generator(nn.Module):
	def __init__(self, DIM):
		super(Generator, self).__init__()

		preprocess = nn.Sequential(
			nn.Linear(z_dim, 4*4*4*DIM),
			nn.LeakyReLU(True),
		)
		block1 = nn.Sequential(
			nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
			nn.BatchNorm2d(2*DIM),
			nn.LeakyReLU(True),
		)
		block2 = nn.Sequential(
			nn.ConvTranspose2d(2*DIM, DIM, 6, stride=2, padding=1),
			nn.BatchNorm2d(DIM),
			nn.LeakyReLU(True),
		)
		deconv_out = nn.ConvTranspose2d(DIM, 1, 7, stride=2)
		self.DIM = DIM
		self.block1 = block1
		self.block2 = block2
		self.deconv_out = deconv_out
		self.preprocess = preprocess 
		self.sigmoid = nn.Sigmoid()

	def forward(self, input):
		output = self.preprocess(input)
		output = output.view(-1, 4*self.DIM, 4, 4)
		# print output.size()
		output = self.block1(output)
		# print output.size(), 'reference'
		output = output[:, :, :10, :10]
		# print output.size()
		output = self.block2(output)
		# print output.size()
		output = self.deconv_out(output)
		output = output[:, :, :40, :40]
		output = self.sigmoid(output)
		# print output.size()
		# not really sure if this is allowed but will do because I need to reshape
		return output.view(-1, output_dim)
class Discriminator(nn.Module):
	def __init__(self, DIM):
		super(Discriminator, self).__init__()

		main = nn.Sequential(
			nn.Conv2d(1, DIM, 5, stride=2, padding=2),
			# nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
			nn.LeakyReLU(True),
			nn.BatchNorm2d(DIM),
			nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
			# nn.Linear(4*4*4*DIM, 4*4*4*DIM),
			nn.LeakyReLU(True),
			nn.BatchNorm2d(2*DIM),
			nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
			# nn.Linear(4*4*4*DIM, 4*4*4*DIM),
			nn.LeakyReLU(True),
			# nn.Linear(4*4*4*DIM, 4*4*4*DIM),
			# nn.LeakyReLU(True),
			# nn.Linear(4*4*4*DIM, 4*4*4*DIM),
			# nn.LeakyReLU(True),
		)
		self.DIM = DIM
		self.main = main
		self.output = nn.Linear(100*DIM, 1)

	def forward(self, input):
		input = input.view(-1, 1, 40, 40)
		out = self.main(input)
		out = out.view(-1, 100*self.DIM)
		out = self.output(out)
		return out.view(-1)

def generate_image(frame, netG):
	noise = sample_z(realdata.shape[0], z_dim).to(device)
	samples = netG(noise)
	samples = samples.view(batch_size, 40, 40)

	samples = samples.cpu().data.numpy()

	# lib.save_images.save_images(
	# 	samples,
	# 	'tmp/samples_{}.png'.format(frame)
	# )
	fig = plt.figure(figsize=(15, 15))
	
	for i in range(16):
		plt.subplot(4, 4, i+1)
		# plt.axis('off')
		plt.imshow(samples[i], interpolation='nearest')
		# plt.tight_layout()
		plt.colorbar()
	# plt.show()
	
	plt.savefig('tmp/samples_{}.png'.format(frame))#, bbox_inches='tight', pad_inches = 0)
	plt.close()
def calc_gradient_penalty(netD, real_data, fake_data):
	alpha = torch.rand(real_data.shape[0], 1)
	alpha = alpha.expand((-1, output_dim)).to(device)
	real_data = real_data.view(-1, output_dim)
	# print real_data.shape, alpha.shape

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)

	interpolates = interpolates.to(device)
	interpolates = autograd.Variable(interpolates, requires_grad=True)

	disc_interpolates = netD(interpolates)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
	return gradient_penalty
def lazy_data_loader(trainloader):
	while True:
		for imgs, labels in trainloader:
			yield imgs, labels

## Training Data ##
trainset = Dataset('../../data', train=True, split_size=.8)  			# X is images, y represents [zenith, azimuth] labels
trainloader = lazy_data_loader(DataLoader(dataset=trainset, shuffle=True, batch_size=batch_size))
print (np.min(trainset.data), np.max(trainset.data))  # we want this number to be relatively small (i.e < 50)


netG = Generator(dim).to(device)
netD = Discriminator(dim).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
one, mone = one.to(device), mone.to(device)

for epoch in range(epochs):
	train_t = time()
	netD.train()
	netG.train()
	for _ in range(d_iter):
		realdata, reallabel = trainloader.next()
		realdata, reallabel = realdata.to(device)[:,:,:,:1], reallabel.to(device) 
		netD.zero_grad()
		# train with real
		D_real = netD(realdata)
		D_real = D_real.mean()
		# print D_real
		D_real.backward(mone)

		noise = sample_z(realdata.shape[0], z_dim).to(device)

		netG.zero_grad()
		fake = netG(noise)
		inputv = fake
		D_fake = netD(inputv)
		D_fake = D_fake.mean()
		D_fake.backward(one)

		# train with gradient penalty
		gradient_penalty = calc_gradient_penalty(netD, realdata.data, fake.data)
		gradient_penalty.backward()

		D_cost = D_fake - D_real + gradient_penalty
		Wasserstein_D = D_real - D_fake
		optimizerD.step()

	netG.zero_grad()
	netD.zero_grad()

	noise = sample_z(realdata.shape[0], z_dim).to(device)
	fake = netG(noise)
	G = netD(fake)
	G = G.mean()
	G.backward(mone)
	G_cost = -G
	optimizerG.step()

	# lib.plot.plot('./tmp/train disc cost', D_cost.cpu().data.numpy())
	# lib.plot.plot('./tmp/train gen cost', G_cost.cpu().data.numpy())
	# lib.plot.plot('./tmp/wasserstein distance', Wasserstein_D.cpu().data.numpy())
	print  "iteration", epoch, 'Dcost', D_cost.cpu().item(), "Gcost", G_cost.cpu().item(), "W distance", Wasserstein_D.cpu().item()
	if epoch % 100 == 99:
		print 'Time', time() - train_t
		generate_image(epoch, netG)


	# # Write logs every 100 iters
	# if (epoch < 5) or (epoch % 100 == 99):
	# 	lib.plot.flush()

	# lib.plot.tick()