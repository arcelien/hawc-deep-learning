import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
	def __init__(self, z_dim, imgdim, d=128):
		super(Generator, self).__init__()
		self.z_dim = z_dim
		self.deconv1 = nn.ConvTranspose2d(z_dim, d*8, 3, 1, 0)
		self.deconv1_bn = nn.BatchNorm2d(d*8)
		self.deconv2 = nn.ConvTranspose2d(  d*8, d*4, 3, 2, 1) 
		self.deconv2_bn = nn.BatchNorm2d(d*4)
		self.deconv3 = nn.ConvTranspose2d(  d*4, d*2, 4, 2, 1)
		self.deconv3_bn = nn.BatchNorm2d(d*2)
		self.deconv4 = nn.ConvTranspose2d(  d*2, d*1, 4, 2, 1)
		self.deconv4_bn = nn.BatchNorm2d(d)
		self.deconv5 = nn.ConvTranspose2d(  d*1, 1*1, 4, 2, 1)

	# weight_init
	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	# forward method
	def forward(self, input):
		x = F.leaky_relu(self.deconv1(input))
		x = F.leaky_relu(self.deconv1_bn(self.deconv1(input)))
		x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)))
		x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)))
		x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)))
		x = F.tanh(self.deconv5(x))				
		return x

class Discriminator(nn.Module):
	def __init__(self, imgdim, d=128):
		super(Discriminator, self).__init__()
		self.d = d
		self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
		self.conv2 = nn.Conv2d(d, d*2, 4, 1, 1)
		self.conv2_bn = nn.BatchNorm2d(d*2)
		self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
		self.conv3_bn = nn.BatchNorm2d(d*4)
		self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
		self.conv4_bn = nn.BatchNorm2d(d*8)
		self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
		self.linear3 = nn.Linear(d*4*9*9, 1)

	# weight_init
	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	# forward method
	def forward(self, input):
		x = F.leaky_relu(self.conv1(input), 0.2)
		x = F.leaky_relu(self.conv2(x), 0.2)
		x = F.leaky_relu(self.conv3(x), 0.2)
		x = x.view(-1, self.d*4 * 9 * 9)
		x = F.sigmoid(self.linear3(x))
		
		return x