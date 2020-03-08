import torch
import torch.nn as nn
import numpy as np
import os
import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

os.makedirs('images', exist_ok=True)
img_shape = (opt.channels, opt.img_size, opt.img_size)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(int(np.prod(img_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = x.view(x.size(0),-1)
		val = self.model(x)
		return val

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		def block(in_c, out_c, normalize = True):
			layers = [nn.Linear(in_c, out_c)]
			if normalize:
				layers.append(nn.BatchNorm1d(out_c, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*block(opt.latent_dim, 128, normalize=False),
			*block(128, 256),
			*block(256, 512),
			*block(512, 1024),
			nn.Linear(1024, int(np.prod(img_shape))),
			nn.Tanh()
		)

	def forward(self, z):
		img = self.model(torch.from_numpy(z).float())
		img = img.view(img.size(0), *img_shape)
		return img

# -----------------------------------
# Generate Discriminator & Generator
# -----------------------------------
D = Discriminator()
G = Generator()

# -------------------
# Load mnist dataset
#--------------------
os.makedirs('~/.gan/data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
	datasets.MNIST('data/mnist',
	train = True,
	download = True,
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),),
	batch_size=opt.batch_size,
	shuffle = True,
)


# ---------------------------------
# define loss function & optimizer
# ---------------------------------
BCE = torch.nn.BCELoss()
opt_g = torch.optim.Adam(G.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))
opt_d = torch.optim.Adam(D.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))

# ------------
# Training
# ------------

g_loss_mean = 0
d_loss_mean = 0

for epoch in range(opt.n_epochs):
	for i, (imgs, label) in enumerate(dataloader):
		
		# --------------------
		# Train Discriminator
		# --------------------

		z = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
		gen_imgs = G(z)
		real_loss = BCE(D(imgs), torch.ones(imgs.shape[0]))
		fake_loss = BCE(D(gen_imgs), torch.zeros(imgs.shape[0]))
		d_loss = real_loss + fake_loss
		
		d_loss_mean = d_loss_mean + d_loss

		opt_d.zero_grad()
		d_loss.backward(retain_graph=True)
		opt_d.step()

		# ------------------
		# Train Generator
		# ------------------
		g_loss = BCE(D(gen_imgs), torch.ones(imgs.shape[0]))
		
		g_loss_mean = g_loss_mean + g_loss

		opt_g.zero_grad()
		g_loss.backward()
		opt_g.step()
		


		batches_done = epoch*len(dataloader) + i
		if batches_done % 400 == 0:
			save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

		if batches_done % opt.sample_interval == 0:
			print('[Epoch %d/%d][Batch %d/%d][D loss: %f][G loss: %f]'%\
				(epoch, opt.n_epochs, i, len(dataloader), d_loss_mean.item()/opt.sample_interval, g_loss_mean.item()/opt.sample_interval))
			d_loss_mean = 0
			g_loss_mean = 0