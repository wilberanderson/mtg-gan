import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm
from image_slicer import slice

import matplotlib.pyplot as plt

import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=250, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=25, help="number of classes for dataset")
parser.add_argument("--img_size_x", type=int, default=180, help="size of each image dimension")
parser.add_argument("--img_size_y", type=int, default=150, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--continue_training", type=bool, default=False, help="continue training")
parser.add_argument("--images_to_save", type=int, default=20, help="images to generate after training")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size_x, opt.img_size_y)

ordered_labels = ["Creature_B","Creature_G","Creature_R","Creature_U","Creature_W",
"Enchantment_B","Enchantment_G","Enchantment_R","Enchantment_U","Enchantment_W",
"Instant_B","Instant_G","Instant_R","Instant_U","Instant_W",
"Land_B","Land_G","Land_R","Land_U","Land_W",
"Sorcery_B","Sorcery_G","Sorcery_R","Sorcery_U","Sorcery_W"]

cuda = True if torch.cuda.is_available() else False
if cuda:
	print(torch.cuda.get_device_name())
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

		def block(in_feat, out_feat, normalize=True):
			layers = [nn.Linear(in_feat, out_feat)]
			if normalize:
				layers.append(nn.BatchNorm1d(out_feat, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*block(opt.latent_dim + opt.n_classes, 128, normalize=False),
			*block(128, 256),
			*block(256, 512),
			*block(512, 1024),
			nn.Linear(1024, int(np.prod(img_shape))),
			nn.Tanh()
		)

	def forward(self, noise, labels):
		# Concatenate label embedding and image to produce input
		gen_input = torch.cat((self.label_emb(labels), noise), -1)
		img = self.model(gen_input)
		img = img.view(img.size(0), *img_shape)
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

		self.model = nn.Sequential(
			nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 512),
			nn.Dropout(0.4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 512),
			nn.Dropout(0.4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 1),
		)

	def forward(self, img, labels):
		# Concatenate label embedding and image to produce input
		d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
		validity = self.model(d_in)
		return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# set default; change if loading weights
start_epoch = 0

if opt.continue_training:
	print("Attempting to load saved weights...")
	if os.path.isfile("/m2docs/mtg-gan/checkpoints/discriminator0.weights") and os.path.isfile("/m2docs/mtg-gan/checkpoints/generator0.weights"):
		maximum = 0
		files = os.listdir("/m2docs/mtg-gan/checkpoints/")
		for f in files:
			curr = int(f.replace("generator","").replace("discriminator","").replace(".weights",""))
			if curr > maximum:
				maximum = curr
		if os.path.isfile("/m2docs/mtg-gan/checkpoints/discriminator"+str(maximum)+".weights") and os.path.isfile("/m2docs/mtg-gan/checkpoints/generator"+str(maximum)+".weights"):
			print("Continuing from epoch {}.".format(maximum))
			print("Loading generator weights generator{}.weights...".format(maximum))
			generator = torch.load("/m2docs/mtg-gan/checkpoints/generator"+str(maximum)+".weights")
			print("Done.")
			print("Loading discriminator weights discriminator{}.weights...".format(maximum))
			print("Done.")
			discriminator = torch.load("/m2docs/mtg-gan/checkpoints/discriminator"+str(maximum)+".weights")
		else:
			print("A weights file is missing, make sure the highest epoch reached has both a generator and discriminator weights file.")

if cuda:
	print("Using cuda...")
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()

# Configure data loader
dataloader = torch.utils.data.DataLoader(
	datasets.ImageFolder(root = '/m2docs/mtg-gan/data/train', 
						 transform = transforms.Compose([
							 transforms.Resize((180,150)),
							 transforms.ToTensor()
						 ])),
	batch_size=opt.batch_size,
	shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, batches_done):
	"""Saves a grid of generated digits ranging from 0 to n_classes"""
	# Sample noise
	z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
	# Get labels ranging from 0 to n_classes for n rows
	labels = np.array([num for _ in range(n_row) for num in range(n_row)])
	labels = Variable(LongTensor(labels))
	gen_imgs = generator(z, labels)
	save_image(gen_imgs.data, "/m2docs/mtg-gan/images/%d.png" % batches_done, nrow=n_row, normalize=True)

def create_images(n_row, count):
	for i in range(count):
		"""Saves a grid of generated digits ranging from 0 to n_classes"""
		# Sample noise
		z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
		# Get labels ranging from 0 to n_classes for n rows
		labels = np.array([num for _ in range(n_row) for num in range(n_row)])
		labels = Variable(LongTensor(labels))
		gen_imgs = generator(z, labels)
		save_image(gen_imgs.data, "/m2docs/mtg-gan/gen/raw/%d.png" % i, nrow=n_row, normalize=True)

# ----------
#  Training
# ----------
d_loss_hist = []
g_loss_hist = []
for epoch in range(start_epoch,opt.n_epochs):
	running_d_loss = 0
	running_g_loss = 0
	counts = 0 
	for i, (imgs, labels) in enumerate(dataloader):
		counts+=1
		torch.cuda.empty_cache()
		if cuda:
			imgs, labels = imgs.cuda(), labels.cuda()
		batch_size = imgs.shape[0]

		# Adversarial ground truths
		valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
		fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

		# Configure input
		real_imgs = Variable(imgs.type(FloatTensor))
		labels = Variable(labels.type(LongTensor))

		# -----------------
		#  Train Generator
		# -----------------

		optimizer_G.zero_grad()

		# Sample noise and labels as generator input
		z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
		gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

		# Generate a batch of images
		gen_imgs = generator(z, gen_labels)

		# Loss measures generator's ability to fool the discriminator
		validity = discriminator(gen_imgs, gen_labels)
		g_loss = adversarial_loss(validity, valid)
		g_loss.backward()
		optimizer_G.step()

		# ---------------------
		#  Train Discriminator
		# ---------------------

		optimizer_D.zero_grad()

		# Loss for real images
		validity_real = discriminator(real_imgs, labels)
		d_real_loss = adversarial_loss(validity_real, valid)

		# Loss for fake images
		validity_fake = discriminator(gen_imgs.detach(), gen_labels)
		d_fake_loss = adversarial_loss(validity_fake, fake)

		# Total discriminator loss
		d_loss = (d_real_loss + d_fake_loss) / 2

		d_loss.backward()
		optimizer_D.step()

		running_g_loss+=g_loss.item()
		running_d_loss+=d_loss.item()
		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
			% (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
		)

		batches_done = epoch * len(dataloader) + i
		if batches_done % opt.sample_interval == 0:
			sample_image(n_row=25, batches_done=batches_done)
	
	# save states
	torch.save(generator,"/m2docs/mtg-gan/checkpoints/generator{}.weights".format(epoch))
	torch.save(discriminator,"/m2docs/mtg-gan/checkpoints/discriminator{}.weights".format(epoch))

	# write info
	d_loss_hist.append(running_d_loss/counts)
	g_loss_hist.append(running_g_loss/counts)
	print(
		"[Epoch %d/%d] [Complete] [D loss avg: %f] [G loss avg: %f]"
		% (epoch, opt.n_epochs, running_d_loss/counts, running_g_loss/counts)
	)
## Create graph
x = np.arange(opt.n_epochs)
 
fig=plt.figure()
ax=fig.add_subplot(111)
 
ax.plot(x,d_loss_hist,c='r',label='Discriminator')
ax.plot(x,g_loss_hist,c='g',label='Generator')
 
plt.legend(loc=1)
plt.savefig('/m2docs/mtg-gan/images/loss-metric.png',dpi=600)

## Save final images

save_dir = "/m2docs/mtg-gan/gen"
n_row = 25
labels = ["Creature_B","Creature_G","Creature_R","Creature_U","Creature_W","Enchantment_B","Enchantment_G","Enchantment_R","Enchantment_U","Enchantment_W","Instant_B","Instant_G","Instant_R","Instant_U","Instant_W","Land_B","Land_G","Land_R","Land_U","Land_W","Sorcery_B","Sorcery_G","Sorcery_R","Sorcery_U","Sorcery_W"]
print("Generating {} images...".format(opt.images_to_save))

create_images(n_row,opt.images_to_save)

# for label in labels:
#     if not os.path.exists(save_dir+'/'+label):
#         os.makedirs(save_dir+'/'+label)

im_list = os.listdir(save_dir+"/raw")

for image in range(opt.images_to_save):
	# split
	slices = None
	slices = slice(save_dir+"/raw/"+str(image)+".png",n_row**2)

	index = 0
	for t in slices:
		# for each, move where it goes
		shutil.move(t.filename,save_dir+"/"+labels[index%25]+"/"+t.basename+".png")
		index += 1

print("Done.")