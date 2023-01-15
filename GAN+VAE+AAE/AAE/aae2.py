import argparse
import os
import numpy as np
import math
import itertools
import glob

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torch.autograd import Variable
from torchtoolbox.transform import Cutout

import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch_fid import fid_score


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(1024, opt.latent_dim)
        self.logvar = nn.Linear(1024, opt.latent_dim)

    def forward(self, img):
        w = self.cnn(img)
        #img_flat = img.view(img.shape[0], -1)
        img_flat = img.view(w.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 1024),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Configure data loader
os.makedirs("../../data/horse", exist_ok=True)
class MyDataset(Dataset):
    def __init__(self, filenames, transform):
        self.transform = transform
        self.filenames = filenames
        self.num = len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num


def get_dataset(dir):
    fnames = glob.glob(os.path.join(dir, '*'))

    tfm_ = [
        transforms.ToPILImage(),
        #transforms.Resize((64, 64)),  # 该任务由于需要生成32*32的马的图片，所以不需要Resize 目前由于设置原因没有注释掉
        #Cutout(),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    tfm = transforms.Compose(tfm_)
    dataset = MyDataset(fnames, tfm)
    return dataset
dir = "C:/Users/zhanglk9/PycharmProjects/pythonProject4/cifar10_horse/cifar10_horse/train"

dataloader = torch.utils.data.DataLoader(
    dataset = get_dataset(dir),
    batch_size=opt.batch_size,
    shuffle=True,
)


# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2),
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = decoder(z)
    #print(gen_imgs.shape)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------
if __name__ == '__main__':
 for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        #print(valid.shape)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        #real_imgs = real_imgs.permute(1,0,2,3)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()


        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)
        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)



        #real_loss.backward()
        #fake_loss.backward()
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
    if epoch == epoch:
            z = Variable(Tensor(np.random.normal(0, 1, (1000, opt.latent_dim))))
            gen_imgs = decoder(z)
            for j in range(1000):
                save_image(gen_imgs[j].data, "fakes/%d.png" % (j + 1), normalize=True)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #data_path = dir
    #fake_path = "C:/Users/zhanglk9/PycharmProjects/pythonProject4/venv/fakes"

    #fid = fid_score.calculate_fid_given_paths([str(data_path), str(fake_path)], 128, torch.device(device),
    #                                                      2048)
    #print('Fid score:' + str(fid))

