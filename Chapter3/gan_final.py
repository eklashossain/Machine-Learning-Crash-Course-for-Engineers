#https://gitee.com/nj520/PyTorch-GAN/blob/master/implementations/sgan/sgan.py
#Paper: https://arxiv.org/abs/1606.01583
import numpy as np
import math
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

#Loading dataset
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
dataset = datasets.MNIST('./mnist', download=True, train=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

#Variables
latent_dim = 100
img_size = 32
num_epochs = 3
batch_size = 64
num_classes = 10

#Defining generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

#Defining discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
                              nn.Conv2d(1, 16, 3, 2, 1), 
                              nn.LeakyReLU(0.2, inplace=True), 
                              nn.Dropout2d(0.25),

                              nn.Conv2d(16, 32, 3, 2, 1), 
                              nn.LeakyReLU(0.2, inplace=True), 
                              nn.Dropout2d(0.25),
                              nn.BatchNorm2d(32, 0.8),
                              
                              nn.Conv2d(32, 64, 3, 2, 1), 
                              nn.LeakyReLU(0.2, inplace=True), 
                              nn.Dropout2d(0.25),
                              nn.BatchNorm2d(64, 0.8),
                              
                              nn.Conv2d(64, 128, 3, 2, 1), 
                              nn.LeakyReLU(0.2, inplace=True), 
                              nn.Dropout2d(0.25),
                              nn.BatchNorm2d(128, 0.8)
                            )

        ds_size = img_size // 2 ** 4 # The height and width of downsampled image

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, num_classes + 1), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

#Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

#Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

#Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

#  Training
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]
        
        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        fake_aux_gt = Variable(torch.LongTensor(batch_size).fill_(num_classes), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(torch.FloatTensor))
        labels = Variable(labels.type(torch.LongTensor))

        ###Train Generator

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        ###Train Discriminator

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.numpy(), fake_aux.data.numpy()], axis=0)
        gt = np.concatenate([labels.data.numpy(), fake_aux_gt.data.numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        # Calculate discriminator mnist accuracy
        d_acc_mnist = np.mean(np.argmax(real_aux.data.numpy(), axis=1) == labels.data.numpy())
        d_acc_fake = np.mean(np.argmax(fake_aux.data.numpy(), axis=1) == fake_aux_gt.data.numpy())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss_real: %f, loss_fake: %f, acc_mnist: %d%%, acc_fake: %d%%] [G loss: %f]"
             % (epoch, num_epochs, i, len(dataloader), d_real_loss.item(), d_fake_loss.item(), 100*d_acc_mnist, 100*d_acc_fake, g_loss.item())
        )


