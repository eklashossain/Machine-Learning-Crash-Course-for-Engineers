# -------------------------Torch Modules-------------------------
from __future__ import print_function
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader as DL
import torchvision.transforms as Trans
import torchvision.datasets as ds
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt;
# ---------------------------Variables---------------------------
# batch size
BATCH_SIZE = 128
Iterations = 20
feature_dims = 2

# ------Commands to Download and Prepare the MNIST Dataset-------
train_transform = Trans.Compose([
    Trans.ToTensor(),
])
test_transform = Trans.Compose([
    Trans.ToTensor(),
])  # no normalization for Autoencoder training

# train dataset
train_dataloader = DL(ds.MNIST('./mnist', train=True,
                               download=True,
                               transform=train_transform),
                      batch_size=BATCH_SIZE, shuffle=True)
# test dataset
test_dataloader = DL(ds.MNIST('./mnist', train=False,
                              transform=test_transform),
                     batch_size=BATCH_SIZE, shuffle=False)


# -------------------Defining the Autoencoder--------------------
class Encoder(nn.Module):
    def __init__(self, feature_dims):
        # Encoder part of the Autoencoder whcih projects x to a latent space z
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, feature_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, feature_dims):
        # Decoder part of the latent space which project the intermediate features back to reconstruct the image
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(feature_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class Autoencoder(nn.Module):
    def __init__(self, feature_dims):
        # combining the encoer and decoder to create the auto encoder
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(feature_dims)
        self.decoder = Decoder(feature_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# defining Autoencoder model
model = Autoencoder(feature_dims)

# defining which paramters to train only the CNN model parameters
optimizer = torch.optim.Adam(model.parameters())


# -------------------Training of Autoencoder---------------------
# Train baseline classifier on clean data
def train(model, optimizer, epoch):
    model.train()  # setting up for training
    for id_batch, (data, target) in enumerate(
            train_dataloader):  # data contains the image and target contains the label = 0/1/2/3/4/5/6/7/8/9
        optimizer.zero_grad()  # setting gradient to zero
        output = model(data)  # forward
        loss = ((output - data) ** 2).sum()  # MSE loss
        loss.backward()  # back propagation here pytorch will take care of it
        optimizer.step()  # updating the weight values
        if id_batch % 100 == 0:
            print('Epoch No: {} [ {:.0f}% ]   \tLoss: {:.3f}'.format(
                epoch, 100. * id_batch / len(train_dataloader), loss.item()))



## training the Autoencoder
for i in range(Iterations):
    train(model, optimizer, i)  # train Function


# plotting function
def plot(model, data_loader):
    for i, (x, y) in enumerate(data_loader):
        z = model.encoder(x)
        z = z.detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.xlabel("Value of Feature 1")
        plt.ylabel("Value of Feature 2")
        plt.savefig("./results/features.png")


# plotting the latent space feature
plot(model, train_dataloader)