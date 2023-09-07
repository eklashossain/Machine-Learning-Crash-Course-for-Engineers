# -------------------------Torch Modules-------------------------
from __future__ import print_function
import numpy as np
import pandas as pd
import torch.nn as nn
import math, torch
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torchvision import datasets as ds
from torchvision import transforms as Trans
from torchvision import models
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader as DL
# ---------------------------Variables---------------------------
# for Normalization
mean = [0.5]
std = [0.5]
# batch size
bs = 128
Iterations = 5
learn_rate = 0.01


# ------Commands to download and perpare the MNIST dataset-------
train_transform = Trans.Compose([
    Trans.ToTensor(),
    Trans.Normalize(mean, std)
])

test_transform = Trans.Compose([
    Trans.ToTensor(),
    Trans.Normalize(mean, std)
])

# train dataset
train_loader = DL(ds.MNIST('./mnist', train=True, download=True, transform=train_transform),
                  batch_size=bs, shuffle=True)

# test dataset
test_loader = DL(ds.MNIST('./mnist', train=False, transform=test_transform),
                 batch_size=bs, shuffle=False)


# --------------------------Defining CNN-------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# defining CNN model
model = CNN()

# Loss function
criterion = torch.nn.CrossEntropyLoss()  # pytorch's cross entropy loss function

# defining which parameters to train only the CNN model parameters
optimizer = torch.optim.SGD(model.parameters(), learn_rate)


# -----------------------Training Function-----------------------
# Train baseline classifier on clean data
def train(model, optimizer, criterion, epoch):
    model.train()  # setting up for training
    for id, (data, target) in enumerate(
            train_loader):  # data contains the image and target contains the label = 0/1/2/3/4/5/6/7/8/9
        optimizer.zero_grad()  # setting gradient to zero
        output = model(data)  # forward
        loss = criterion(output, target)  # loss computation
        loss.backward()  # back propagation here pytorch will take care of it
        optimizer.step()  # updating the weight values
        if id % 100 == 0:
            print('Train Epoch No: {} [ {:.0f}% ]   \tLoss: {:.6f}'.format(
                epoch, 100. * id / len(train_loader), loss.item()))


# -----------------------Testing Function------------------------
# validation of test accuracy
def test(model, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for id, (data, target) in enumerate(val_loader):
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # if pred == target then correct +=1

    test_loss /= len(val_loader.dataset)  # average test loss
    print('\nTest set: \nAverage loss: {:.4f}, \nAccuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__()))


# training the CNN
for i in range(Iterations):
    train(model, optimizer, criterion, i)
    test(model, criterion, test_loader)  # Testing the the current CNN


# -----------------------FGSM Attack Code------------------------
def fgsm(model, data, target, epsilon=0.1, data_min=0, data_max=1):
    'this function takes a data and target model as input and produces a adversarial image to fool model'
    data_min = data.min()
    data_max = data.max()
    model.eval()  # evaluation mode
    perturbed_data = data.clone()  # data setup

    perturbed_data.requires_grad = True
    output = model(perturbed_data)  # ouput
    loss = F.cross_entropy(output, target)  # loss

    if perturbed_data.grad is not None:
        perturbed_data.grad.data.zero_()

    loss.backward()  # backward loss

    # Again set gradient requirement to true
    perturbed_data.requires_grad = False

    with torch.no_grad():
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data += epsilon * perturbed_data.grad.data.sign()
        # Adding clipping to maintain [min,max] range, default 0,1 for image
        perturbed_data.clamp_(data_min, data_max)

    return perturbed_data


# ---------------Evaluating the Attack Success Rate--------------
def Attack(model, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0

    for id, (data, target) in enumerate(val_loader):

        adv_img = fgsm(model, data, target, epsilon=0.3)
        if id == 0:  # saving the image
            save_image(data[0:100], './results/data' + '.png', nrow=10)
            save_image(adv_img[0:100], './results/adv' + '.png', nrow=10)
        output = model(adv_img)
        test_loss += criterion(output, target).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()  # if pred == target then correct +=1

    test_loss /= len(val_loader.dataset)  # average test loss
    print('\nTest set: \nAverage loss: {:.4f}, \nAccuracy After Attack: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__()))


# Executing the attack code
Attack(model, criterion, test_loader)