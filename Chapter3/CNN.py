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
from torchvision import datasets, transforms
from torchvision import models
import torch.nn.functional as F


# ---------------------------Variables---------------------------
mean = [0.5] # for Normalization
std = [0.1]
# batch size
BATCH_SIZE =128
Iterations = 20
learning_rate = 0.01


# -------Commands to download and prepare the MNIST dataset------
train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=True, download=True,
                       transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True) # train dataset

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=False, 
                         transform=test_transform),
        batch_size=BATCH_SIZE, shuffle=False) # test dataset

# -------------------------Defining CNN--------------------------
# Pytorch official Example site: https://github.com/pytorch/examples/blob/master/mnist/main.py
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

## Loss function
criterion = torch.nn.CrossEntropyLoss() # pytorch's cross entropy loss function

# definin which paramters to train only the ANN model parameters
optimizer = torch.optim.SGD(model.parameters(),learning_rate)

# defining the training function
# Train baseline classifier on clean data
def train(model, optimizer,criterion,epoch): 
    model.train() # setting up for training
    for batch_idx, (data, target) in enumerate(train_loader): # data contains the image and target contains the label = 0/1/2/3/4/5/6/7/8/9
        optimizer.zero_grad() # setting gradient to zero
        output = model(data) # forward
        loss = criterion(output, target) # loss computation
        loss.backward() # back propagation here pytorch will take care of it
        optimizer.step() # updating the weight values
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# to evaluate the model
## validation of test accuracy
def test(model, criterion, val_loader, epoch):    
    model.eval()
    test_loss = 0
    correct = 0  
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):

            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # if pred == target then correct +=1
        
    test_loss /= len(val_loader.dataset) # average test loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))


## training the ANN 
for i in range(Iterations):
    train(model, optimizer,criterion,i)
    test(model, criterion, test_loader, i) #Testing the the current ANN