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
mean = [0.5] # For Normalization
std = [0.1]

BATCH_SIZE =128 # Batch size
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


# -------------------------Defining ANN--------------------------
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.l1 = nn.Linear(784, 100)  # input layer 784 for mnist and 100 neurons hidden layer
        self.relu = nn.ReLU() # activation function
        self.l3 = nn.Linear(100, 10) ## from 100 neuron hidden layer to output 10 layer for 10 digits
        
    def forward(self, x):
        x = torch.flatten(x, 1) ## making the 28 x 28 images into a 784 dimension input
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

# defining ANN model
model = ANN()

## Loss function
criterion = torch.nn.CrossEntropyLoss() # pytorch's cross entropy loss function

# Definin which paramters to train only the ANN model parameters
optimizer = torch.optim.SGD(model.parameters(),learning_rate)

# Defining the training function
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


# To evaluate the model
# Validation of test accuracy
def test(model, criterion, val_loader, epoch):    
    model.eval()
    test_loss = 0
    correct = 0  
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):

            output = model(data)
            test_loss += criterion(output, target).item() # Sum up batch loss
            pred = output.max(1, keepdim=True)[1] # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # If pred == target then correct +=1
        
    test_loss /= len(val_loader.dataset) # Average test loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))


# Training the ANN
for i in range(Iterations):
    train(model, optimizer,criterion,i)
    test(model, criterion, test_loader, i) # Testing the the current ANN