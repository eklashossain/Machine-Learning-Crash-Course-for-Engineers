# -------------------------Torch Modules-------------------------
from __future__ import print_function
import numpy as np
import pandas as pd
import torch.nn as nn
import math, torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.nn import init
import torch.optim as optim
from torchvision import datasets as ds
from torchvision import transforms as Trans
from torchvision import models
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader as DL
# ---------------------------Variables---------------------------
mean = [0.5] # for Normalization
std = [0.1]
# batch size
BATCH_SIZE =128
Iterations = 20
learn_rate = 0.01


# -------Commands to download and prepare the MNIST dataset------
train_transform = Trans.Compose([
        Trans.ToTensor(),
        Trans.Normalize(mean, std)
        ])

test_transform = Trans.Compose([
        Trans.ToTensor(),
        Trans.Normalize(mean, std)
        ])

    
train_dataloader = DL(ds.MNIST('./mnist', train=True, download=True,
                           transform=train_transform),
                  batch_size=BATCH_SIZE, shuffle=True)


test_dataloader = DL(ds.MNIST('./mnist', train=False,
                          transform=test_transform),
                 batch_size=BATCH_SIZE, shuffle=False)



# -------------------------Defining CNN--------------------------
# Pytorch official Example site: https://github.com/pytorch/examples/blob/master/mnist/main.py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128,bias=False)
        self.fc2 = nn.Linear(128, 10,bias=False)

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


#defining CNN model
model = CNN()

## Loss function
loss_criterion = torch.nn.CrossEntropyLoss() # pytorch's cross entropy loss function

# definin which paramters to train only the CNN model parameters
optimizer = SGD(model.parameters(),learn_rate)

# defining the training function
# Train baseline classifier on clean data
def train(model, optimizer,train_dataloader,loss_criterion,epoch):
    model.train() # setting up for training
    for id, (data, target) in enumerate(train_dataloader): # data contains the image and target contains the label = 0/1/2/3/4/5/6/7/8/9
        optimizer.zero_grad() # setting gradient to zero
        output = model(data) # forward
        loss = loss_criterion(output, target) # loss computation
        loss.backward() # back propagation here pytorch will take care of it
        optimizer.step() # updating the weight values
        if id % 100 == 0:
            print('Epoch No: {} [ {:.0f}% ]   \tLoss: {:.6f}'.format(
                epoch, 100. * id / len(train_dataloader), loss.item()))



# to evaluate the model
## validation of test accuracy
def test(model, loss_criterion, val_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for id, (data, target) in enumerate(val_loader):

            output = model(data)
            test_loss += loss_criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # if pred == target then correct +=1

    test_loss /= len(val_loader.dataset) # average test loss
    print('\nTest set: Average loss: {:.4f},\n          Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))


## training the CNN
for i in range(Iterations):
    train(model, optimizer,train_dataloader,loss_criterion,i)
    test(model, loss_criterion, test_dataloader, i) #Testing the the current CNN

print("The model was initially trained for an MNIST dataset")
test(model, loss_criterion, test_dataloader, i)
print("The trained model has a good accuracy on MNIST")


# --------Transfer the Learning from One Domain to Another-------
# Downloading Fashion MNIST dataset
train_dataloader = DL(
        ds.FashionMNIST('./mnist', train=True, download=True,
                       transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True) # train dataset

test_dataloader = DL(
        ds.FashionMNIST('./mnist', train=False,
                         transform=test_transform),
        batch_size=BATCH_SIZE, shuffle=False) # test datase

## Testing shows the model fails on Fashion MNIST dataset
print("But fails on Fashion-Mnist dataset:")
test(model, loss_criterion, test_dataloader, i)


## No need to train the conv layer
for name,  module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        module.weight.requires_grad = False
        module.bias.requires_grad = False

# Only train the last 2 fully connected layers
for name,  module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.weight.requires_grad = True

optimizer = torch.optim.SGD(model.parameters(),0.005) ## selecting a smaller learning rate for transfer learning

## training the CNN for F-MNIST
for i in range(10):
    train(model, optimizer,train_dataloader,loss_criterion,i)
    test(model, loss_criterion, test_dataloader, i) #Testing the the current CNN


print("After fine-tuning the model on the last layers the model recovers good accuracy on Fashion MNIST as well")
test(model, loss_criterion, test_dataloader, i)