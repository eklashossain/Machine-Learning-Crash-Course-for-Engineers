# -------------------------Torch Modules-------------------------
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torch.nn import init
import torch.optim as optim
from torchvision.datasets import ImageFolder as IF
from torchvision import models
import torch.nn.functional as F
import torchvision
from torchvision import transforms as Trans
from torch.utils.data import DataLoader as DL
from torch.utils.data import random_split
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.optim import SGD
# ---------------------------Variables---------------------------
bs = 128 #Batch Size
learning_rate = 0.005
Iterations = 5
CUDA_av = 1 # set to 1 for GPU training


# ----------------Prepare the German Sign Dataset----------------
# Define the transformations.
# To begin with, we shall keep it minimum
# Only resizing the images and converting them to PyTorch tensors

data_transforms = Trans.Compose([
    Trans.Resize([112, 112]),
    Trans.ToTensor(),
    ])


# Create data loader for training and validation

train_directory = "../input/gtsrb-german-traffic-sign/Train"
train_dataset = IF(root = train_data_path, transform = data_transforms)

# Divide data into training and validation (0.8 and 0.2)
ratio = 0.8
n_train_examples = int(len(train_dataset) * ratio)
n_val_examples = len(train_dataset) - n_train_examples

train_dataset, validation_data = random_split(train_dataset, [n_train_examples, n_val_examples])

print(f"Training dataset samples: {len(train_dataset)}")
print(f"Validation dataset samples: {len(validation_data)}")

train_dataloader = DL(train_dataset, shuffle=True, batch_size = bs)
test_dataloader = DL(validation_data, shuffle=True, batch_size = bs)


# -------------------Defining a ResNet-18 Model------------------
class ModifiedBlock(nn.Module):
    expansion_factor = 1
    def __init__(self, input_channels, output_channels, stride=1):
        super(ModifiedBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)

        self.shortcut_connection = nn.Sequential()
        if stride != 1 or input_channels != self.expansion_factor*output_channels:
            self.shortcut_connection = nn.Sequential(
                nn.Conv2d(input_channels, self.expansion_factor*output_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion_factor*output_channels)
            )

    def forward(self, x):
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        out += self.shortcut_connection(x)
        out = F.relu(out)
        return out


class ModifiedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=43):
        super(ModifiedResNet, self).__init__()
        self.input_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(4608, num_classes)

    def _make_layer(self, block, output_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.input_channels, output_channels, stride))
            self.input_channels = output_channels * block.expansion_factor
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out


def ModifiedResNet18():
    return ModifiedResNet(ModifiedBlock, [2, 2, 2, 2])


# defining CNN model
CNN_Model = ModifiedResNet18()
if CUDA_av == 1:
    CNN_Model = CNN_Model.cuda()
## Loss function
loss_criterion = torch.nn.CrossEntropyLoss() # pytorch's cross entropy loss function
if CUDA_av == 1:
    loss_criterion = loss_criterion.cuda()
# definin which paramters to train only the CNN model parameters
optimizer = SGD(CNN_Model.parameters(),learning_rate)

# defining the training function
# Train baseline classifier on clean data
def train_model(CNN_Model, optimizer,loss_criterion,epoch_no):
    CNN_Model.train() # setting up for training
    for id, (input_images, labels) in enumerate(train_dataloader): # data contains the image and target contains the label = 0/1/2/3/4/5/6/7/8/9
        if CUDA_av == 1:
            input_images, labels = input_images.cuda(), labels.cuda()
        optimizer.zero_grad() # setting gradient to zero
        output = CNN_Model(input_images) # forward
        loss = loss_criterion(output, labels) # loss computation
        loss.backward() # back propagation here pytorch will take care of it
        optimizer.step() # updating the weight values
        if id % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_no, id * len(input_images), len(train_dataloader.dataset),
                100. * id / len(train_dataloader), loss.item()))


# to evaluate the model
## validation of test accuracy
def test_model(CNN_Model, loss_criterion, val_loader, epoch_no):
    CNN_Model.eval()
    test_loss = 0
    correct_flag = 0

    with torch.no_grad():
        for id, (input_images, labels) in enumerate(val_loader):
            if CUDA_av == 1:
                input_images, labels = input_images.cuda() ,labels.cuda()
            output = CNN_Model(input_images)
            test_loss += loss_criterion(output, labels).item()
            pred = output.max(1, keepdim=True)[1]
            correct_flag += pred.eq(labels.view_as(pred)).sum().item() # if pred == labels then correct_flag +=1

    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct_flag, val_loader.sampler.__len__(),
        100. * correct_flag / val_loader.sampler.__len__() ))


## training the CNN
for i in range(Iterations):
    train_model(CNN_Model, optimizer,loss_criterion,i)
    test_model(CNN_Model, loss_criterion, test_dataloader, i) #Testing the the current CNN