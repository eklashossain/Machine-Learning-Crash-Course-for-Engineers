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
# ---------------------------Variables---------------------------
mean = [0.5] # for Normalization
std = [0.5]

BATCH_SIZE =128
Iterations = 2
learning_rate = 0.001


# ------Commands to Download and Perpare the MNIST Dataset-------
train_transform = Trans.Compose([
        Trans.ToTensor(),
        Trans.Normalize(mean, std)
        ])

test_transform = Trans.Compose([
        Trans.ToTensor(),
        Trans.Normalize(mean, std)
        ])

    
train_dataloader = DL(
        ds.MNIST('./mnist', train=True, download=True,
                       transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True) # train dataset

test_dataloader = DL(
        ds.MNIST('./mnist', train=False, 
                         transform=test_transform),
        batch_size=BATCH_SIZE, shuffle=False) # test dataset


# -----------------Loading the pre-trained model-----------------
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
# loading a pre-trained model 
model = torch.load('./data/mnist.pt')


# -----------------------Defining Detector-----------------------
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*64, 200)
        self.linear2 = nn.Linear(200, 1)
        
      
    def forward(self, x):
        
        out = self.maxpool1(F.relu((self.conv1((x)))))
        out = self.maxpool2(F.relu((self.conv2(out))))
        out = out.view(out.size(0), -1)
        #print(out.size()) 
        out = F.relu((self.linear1(out)))
        out = F.sigmoid(self.linear2(out))
        return out

# defining the detector
D= Detector()



# loss function for binary classification
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)


# -----------------------FGSM Attack Code------------------------
def fgsm(model, data, target, epsilon=0.1, data_min=0, data_max=1):
        'this function takes a data and target model as input and produces a adversarial image to fool model'
        data_min = data.min()
        data_max = data.max()
        model.eval() # evaluation mode
        perturbed_data = data.clone() # data setup
        
        perturbed_data.requires_grad = True 
        output = model(perturbed_data) # ouput 
        loss = F.cross_entropy(output, target) # loss 
        
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward() ## backward loss
        
        # Again set gradient requirement to true
        perturbed_data.requires_grad = False

        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_data += epsilon*perturbed_data.grad.data.sign()
            # Adding clipping to maintain [min,max] range, default 0,1 for image
            perturbed_data.clamp_(data_min, data_max)
        return perturbed_data


# ---------------------Training and Testing----------------------
# Train the detector model
def train(model, adv, optimizer,criterion,epoch): 
    model.train() # setting up for training
    for id_batch, (data, target) in enumerate(train_dataloader): # data contains the image and target contains the label = 0/1/2/3/4/5/6/7/8/9
        optimizer.zero_grad() # setting gradient to zero

        # clean data loss
        output = model(data) # forward
        real_labels = torch.zeros(target.size()[0], 1) # clean sample labels = 0
        loss = criterion(output, real_labels) # loss computation
    
        # adversarial data loss
        adv_img = fgsm(adv, data, target, epsilon=0.3) ## geerating the adversarial samples 
        output1 = model(adv_img)
        fake_labels = torch.ones(target.size()[0], 1) # adversarialsample label =1
        
        loss1 = criterion(output1, fake_labels)
        loss = (loss+ loss1)/2 # overall loss function
        loss.backward() # back propagation here pytorch will take care of it
        
        optimizer.step() # updating the weight values
        if id_batch % 100 == 0:
            print('Epoch No: {} [ {:.0f}% ]   \tLoss: {:.3f}'.format(
                epoch, 100. * id_batch / len(train_dataloader), loss.item()))



# --------------Evaluating the Attack Success Rate---------------
# Validation of detection rate of malicious samples
def test(model, adv,val_loader, epoch):    
    model.eval()
    test_loss = 0
    correct = 0  
    
    for id_batch, (data, target) in enumerate(val_loader):
            adv_img = fgsm(adv, data, target, epsilon=0.3)
            output = model(adv_img)
          
            for i in range(data.size()[0]):
                if output [i] > 0.9:
                    correct +=1
                    
    print('\n Detection Rate:',
        100. * correct / 10000 )


# Training the detector and testing
for i in range(Iterations):
    train(D, model,optimizer,criterion,i)
    test(D, model, test_dataloader, i) # Testing the the current CNN