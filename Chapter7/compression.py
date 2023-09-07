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
from torchvision import datasets as ds
from torchvision import transforms as Trans
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader as DL
# ---------------------------Variables---------------------------
# for Normalization
mean = [0.5]
std = [0.5]

# batch size
bs =128 #Batch Size 
Iterations = 20
learn_rate = 0.01

# compresion hyper-paramters
Quantized_bit = 8 # quantization bit
Lasso_penalty = 0.000001 # lasso penalty on weight
Thresholds = 0.005 # threshold


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
                  batch_size=bs, shuffle=True)


test_dataloader = DL(ds.MNIST('./mnist', train=False,
                          transform=test_transform),
                 batch_size=bs, shuffle=False)


# -------------------------Defining CNN--------------------------
# Model Definition

#quantization function
class _Quantize(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, step):         
        ctx.step = step.item()
        output = torch.round(input/ctx.step) ## quantized output
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step ## Straight through estimator
        return grad_input, None
                
quantize1 = _Quantize.apply

class quantized_conv(nn.Conv2d):
    def __init__(self,nchin,nchout,kernel_size,stride,padding=0,bias=False):
        super().__init__(in_channels=nchin,out_channels=nchout, kernel_size=kernel_size, padding=padding, stride=stride, bias=False) 
        "this function manually changes the original pytorch convolution function into a quantized weight convolution"   
        
    def forward(self, input):
        self.N_bits = Quantized_bit - 1
        step = self.weight.abs().max()/((2**self.N_bits-1))
       
        QW = quantize1(self.weight, step)
        
        return F.conv2d(input, QW*step, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
    

class quantized_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        "this function manually changes the original pytorch Linear function into a quantized weight with Linear value"   
        
    def forward(self, input):
       
        self.N_bits = Quantized_bit - 1
        step = self.weight.abs().max()/((2**self.N_bits-1))
        
        QW = quantize1(self.weight, step)
        
        return F.linear(input, QW*step, self.bias)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = quantized_conv(1, 32, 3, 1)
        self.conv2 = quantized_conv(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = quantized_linear(9216, 128)
        self.fc2 = quantized_linear(128, 10)

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

# definin which paramters to train only the CNN model parameters
optimizer = torch.optim.SGD(model.parameters(),learn_rate)


## Lasso weight penalty
def lasso_p(var):
    return var.abs().sum()

# defining the training function
# Train baseline classifier on clean data
def train(model, optimizer,criterion,epoch): 
    model.train() # setting up for training
    lasso_penalty = 0
    for id, (data, target) in enumerate(train_dataloader): # data contains the image and target contains the label = 0/1/2/3/4/5/6/7/8/9
        optimizer.zero_grad() # setting gradient to zero
        output = model(data) # forward
        loss = criterion(output, target) # loss computation

        ## iterating all the layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                lasso_penalty += lasso_p(module.weight.data)  # penalty on the weight


        loss += lasso_penalty * Lasso_penalty
        loss.backward() # back propagation here pytorch will take care of it
        optimizer.step() # updating the weight values
        if id % 100 == 0:
            print('Epoch No: {} [ {:.0f}% ]   \tLoss: {:.6f}'.format(
                epoch, 100. * id / len(train_dataloader), loss.item()))



# to evaluate the model
## validation of test accuracy
def test(model, criterion, val_loader, epoch):    
    model.eval()
    test_loss = 0
    correct = 0  
    
    with torch.no_grad():
        for id, (data, target) in enumerate(val_loader):

            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # if pred == target then correct +=1
        
    test_loss /= len(val_loader.dataset) # average test loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))


## training the CNN 
for i in range(Iterations):
    train(model, optimizer,criterion,i)

    # pruning certain weights
    ## iterating all the layers
    layer_count = 1
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            # pruning weights below a threshold = Thresholds * maximum weight value at that layer
                module.weight.data[module.weight.data.abs() < Thresholds* module.weight.data.abs().max()] = 0 

                # percentage of weight pruned = no of weights equal to zero / total weights * 100
                weight_pruned =  module.weight.data.view(-1)[module.weight.data.view(-1) == 0].size()[0]/module.weight.data.view(-1).size()[0]*100
                print("Percentage of weights pruned at Layer " + str(layer_count) + ":\t" + str(weight_pruned) + "%")
                layer_count += 1

    test(model, criterion, test_dataloader, i) #Testing the the current CNN