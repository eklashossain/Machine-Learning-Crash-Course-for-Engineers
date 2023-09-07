#Source: https://www.kaggle.com/esathyaprakash/electrical-fault-detection-and-classification
#Paper: https://springerplus.springeropen.com/articles/10.1186/s40064-015-1080-x


# -------------------------Torch Modules-------------------------
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import torch
from torch.nn import init
import torch.utils.data as data_utils
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F


# ---------------------------Variables---------------------------
BATCH_SIZE = 128
Iterations = 100
learning_rate = 0.01


# ------------------Commands to Prepare Dataset------------------
data_set = pd.read_csv("./data/data_fault_detection.csv")
torch.manual_seed(18)

target_value = torch.tensor(data_set["Fault_type"].values.astype(np.float32))
input_value = torch.tensor(data_set.drop(columns = ["Fault_type"]).values.astype(np.float32))

data_tensor = data_utils.TensorDataset(input_value, target_value)

train_set_size = math.floor(input_value.size()[0]*0.90)
test_set_size = input_value.size()[0]-train_set_size

train_set, test_set = torch.utils.data.random_split(data_tensor, [train_set_size, test_set_size])

train_loader = data_utils.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
test_loader = data_utils.DataLoader(dataset = test_set, batch_size = BATCH_SIZE, shuffle = True)


# --------------------------Defining ANN-------------------------
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.l1 = nn.Linear(6, 30)  
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(30, 30)
        self.l3 = nn.Linear(30, 4) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
     
        x = self.l2(x)
        x = self.relu(x)

        x = self.l2(x)
        x = self.relu(x)

        x = self.l2(x)
        x = self.relu(x)

        x = self.l2(x)
        x = self.relu(x)

        x = self.l3(x)
        x = self.sigmoid(x)
        return x

# defining ANN model
model = ANN()
## Loss function
criterion = torch.nn.CrossEntropyLoss() 

# definin which paramters to train only the ANN model parameters
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# defining the training function
def train(model, optimizer, criterion,epoch): 
    model.train() 
    total_trained_data = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() 
        output = model(data)
        loss = criterion(output, target.type(torch.LongTensor)) 
        loss.backward() 
        optimizer.step()
        total_trained_data += len(data)
        if (batch_idx !=0 and batch_idx % 30 == 0) or total_trained_data == len(train_loader.dataset):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, total_trained_data, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# ---------------------------Evaluation--------------------------
def test(model, criterion, val_loader, epoch):    
    model.eval()
    test_loss = 0
    correct = 0  
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            output = model(data)
            test_loss += criterion(output, target.type(torch.LongTensor)).item() 
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    test_loss /= len(val_loader.dataset) 
    if epoch:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, val_loader.sampler.__len__(),
            100. * correct / val_loader.sampler.__len__() ))
    else:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, val_loader.sampler.__len__(),
            100. * correct / val_loader.sampler.__len__() ))

## training the ANN 
for i in range(Iterations):
    train(model, optimizer,criterion,i)
    test(model, criterion, train_loader, i)
    test(model, criterion, test_loader, False)

def pred(data): #prediciton function for single data
    output = model(data)
    output = output.tolist()
    index = output.index(max(output))
    
    if index == 3:
        print("No fault detected.")
    else:
        string = ["SLG", "LL or LLG", "LLL or LLLG"]
        type_fault = string[index]
        print(f"Fault detected. Type of fault is {type_fault}.")

for input_data, _ in test_loader:
    for value in input_data:
        pred(value)