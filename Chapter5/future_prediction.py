# Dataset: https://www.kaggle.com/robikscube/hourly-energy-consumption

# -------------------------Torch Modules-------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset as ds
from torch.utils.data import DataLoader as DL
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------Hyper-Parameters------------------------
# to keep track of index column
tar_idx = 0
in_idx = range(5)

# Define window_size period
kw = 30

# batch size
bs = 256

# number of training iterations
iters = 5
# number of layers
layers = 3
# learning rate
lr = 0.001


# -----------------------Data Preperation------------------------
data_path = "./data/NI_hourly.csv"
filename = "NI_hourly.csv"


# prepares the dataset for training and testing
def data_prep(data_values, kw, in_idx, tar_idx):

    '''
    This function creates a sliding window of the data and each slice will be a potential input to the model with a target label
    '''

    # creates input and label for training and testing
    inputs = np.zeros((len(data_values) - kw, kw, len(in_idx)))
    target = np.zeros(len(data_values) - kw)

    # this loop creates input containing samples from kw window and target value
    for i in range(kw, len(data_values)):

        inputs[i - kw] = data_values[i - kw:i, in_idx]
        target[i - kw] = data_values[i, tar_idx]

    inputs = inputs.reshape(-1, kw, len(in_idx))
    target = target.reshape(-1, 1)
    print(inputs.shape, target.shape)

    return inputs, target


# This dictionary re-scale the target during evaluation
target_scalers = {}
trainX = []
testX = {}
testY = {}


# --------Reading the File: North Ilinois Power Load (MW)--------
df = pd.read_csv(f'{data_path}', parse_dates=[0])

#reading each input
df['Hours'] = df['Datetime'].dt.hour
df['Day_of_weeks'] = df['Datetime'].dt.dayofweek
df['Months'] = df['Datetime'].dt.month
df['Day_of_year'] = df['Datetime'].dt.dayofyear
df = df.sort_values("Datetime").drop('Datetime', axis=1)

# scaling the input
scale = MinMaxScaler()
target_scale = MinMaxScaler()
data_values = scale.fit_transform(df.values)

# target scaling for evaluation
target_scale.fit(df.iloc[:, tar_idx].values.reshape(-1, 1))
target_scalers[filename] = target_scale

# prepare dataset
inputs, target = data_prep(data_values, kw, in_idx=in_idx, tar_idx=tar_idx)


testing_percent = int(0.2*len(inputs)) # 20 percent will be used for testing

if len(trainX) == 0:
    trainX = inputs[:-testing_percent]
    trainY = target[:-testing_percent]
else:
    trainX = np.concatenate((trainX, inputs[:-testing_percent]))
    trainY = np.concatenate((trainY, target[:-testing_percent]))
testX[filename] = (inputs[-testing_percent:])
testY[filename] = (target[-testing_percent:])


# prepare train data
train_load = ds(torch.from_numpy(trainX), torch.from_numpy(trainY))
train_dataloader = DL(train_load, shuffle=True, batch_size=bs, drop_last=True)

# checking GPU availability
is_cuda = torch.cuda.is_available()

# If GPU available then train on GPU
device = torch.device("cuda") if is_cuda else torch.device("cpu")





# --------------------Defining the LSTM Model--------------------

class LSTMModel(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, layers):

        "LSTM model"

        super(LSTMModel, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.layers = layers

        self.lstm = nn.LSTM(input_dimension, hidden_dimension, layers,
                            batch_first=True, dropout=0.1)
        # lstm layer
        self.fc = nn.Linear(hidden_dimension, output_dimension)


    def forward(self, x, h):
        # forward path
        out, h = self.lstm(x, h)
        out = self.fc(F.relu(out[:, -1]))
        return out, h

    def init_hidden(self, bs):
        w = next(self.parameters()).data

        h = (w.new(self.layers, bs, self.hidden_dimension).zero_().to(device),
                  w.new(self.layers, bs, self.hidden_dimension).zero_().to(device))
        return h


# ------------------------Train Function-------------------------

def train_model(train_dataloader, learning_rate, hidden_dimension, layers, num_of_epoch):

    ## training parameters
    input_dimension = next(iter(train_dataloader))[0].shape[2]
    output_dimension = 1

    model = LSTMModel(input_dimension, hidden_dimension, output_dimension, layers)
    model.to(device)

    #   Mean Squared Error
    loss_criterion = nn.MSELoss()
    Adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()  # set to train mode


    # Training start
    for iteration in range(1, num_of_epoch+1):

        h = model.init_hidden(bs)
        avg_loss_cal = 0.

        for data_values, target in train_dataloader:

            h = tuple([e.data for e in h])

            # as usual
            model.zero_grad()

            out, h = model(data_values.to(device).float(), h)
            loss = loss_criterion(out, target.to(device).float())

            # Perform backward differentiation
            loss.backward()
            Adam_optimizer.step()
            avg_loss_cal += loss.item()

        print(f"Epoch [{iteration}/{num_of_epoch}]: MSE: {avg_loss_cal/len(train_dataloader)}")
    return model
# Defining the model
model = train_model(train_dataloader, lr, 256,  layers,iters)


# --------------------------Test Phase---------------------------
def test_model(model, testX, testY, target_scalers):
    model.eval()
    predictions = []
    true_values = []

    # get data of test data for each state
    for filename in testX.keys():
        inputs = torch.from_numpy(np.array(testX[filename]))
        target = torch.from_numpy(np.array(testY[filename]))

        h = model.init_hidden(inputs.shape[0])

        # predict outputs
        out, h = model(inputs.to(device).float(), h)

        predictions.append(target_scalers[filename].inverse_transform(
            out.cpu().detach().numpy()).reshape(-1))

        true_values.append(target_scalers[filename].inverse_transform(
            target.numpy()).reshape(-1))

    # Merge all files
    f_outputs = np.concatenate(predictions)
    f_targets = np.concatenate(true_values)
    Evaluation_error = 100/len(f_targets) * np.sum(np.abs(f_outputs - f_targets) / (np.abs(f_outputs + f_targets))/2)
    print(f"Evaluation Error: {round(Evaluation_error, 3)}%")

    # list of targets/outputs for each state
    return predictions, true_values


predictions, true_values = test_model(model, testX, testY, target_scalers)


# --------------------------Visualizing--------------------------
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 10))
plt.plot(predictions[0][-100:], "-r", color="r", label="LSTM Output", markersize=2)
plt.plot(true_values[0][-100:], color="b", label="True Value")
plt.xlabel('Time (Data points)')
plt.ylabel('Energy Consumption (MW)')
plt.title(f'Energy Consumption for North Illinois state')
plt.legend()
plt.savefig('./results/load_forecasting.png')