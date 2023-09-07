# Dataset: https://www.kaggle.com/competitions/vsb-power-line-fault-detection/data
# Download train.parquet and metadata_train.csv files

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# ---------------Read the dataset & Preprocess-------------------
# Download from the dataset link and replace the path
subset_train = pq.read_pandas('../input/vsb-power-line-fault-detection/train.parquet',
               columns=[str(i) for i in range(5000)]).to_pandas()
# Read half of the data among 800000 samples
subset_train = subset_train.iloc[200000:600000, :]
subset_train.info()

metadata_train = pd.read_csv('../input/vsb-power-line-fault-detection/metadata_train.csv')
metadata_train.info()

# Reduce the sample sizes to stay within memory limits
S_decimation = subset_train.iloc[0:25000:8, :]
small_subset_train = S_decimation
small_subset_train = small_subset_train.transpose()
small_subset_train.index = small_subset_train.index.astype(np.int32)
train_dataset = metadata_train.join(small_subset_train, how='right')

### Uncomment the following to train on the full dataset
# subset_train = subset_train.transpose()
# subset_train.index = subset_train.index.astype(np.int32)
# train_dataset = metadata_train.join(subset_train, how='right')


# ----------Separating positive and negative samples-------------
positive_samples = train_dataset[train_dataset['target'] == 1]
positive_samples = positive_samples.iloc[:, 3:]

print("positive_samples data shape: " + str(positive_samples.shape) + "\n")
positive_samples.info()

y_train_pos = positive_samples.iloc[:, 0]
X_train_pos = positive_samples.iloc[:, 1:]
scaler = StandardScaler()
scaler.fit(X_train_pos.T)  # Normalize the data set
X_train_pos = scaler.transform(X_train_pos.T).T

negative_samples = train_dataset[train_dataset['target'] == 0]
negative_samples = negative_samples.iloc[:, 3:]

print("negative_samples data shape: " + str(negative_samples.shape) + "\n")
negative_samples.info()

y_train_neg = negative_samples.iloc[:, 0]
X_train_neg = negative_samples.iloc[:, 1:]
scaler.fit(X_train_neg.T)
X_train_neg = scaler.transform(X_train_neg.T).T


# -------------------Splitting the dataset-----------------------
X_train_pos, X_valid_pos, y_train_pos, y_valid_pos = train_test_split ( X_train_pos,
                                    y_train_pos,
                                    test_size=0.3,
                                    random_state=0,
                                    shuffle=False)
X_train_neg, X_valid_neg, y_train_neg, y_valid_neg = train_test_split ( X_train_neg,
                                    y_train_neg,
                                    test_size=0.3,
                                    random_state=0,
                                    shuffle=False)

print("X_train_pos data shape: " + str(X_train_pos.shape))
print("X_train_neg data shape: " + str(X_train_neg.shape))
print("y_train_pos data shape: " + str(y_train_pos.shape))
print("y_train_neg data shape: " + str(y_train_neg.shape))

print("\nX_valid_pos data shape: " + str(X_valid_pos.shape))
print("X_valid_neg data shape: " + str(X_valid_neg.shape))
print("y_valid_pos data shape: " + str(y_valid_pos.shape))
print("y_valid_neg data shape: " + str(y_valid_neg.shape))


# -----------Combine positive and negative samples---------------
# Keeping the the samples balanced
# 550 and 270 is used to make sure
# a correct ratio of positive and negative samples
def combine_pos_and_neg_samples(pos_samples, neg_samples, y_pos, y_neg):
    X_combined = np.concatenate((pos_samples, neg_samples))
    y_combined = np.concatenate((y_pos, y_neg))
    combined_samples = np.hstack((X_combined, y_combined.reshape(y_combined.shape[0], 1)))
    np.random.shuffle(combined_samples)
    return combined_samples


train_samples = combine_pos_and_neg_samples(X_train_pos,
                                X_train_neg[:550, :],
                                y_train_pos,
                                y_train_neg[:550])
X_train = train_samples[:, :-1]
y_train = train_samples[:, -1]

print("X_train data shape: " + str(X_train.shape))
print("y_train data shape: " + str(y_train.shape))
print("train_samples data shape: " + str(train_samples.shape))

validation_samples = combine_pos_and_neg_samples(X_valid_pos,
                                     X_valid_neg[:270, :],
                                     y_valid_pos,
                                     y_valid_neg[:270])
X_valid = validation_samples[:, :-1]
y_valid = validation_samples[:, -1]

print("\nX_valid data shape: " + str(X_valid.shape))
print("y_valid data shape: " + str(y_valid.shape))
print("validation_samples data shape: " + str(validation_samples.shape))


# -----Reshape training and validation data for input layer------
X_train = X_train.reshape(-1, 1, 3125)
X_valid = X_valid.reshape(-1, 1, 3125)
print("X_train data shape: " + str(X_train.shape))
print("X_valid data shape: " + str(X_valid.shape))
print("y_train data shape: " + str(y_train.shape))
print("y_valid data shape: " + str(y_valid.shape))

X_valid = X_valid.astype(np.float32)
y_valid = y_valid.astype(np.float32)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
print("Type of data: " + str(X_train.dtype))


# ---------------------Normalize feature-------------------------
print("Total samples in train dataset: " + str(np.sum(y_train)))
print("Total samples in validation dataset: " + str(np.sum(y_valid)))

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu) / std


X_valid = feature_normalize(X_valid)
X_train = feature_normalize(X_train)


# ---------------------Define dataloader-------------------------
class torch_Dataset(Data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        data = (self.x[index], self.y[index])
        return data

    def __len__(self):
        return len(self.y)


def training_loader(train_data, batch_size, shuffle):
    return torch.utils.data.DataLoader(train_data, batch_size, shuffle)

Train_dataset = torch_Dataset(X_train, y_train)
test_dataset = torch_Dataset(X_valid, y_valid)
train_loader = training_loader(Train_dataset, batch_size=1, shuffle=False)
test_loader = training_loader(test_dataset, batch_size=1, shuffle=False)


# ------------------------Define model---------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 779, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


model = CNNModel()
print(model)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()


# ------------------------Training loop--------------------------
for epoch in range(10):
    losses = []
    for data, target in train_loader:
        output = model(data)
        target = target.view([1, 1])
        loss = criterion(output, target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}: loss {sum(losses) / len(losses)}")


# -------------------------Evaluation----------------------------
def validate(model, train_loader, val_loader):
    accdict = {}
    for name, loader in [("train dataset", train_loader), ("test dataset  ", val_loader)]:
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.float()
                outputs = model(imgs)
                predicted = torch.max(outputs)
                if (predicted > 0.5):
                    fault_detected = 1
                else:
                    fault_detected = 0
                total += labels.shape[0]
                correct += int((fault_detected == labels).sum())
                predictions.append(fault_detected)
                true_labels.append(round(labels.item()))


        print("Accuracy {0}: {1:.2f}(%)".format(name, 100 * (correct / total)))
        accdict[name] = correct / total

validate(model, train_loader, test_loader)