import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader as DL
from torch.utils.data import TensorDataset as ds
# ---------------Read the dataset & Preprocess-------------------
df_class = pd.read_csv("./data/Dataset.csv")

df_trans = df_class.iloc[:, 2:].T
df_norm = (df_trans - df_trans.min()) / (df_trans.max() - df_trans.min())


df = pd.concat([df_class.iloc[:, :2].T, df_norm]).T


encoder = LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])

X = df.drop(['Class'], axis=1)
y = df['Class']


signals = torch.from_numpy(X.values).float()
signals = signals.unsqueeze(1)
target = torch.tensor(y, dtype=torch.long)


# -------------------Splitting the dataset-----------------------
X_train_test, X_test, y_train_test, y_test = train_test_split(signals, target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=42)


# ------------------------Define model---------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * ((signal_length // 2) - 1), 4)
        self.fc2 = nn.Linear(4, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Set the hyperparameters
learn_rate = 0.001
bs = 1 #batch_size
no_of_epochs = 320
signal_length = X.shape[1]
num_classes = len(encoder.classes_)


# ----------------------Initialize model-------------------------
model = CNNModel()

loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

# Set the device to use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


training_dataset = ds(X_train, y_train)
training_loader = DL(training_dataset, batch_size=bs, shuffle=True)

validation_dataset = ds(X_val, y_val)
validation_loader = DL(validation_dataset, batch_size=bs)

test_dataset = ds(X_test, y_test)
test_loader = DL(test_dataset, batch_size=bs)
# Lists to store epoch and test accuracy
epoch_list = []
test_accuracy_list = []


# ------------------------Training loop--------------------------
for epoch in range(no_of_epochs):
    model.train()
    running_loss = 0.0
    for data, target in training_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct_flag = 0
    total = 0

    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = loss_criterion(outputs, target)
            val_loss += loss.item()
            _, predicted_value = torch.max(outputs.data, 1)
            total += target.size(0)
            correct_flag += (predicted_value == target).sum().item()

    # Store epoch number and test accuracy
    epoch_list.append(epoch + 1)
    test_accuracy_list.append((correct_flag / total) * 100)

    print(f"Epoch No {epoch+1:3d}: Training Loss: {running_loss/len(training_loader):.4f},Validation Loss: {val_loss/len(validation_loader):.4f},\n              Validation Accuracy: {(correct_flag/total)*100:.2f}%")


# -------------------------Evaluation----------------------------
model.eval()
testing_loss = 0.0
correct_flag = 0
total = 0
predictions = []
true_labels = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = loss_criterion(outputs, target)
        testing_loss += loss.item()
        _, predicted_value = torch.max(outputs.data, 1)
        total += target.size(0)
        correct_flag += (predicted_value == target).sum().item()
        predictions.extend(predicted_value.cpu().numpy())
        true_labels.extend(target.cpu().numpy())

print(f"Testing Loss: {testing_loss / len(test_loader):.4f}, Testing Accuracy: {(correct_flag / total) * 100:.2f}%")

predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Calculate confusion matrix
cm = confusion_matrix(true_labels, predictions)


# --------------------------Plotting-----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()