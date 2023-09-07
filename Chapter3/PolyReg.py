# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Reading and initializing dataset
dataset = pd.read_csv("./data/advertising.csv")
print(dataset.head())

# Detailed info about the dataset
print(dataset.info())

# Checking the number of null data in the dataset
print(dataset.isnull().sum())

# Checking the correlation between different types of advertisement and sales
tv_corr, _ = pearsonr(dataset["TV"], dataset["Sales"])
radio_corr, _ = pearsonr(dataset["Radio"], dataset["Sales"])
news_corr, _ = pearsonr(dataset["Newspaper"], dataset["Sales"])

print("TV Correlation: {}\nRadio Correlation: {}\nNewspaper Correlation: {}\n".format(
    tv_corr, radio_corr, news_corr 
))

# whether TV data contain outliers or not from box plot
dataset["TV"].plot(kind='box', subplots=True, layout=(1,1), figsize=(8,2))
plt.show()
# String data from dataset into variable
tv = dataset["TV"]
sales = dataset["Sales"]

# Maximum value from TV data
tv_max = max(tv)
# Minimum value from TV data
tv_min = min(tv)

# Mean value from TV data
tv_mu = np.mean(tv)
# Standard deviation from TV data
tv_sigma = np.std(tv)

# Applying standard scaling on TV data
# This scaled data will be used as input data
tv = (tv-tv_mu) / (tv_sigma + 1e-6)
tv = np.array(tv)
tv = np.reshape(tv, (-1,1))


# Creating a vector of ones
# It has the same length as TV data
X_ones = np.ones_like(tv)

# The one vector is concatenated to the TV data
# The concatenation is done row-wise
# Thus the input data now contains two columns
X = np.concatenate((X_ones, tv), axis=1)

# The sales data are stored as output
y_true = np.array(sales)
y_true = np.reshape(y_true, (-1,1))

# Defining MSE function
# MSE is used as loss function here
def loss_mse(y_hat, y_true):

  loss = np.mean((y_hat - y_true)**2)

  return loss

# Function for gradient calculation
# Gradients are required for implementing the Gradient Descent algorithm
def grad(y_hat, y_true, x):

  # The expression for gradients are calculated manually
  grad_c = 2 * np.mean(y_hat - y_true)
  grad_m = 2 * np.mean((y_hat - y_true) * x)

  return [grad_c, grad_m]

# Function for implementing model training
def train(x, y_true, params, learning_rate):

  # Model prediction as y_hat
  y_hat = (params * x).sum(axis=1)
  y_hat = np.reshape(y_hat, (-1,1))

  # Loss calculation from the loss function (MSE)
  loss = loss_mse(y_hat, y_true)

  # The next two steps are required for model optimization
  # These two steps are the core learning process
  # Gradient computation
  grads = grad(y_hat, y_true, x)

  # Parameter update
  new_params = params - learning_rate*np.array(grads)

  return loss, new_params

# Splitting the dataset
# First 180 datapoints are selected as training data
X_train = X[:180]
y_train = y_true[:180]
# Last 20 datapoints are selected as test data
X_test = X[180:]
y_test = y_true[180:]

# Number of iteration or epoch
epoch = 50
# Learning rate (alpha)
alpha = 0.1

# Empty list for storing training losses at each epoch
losses = []
# Both learnable parameters are initialized as 1
params = np.ones((1,2), dtype=np.float64)

# Commencing the training of the linear regression model
# Loop is used to iterate through every epoch
for i in range(epoch):
  loss, new_params = train(X_train, y_train, params, learning_rate=alpha)
  params = new_params
  losses.append(loss)
  print("Epoch {}   Loss: {}".format(i+1, loss))

# Training loss curve visualization
fig = plt.figure()
epochs = np.arange(1, epoch+1)
plt.plot(epochs, losses)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.show()

# Calculation of training R2 score
train_pred = (params*X_train).sum(axis=1)
train_pred = np.reshape(train_pred, (-1,1))
train_r2 = r2_score(y_train, train_pred)

# Calculation of test R2 score
test_pred = (params*X_test).sum(axis=1)
test_pred = np.reshape(test_pred, (-1,1))
test_r2 = r2_score(y_test, test_pred)

print("Training R2 score: {}\nTest R2 score: {}\n".format(train_r2, test_r2))

# Subplots for linear regression output
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7))

# Plotting output on training data
x = X_train[:, 1]
y = params[:, 0] + params[:, 1]*x

ax1.scatter(tv[:180], sales[:180], label='Training Data')
ax1.plot(x, y, c='r', label='Linear Regression Model')
ax1.set(xlabel='TV Advertisement', ylabel='Sales', title='Training Output')
ax1.legend()

# Plotting output on test data
x = X_test[:, 1]
y = params[:, 0] + params[:, 1]*x

ax2.scatter(tv[180:], sales[180:], label='Test Data')
ax2.plot(x, y, c='r', label='Linear Regression Model')
ax2.set(xlabel='TV Advertisement', ylabel='Sales', title='Test Output')
ax2.legend()
plt.show()