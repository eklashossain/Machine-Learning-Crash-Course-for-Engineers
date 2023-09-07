import numpy as np

# MSE function
def MSE(y, y_hat):

  y, y_hat = np.array(y), np.array(y_hat)           # Converting python list into numpy array
  dif = np.subtract(y, y_hat)                       # Subtraction operation
  squared_dif = np.square(dif)                      # Squaring the terms

  return np.mean(squared_dif)                       # Taking the mean of squared terms

# MAE function
def MAE(y, y_hat):

  y, y_hat = np.array(y), np.array(y_hat)

  return np.mean(np.abs(y - y_hat))                 # Taking the mean of the absolute values

# Huber Loss function
def huber_loss(y, y_hat, delta=1.0):

  y, y_hat = np.array(y), np.array(y_hat)
  huber_mse = 0.5*np.square(y - y_hat)              # MSE part of Huber Loss
  huber_mae = delta*(np.abs(y -y_hat) - 0.5*delta)  # MAE part of Huber Loss

  # Taking the mean of conditional error values
  return np.mean(np.where(np.abs(y-y_hat) <= delta, huber_mse, huber_mae))


y = [1.08, 1.2, 1.4, 2.1, 1.9, 7, 2.9]
y_hat = [0.7, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1]

mse = MSE(y=y, y_hat=y_hat)
mae = MAE(y=y, y_hat=y_hat)
huber = huber_loss(y=y, y_hat=y_hat, delta=1.35)
print("MSE = {}\nMAE = {}\nHuber Loss = {}".format(mse, mae, huber))