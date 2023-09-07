import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

y = [1.08, 1.2, 1.4, 2.1, 1.9, 7, 2.9]
y_hat = [0.7, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1]

y = np.array(y)
y_hat = np.array(y_hat)

# The error metrics are determined using pre-defined library functions
mse = mean_squared_error(y, y_hat)
mae = mean_absolute_error(y, y_hat)

huber = tf.keras.losses.Huber(delta=1.35)   # Creating huber() api from Tensorflow
huber = huber(y, y_hat).numpy()

print("MSE = {}\nMAE = {}\nHuber Loss = {}".format(mse, mae, huber))