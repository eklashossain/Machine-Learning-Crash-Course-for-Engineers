Dataset: https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set

# ----------------------Importing Modules------------------------
import numpy as np
import pandas as pd
import missingno
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# -------------------------Reading Data--------------------------

dataset = pd.read_csv("./data/PRSA_Data_Aotizhongxin_20130301-20170228.csv")


# preprocessing dataset
dataset.isnull().sum()
df = dataset.dropna()

X = df[['PM10', 'CO']]
Y = df['PM2.5']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)


# ------------------Local Outlier Factor (LOF)-------------------
model = LocalOutlierFactor(n_neighbors= 35 , contamination= 0.1)
predict = model.fit_predict(X_train)
mask = predict != -1
X_train, y_train = X_train.iloc[mask, :], y_train.iloc[mask]


model = LinearRegression()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)