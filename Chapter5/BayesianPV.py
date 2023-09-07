from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy import sqrt
import pandas as pd
import numpy as np


# --------------------------Reading Data-------------------------
data = pd.read_csv('./data/PV.csv',header=0)
X = data.iloc[:, 0].values  # values converts it into a numpy array
Y = data.iloc[:, 1].values  
xtrain, xtest, ytrain, ytest=train_test_split(X, Y, test_size=0.1, shuffle=False)


# -------------------------Data Processing-----------------------
n_order = 3
Xtrain = np.vander(xtrain, n_order + 1, increasing=True)
Xtest = np.vander(xtest, n_order + 1, increasing=True)


# -----------------------Setting Parameter-----------------------
reg = BayesianRidge(tol=1e-18, fit_intercept=False, compute_score=True)


# -------------------------Fit and Predict-----------------------
reg.set_params(alpha_init=0.1, lambda_init=1e-15)
reg.fit(Xtrain, ytrain)
ymean = reg.predict(Xtest)


# ----------------------------Plotting---------------------------
plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(xtest, ytest, color="blue", label="Test data")
plt.scatter(xtrain, ytrain, s=50, alpha=0.5, label="Training data")
plt.plot(xtest, ymean, color="red", label="Predicted data")
plt.legend()


# ----------------------Fitting on Full Data---------------------
n_order = 2
Xfull = np.vander(X, n_order + 1, increasing=True)
Xpred = np.array([2025, 2030, 2035, 2040])
XPred = np.vander(Xpred, n_order + 1, increasing=True)


# Setting parameter
bay = BayesianRidge(tol=1e-18, fit_intercept=False, compute_score=True)


# Fit & predict
bay.set_params(alpha_init=0.1, lambda_init=1e-30)
bay.fit(Xfull, Y)
Yfull = bay.predict(XPred)


# Plotting
plt.subplot(1, 2, 2)
plt.scatter(X, Y, s=50, alpha=0.5, label="Training data")
plt.plot(Xpred, Yfull, color="red", label="Predicted data")
plt.legend()
plt.show()