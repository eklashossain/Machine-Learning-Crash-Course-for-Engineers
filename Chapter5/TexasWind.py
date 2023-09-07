# Dataset: https://www.kaggle.com/datasets/pravdomirdobrev/texas-wind-turbine-dataset-simulated
# ---------------Import the necessary libraries------------------
import pandas as pd; import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --------------------Import the dataset-------------------------
df = pd.read_csv("./data/TexasTurbine.csv")
df.set_index("Time stamp", inplace=True)
print(df.head())


# -------------------Define X and y values-----------------------
X = df.drop(columns="System power generated | (kW)")
y = df["System power generated | (kW)"]


# --------------------Split the dataset--------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=None, shuffle=False)

print("X Train shape:", X_train.shape)
print("X Test shape:", X_test.shape)
print("Y Train shape:", y_train.shape)
print("Y Test shape:", y_test.shape)


# --------------------Create a RFR model-------------------------
RFR = RandomForestRegressor()


# ---------------------Train the model---------------------------
RFR.fit(X_train, y_train)
train_preds = RFR.predict(X_train)
test_preds = RFR.predict(X_test)


# ------Print the model score, train RMSE, and test RMSE---------
print("Model score:", RFR.score(X_train, y_train))
print("Train RMSE:", mean_squared_error(y_train, train_preds)**(0.5))
print("Test RMSE:", mean_squared_error(y_test, test_preds)**(0.5))


# ----------Plot the predictions and actual values---------------
plt.figure().set_figwidth(12)
X_test["RFR Prediction"] = test_preds
X_test["System power generated | (kW)"] = y_test
x_index = np.linspace(0, 250, 250)
plt.plot(x_index, X_test["RFR Prediction"].tail(250), color='red', linewidth=1, label='RFR prediction')
plt.plot(x_index, X_test["System power generated | (kW)"].tail(250), color='green', linewidth=1,label='Actual power generated')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.show()