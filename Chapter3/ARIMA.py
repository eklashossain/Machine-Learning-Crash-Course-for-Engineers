import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.metrics import mean_squared_error


# -------------------------Reading Data--------------------------
df=pd.read_csv('./data/MaunaLoaDailyTemps.csv',index_col='DATE',parse_dates=True)
df=df['AvgTemp'].dropna()


# -----------------------Splitting Dataset-----------------------
size = int(len(df)*0.80)
train, test = df[0:size], df[size:len(df)]


# --------------------Training with 80% Data---------------------
model=ARIMA(train,order=(1,0,5))
model_fit=model.fit()
model_fit.summary()


# -------------Prediction & Setting Index for Plot---------------
rng = pd.date_range(start='2017-12-30',end='2018-12-29')
prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
prediction.index = rng


# -------------------------Ploting Data--------------------------
plt.figure(figsize=(12,6))
prediction.plot(label="Prediction", legend=True)
test.plot(label="Test Data", legend=True)
plt.show()


# --------------------------Evaluation---------------------------
mean=test.mean()
print('Mean:',mean)
rmse = sqrt(mean_squared_error(test, prediction))
print('Test RMSE: %.3f' % rmse)


# -------------------Training with Full Data---------------------
full_model = ARIMA(df,order=(1,0,5))
full_model_fit = full_model.fit()


# ------------------Prediction & Setting index-------------------
rng2 = pd.date_range(start='2018-12-30', end='2019-02-28', freq='D')
full_prediction = full_model_fit.predict(start=len(df),end=len(df)+2*30,typ='levels').rename('ARIMA Predictions')
full_prediction.index = rng2


# --------------------Plotting with Main Data--------------------
df.plot(figsize=(12,6), label="Full data", legend=True)
full_prediction.plot(figsize=(12,6), label="Full prediction", legend=True)

plt.show()