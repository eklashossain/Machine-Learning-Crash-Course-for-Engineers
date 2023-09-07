import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")


# -------------------------Reading Data--------------------------
df = pd.read_csv('./data/MaunaLoaDailyTemps.csv',
                 index_col='DATE',
                 parse_dates=True).dropna()


# -------------------------Initial Plot--------------------------
df['AvgTemp'].plot(figsize=(12,6))
plt.show()


# ----------------------Check If Stationary----------------------
dftest = adfuller(df['AvgTemp'], autolag='AIC')
print('ADF Value: {0:.2f} \tP-Value: {1:.2f} \nNo of Lags: {2} \t\tNo Of Observations: {3} \nCritical Values:'.format(
    dftest[0], dftest[1], dftest[2], dftest[3]))

for i, values in dftest[4].items():
  print("\t\t\t\t",i, " :", values)


# -------------------Finding Best Model Order--------------------
arima_model = auto_arima(df['AvgTemp'], error_action="ignore",
                          stepwise=True, test='adf',
                          suppress_warnings=True)

print(arima_model.summary())