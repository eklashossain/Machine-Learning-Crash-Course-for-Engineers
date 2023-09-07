#Dataset: https://www.kaggle.com/rakannimer/air-passengers
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# -------------------------Reading Data--------------------------
passenger_data =pd.read_csv("./data/AirPassengers.csv")
passenger_data.Month = pd.to_datetime(passenger_data.Month)
passenger_data = passenger_data.set_index("Month")


# -----------------------Splitting Dataset-----------------------
X_train, X_test = train_test_split(passenger_data, test_size=0.2,
                                random_state=42, shuffle=False)


# --------------------Training with 80% Data---------------------
SARIMA=SARIMAX(X_train["Passengers"],
             order=(0,1,1),
             seasonal_order=(2,1,1,12))
SARIMA_fit=SARIMA.fit()


# ------------Prediction & Plotting with Test Data---------------
trained_results=SARIMA_fit.predict(len(X_train),len(passenger_data)-1)
trained_results.plot(legend=True)
X_test["Passengers"].plot(legend=True)
plt.show()


# --------------------------Evaluation---------------------------
mean=X_test["Passengers"].mean()
print('Mean:',mean)
rmse = sqrt(mean_squared_error(X_test, trained_results))
print('Test RMSE: %.3f' % rmse)


# -------------------Training with Full Data---------------------
year_to_forecast = 5
model_full=SARIMAX(X_train["Passengers"],
                order=(0,1,1),
                seasonal_order=(2,1,1,12))
model_fit_full=model_full.fit()


# -----------------Predicting out-of-sample data------------------
forecast=model_fit_full.predict(start=len(passenger_data),
                      end=(len(passenger_data)-1)+year_to_forecast*12,
                      typ="levels")


# ------------------Plotting with Original Data-------------------
plt.figure(figsize=(12, 6))

data_sets = [(X_train, 'Training Data', 'green'),
             (X_test, 'Test Data', 'blue'),
             (trained_results, 'In-sample Forecast', 'black'),
             (forecast, 'Out-of-sample Forecast', 'red')]

for data, label, color in data_sets:
    plt.plot(data.index, data, label=label, color=color)

plt.legend(loc=2)
plt.xlabel("Time")
plt.ylabel("Passenger Count")
plt.title("Seasonal Forecasting with SARIMA")
plt.grid(True)
plt.show()