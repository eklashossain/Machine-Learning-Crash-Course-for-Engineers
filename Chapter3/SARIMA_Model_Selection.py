#Dataset: https://www.kaggle.com/rakannimer/air-passengers
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose as sd

# -------------------------Reading Data--------------------------
passenger_data =pd.read_csv("./data/AirPassengers.csv")
passenger_data.Month = pd.to_datetime(passenger_data.Month)
passenger_data = passenger_data.set_index("Month")
plt.plot(passenger_data.Passengers)
plt.xlabel("Year")
plt.ylabel("No of Passengers")
plt.show()


# ----------------------Check If Stationary----------------------
dftest = adfuller(passenger_data, autolag='AIC')
print('ADF Value: {0:.2f} \tP-Value: {1:.2f} \nNo of Lags: {2} \t\tNo Of Observations: {3} \nCritical Values:'.format(
    dftest[0], dftest[1], dftest[2], dftest[3]))

for i, values in dftest[4].items():
  print("\t\t\t\t",i, " :", values)


# ---------------------Seasonal Decomposition--------------------
comp = []
comp.append(passenger_data["Passengers"])

#Seasonal Decomposition
components = sd(passenger_data["Passengers"], model='additive')

trend_component = components.trend
comp.append(trend_component)

seasonal_component = components.seasonal
comp.append(seasonal_component)

residual_component = components.resid
comp.append(residual_component)


comp_names = ["Original", "Trend", "Seasonal", "Residual"]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(411 + i)
    plt.plot(comp[i], label=comp_names[i], color='red')
    plt.legend(loc=2)
plt.show()


# -------------------Finding Best Model Order--------------------
sarima_model=auto_arima(passenger_data["Passengers"],start_p=1,d=1,start_q=1,
                      max_p=5,max_q=5,m=12,
                      start_P=0,D=1,start_Q=0,max_P=5,max_D=5,max_Q=5,
                      seasonal=True,
                      error_action="ignore",
                      suppress_warnings=True,
                      stepwise=True,n_fits=50,test='adf')

print(sarima_model.summary())