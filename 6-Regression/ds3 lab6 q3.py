import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from matplotlib.dates import DateFormatter
import sklearn
from statsmodels.tsa.ar_model import AutoReg as AR



data= pd.read_csv("daily_covid_cases.csv")
train, test = train_test_split(data, test_size=.35, random_state=42)

# importing the required module
from statsmodels.tsa.ar_model import AutoReg as AR
# Train test split
series = pd.read_csv('daily_covid_cases.csv',
parse_dates=['Date'],
index_col=['Date'],
sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

lags=  [1, 5, 10, 15 ,25]
RMSEs=[]
MAPEs=[]
for win in lags:
    window = win # The lag=5
    model = AR(train, lags=window)
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model
    #using these coefficients walk forward over time steps in test, one step each time
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1] # Add other values
        obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual

    # calculatinh rmse and mape
    mse = sklearn.metrics.mean_squared_error(test, predictions)
    rmse = (math.sqrt(mse)/np.mean(test))*100
    RMSEs.append(rmse)
    mape=sklearn.metrics.mean_absolute_percentage_error(test, predictions)*100
    MAPEs.append(mape)
    print(rmse)
    print(mape)

print(RMSEs)
print(MAPEs)

# plot rmse vs lag values
plt.bar(lags,RMSEs)
plt.xlabel("Time lag")
plt.ylabel("RMSE")
plt.show()

# plot mape vs lag values

plt.bar(lags,MAPEs)
plt.xlabel("Time lag")
plt.ylabel("MAPE")
plt.show()