import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn import metrics
# import earthpy as et
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.ar_model import AutoReg as AR
import statsmodels.api as sm

from scipy.stats import pearsonr
data = pd.read_csv("daily_covid_cases.csv")
cases= data["new_cases"]

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

# using train data to find the heuristic value for the optimal number of lags up to the condition on autocorrelation
lags =1
trainr= pd.DataFrame(train.ravel())
print(trainr)
svalues=trainr.shift(lags)
print(svalues)
# svalues is the shifted values
corr, _ = pearsonr(trainr.iloc[lags:,0], svalues.iloc[lags:,0])
t= 2/(np.sqrt(len(trainr)))
print(t,corr)


# find the value
while(abs(corr)>t):
    lags+=1
    svalues = trainr.shift(lags)
    corr, _ = pearsonr(trainr.iloc[lags:,0], svalues.iloc[lags:,0])

print(lags)




window = lags # The lag=78
model = AR(train, lags=window)
model_fit = model.fit()  # fit/train the model
coef = model_fit.params  # Get the coefficients of AR model
# using these coefficients walk forward over time steps in test, one step each time
history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()  # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    yhat = coef[0]  # Initialize to w0
    for d in range(window):
        yhat += coef[d + 1] * lag[window - d - 1]  # Add other values
    obs = test[t]
    predictions.append(yhat)  # Append predictions to compute RMSE later
    history.append(obs)  # Append actual

mse = metrics.mean_squared_error(test, predictions)
# mse = sklearn.metrics.mean_squared_error(test, predictions)
rmse = (math.sqrt(mse)/np.mean(test))*100

mape = metrics.mean_absolute_percentage_error(test, predictions)

print(rmse)
print(mape*100)



