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


from statsmodels.tsa.ar_model import AutoReg as AR

# a
# Train test split test size = 0.35
series = pd.read_csv('daily_covid_cases.csv',
parse_dates=['Date'],
index_col=['Date'],
sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# separating the cases
cases= series["new_cases"]
values = pd.DataFrame(cases.values)


window = 5 # The lag=5
model = AR(train, lags=window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print(coef)

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


# b
# Give a scatter plot between actual and predicted values.
plt.scatter(test, predictions)
plt.xlabel(" Test data")
plt.ylabel("Predicted Values")
plt.show()


# Give a line plot showing actual and predicted test values
plt.plot(test,color= "blue" , label= "test")
plt.plot(predictions,color= "red", label= "predictions")
plt.xlabel("time")
plt.ylabel("Test and predicted data")
plt.show()

# Compute RMSE (%) and MAPE between actual and predicted test data.
mse = sklearn.metrics.mean_squared_error(test, predictions)
rmse = (math.sqrt(mse)/np.mean(test))*100
mape=sklearn.metrics.mean_absolute_percentage_error(test, predictions)
print(rmse)
print(mape*100)


