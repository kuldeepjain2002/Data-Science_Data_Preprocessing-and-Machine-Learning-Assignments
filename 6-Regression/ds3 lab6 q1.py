import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd
# import earthpy as et
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.ar_model import AutoReg as AR
import statsmodels.api as sm
from scipy.stats import pearsonr

# read the data

data = pd.read_csv('daily_covid_cases.csv',
parse_dates=['Date'],
index_col=['Date'],
sep=',')
cases= data["new_cases"]
values = pd.DataFrame(cases.values)

# a
#  plot the whole data and also change the date format
x= list(data.index)
y= data["new_cases"]
fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(x,y)
# date format
date_form = DateFormatter("%b-%y")
ax.xaxis.set_major_formatter(date_form)
plt.xlabel(" Month - Year")
plt.ylabel("No. of cases")
plt.title("Plot of new cases")
plt.show()

# b and c

# form a dataframe that stores all the shifted datas
lags1= 6
columns = [values]
for i in range(1,(lags1 + 1)):
	columns.append(values.shift(i))
dataframe = pd.concat(columns, axis=1)
columns = ['t+1']
for i in range(1,(lags1 + 1)):
	columns.append('t-' + str(i))
dataframe.columns = columns

# list to store pearswon coefficient for all the values of lags
pcoef=[]
for i in range(1,(lags1 + 1)):

    corr, _ = pearsonr(cases[i:], dataframe.iloc[i:,i])
    pcoef.append(corr)
    if i ==1:
        # b
        plt.scatter(cases[i:], dataframe.iloc[i:,i])
        plt.title(" Scatter Plot between 1-day lagged sequence and original")
        plt.xlabel("original data")
        plt.ylabel("lag = 1")
        plt.show()

# c Graph of pearson coefficients of all the lag values
print(pcoef)
plt.plot(columns[1:],pcoef)
plt.xlabel("lags")
plt.ylabel("Correlation coefficient")
plt.title("Correlation coefficient vs. lags in given sequence")
plt.show()

# d
lagsl= [1,2,3,4,5,6]
sm.graphics.tsa.plot_acf(cases,lags = lagsl)
plt.show()