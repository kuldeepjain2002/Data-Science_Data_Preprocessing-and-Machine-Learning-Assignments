import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data= pd.read_csv("abalone.csv")

# # splitting the data to train and test data
train, test = train_test_split(data, test_size=.3, random_state=42)
# print(train)
train.to_csv("abalone-train.csv")
test.to_csv("abalone-test.csv")

xtrain= train.iloc[:, :-1]
xtest = test.iloc[:,:-1]

## separating target attribute

# print(xtrain)
target = train["Rings"]
# print(target)
test_target= test["Rings"]
attributes= list(data.columns)

# implement linear regression

reg = LinearRegression().fit(xtrain, target)

# predicting training data

train_y_pred = reg.predict(xtrain)
print(np.sqrt(metrics.mean_squared_error(target, train_y_pred)))
# predicting test data

test_y_pred = reg.predict(xtest)
print(np.sqrt(metrics.mean_squared_error(test_target, test_y_pred)))

# Plotting the required graph
plt.scatter(test_target, test_y_pred)
plt.title("Multivariate Linear regression ")
plt.xlabel("Actual Rings")
plt.ylabel("Predicted rings")
plt.show()






