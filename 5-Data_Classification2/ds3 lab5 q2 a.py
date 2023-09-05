import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# splitting the data to train and test data
data= pd.read_csv("abalone.csv")
xtrain, xtest = train_test_split(data, test_size=.3, random_state=42)
xtrain.to_csv("abalone-train.csv")
xtest.to_csv("abalone-test.csv")

# separating target attribute
target = xtrain["Rings"]
attributes= list(data.columns)

# collecting pearson coefficient for each attribute with the target attribute
PC= {}
from scipy.stats import pearsonr
for i in attributes[:-1]:
    PC[i]= pearsonr(data["Rings"], data[i])[0]

# convert to series and print the attribute with max Pearsionr
PC = pd.Series(PC)
print(PC)
att=PC.idxmax()
print(att)

# Using the formula to find slope and intercept of bets fitted model
slope = data[att].cov(target)/statistics.variance(data[att])
print(slope)
intercept = target.mean()- slope*data[att].mean()
x = np.array(xtrain[att])
y =  slope*x + intercept

# plotting the graph for the best fitted line
plt.plot(x,y ,color="red")
plt.scatter(data[att], data["Rings"])
plt.title("scatter plot along with best fit line")
plt.xlabel("Shell Weight")
plt.ylabel("Rings")
plt.show()


# implement linear regression
from sklearn.linear_model import LinearRegression
target = np.array(target)
# reshape the train data
x2=x.reshape(-1,1)
reg = LinearRegression().fit(x2, target)

x2_test= np.array(xtest[att]).reshape(-1,1)

# predicting training data
train_y_pred = reg.predict(x2)
print("train y predict",train_y_pred)

# predicting test data
test_y_pred= reg.predict(x2_test)
target_test = np.array(xtest["Rings"])
# print(test_y_pred)
print(np.sqrt(metrics.mean_squared_error(target, train_y_pred)))
print(np.sqrt(metrics.mean_squared_error(xtest["Rings"], test_y_pred)))
# # Scatter plot the predicted value
plt.scatter(target_test,test_y_pred)
plt.title("Univariate Lnear regression ")
plt.xlabel("Actual Rings")
plt.ylabel("Predicted rings")
plt.show()














