import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# splitting the data to train and test data

data= pd.read_csv("abalone.csv")
xtrain, xtest = train_test_split(data, test_size=.3, random_state=42)
xtrain.to_csv("abalone-train.csv")
xtest.to_csv("abalone-test.csv")

# separating target attribute

target = xtrain["Rings"]
test_target = xtest["Rings"]
attributes= list(data.columns)

# collecting pearson coefficient for each attribute with the target attribute

PC= {}
from scipy.stats import pearsonr
for i in attributes[:-1]:
    PC[i]= pearsonr(data["Rings"], data[i])[0]
# print(PC)
PC = pd.Series(PC)
print(PC)
att=PC.idxmax()

# reshape the train data and the test data of the best suited attribute

x= np.array(xtrain[att]).reshape(-1,1)
xt= np.array(xtest[att]).reshape(-1,1)

# implementing polynomial regression for different degrees
# 1 and 2
xd = [2,3,4,5]
yd=[]
y2d=[]
for d in [2,3,4,5]:
    x2 = PolynomialFeatures(degree=d, include_bias=False).fit_transform(x)

    model = LinearRegression().fit(x2,target)

    # predicting training data collect rmse error value

    train_y_pred = model.predict(x2)

    print(f"for degree {d} , RMSE error for train data")
    rmse=np.sqrt(metrics.mean_squared_error(target, train_y_pred))
    yd.append(rmse)
    print(rmse)

    # predicting test data and collect rmse error value

    xx= np.array(xtest[att]).reshape(-1,1)
    xx= PolynomialFeatures(degree=d, include_bias=False).fit_transform(xx)

    test_y_pred = model.predict(xx)
    print(f"for degree {d} , RMSE error for test data")
    rmse2=np.sqrt(metrics.mean_squared_error(test_target, test_y_pred))
    y2d.append(rmse2)
    print(rmse2)

# Plot the bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis).for train data
plt.bar(xd,yd)
plt.title("Multivariate Polynomial Regression")
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE value")
plt.show()

# Plot the bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis) for test data.
plt.bar(xd,y2d)
plt.title("Multivariate Polynomial Regression")
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE value")
plt.show()
# test rmse lowest for degree 4

# 3
k = np.linspace(0,1,1000)


# implementing linear regrssion after transformation
x2 = PolynomialFeatures(degree=4, include_bias=False).fit_transform(x)
model = LinearRegression().fit(x2,target)
kk= np.array(k).reshape(-1,1)
kk= PolynomialFeatures(degree=4, include_bias=False).fit_transform(kk)
y = model.predict(kk)

# plot the graph
plt.scatter(xtrain[att], target)
plt.scatter(k,y, color = "red")
plt.title("scatter plot along with best fit line")
plt.xlabel("Shell Weight")
plt.ylabel("Rings")
plt.show()


# 4

x_test = PolynomialFeatures(degree=4, include_bias=False).fit_transform(xt)
y_pred_test = model.predict(x_test)

# Plot the graph
plt.scatter(test_target, y_pred_test)
plt.title("Univariate Polynomial regression ")
plt.xlabel("Actual Rings")
plt.ylabel("Predicted rings")
plt.show()

