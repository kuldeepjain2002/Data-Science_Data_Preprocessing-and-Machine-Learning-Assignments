import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data= pd.read_csv("abalone.csv")
train, test = train_test_split(data, test_size=.3, random_state=42)
# print(train)
train.to_csv("abalone-train.csv")
test.to_csv("abalone-test.csv")

xtrain= train.iloc[:, :-1]
xtest = test.iloc[:,:-1]

# print(xtrain)
target = train["Rings"]
# print(target)
test_target= test["Rings"]
attributes= list(data.columns)

x, y = np.array(xtrain), np.array(target)
xt , yt = np.array(xtest), np.array(test_target)

# collecting the degrees
xd = [2,3,4,5]
yd=[]
y2d=[]
for d in [2,3,4,5]:
    #  Transform input data
    x_ = PolynomialFeatures(degree=d, include_bias=False).fit_transform(x)
    # Create a model and fit it
    model = LinearRegression().fit(x_, y)

    #  Predict train data
    y_pred = model.predict(x_)
    print(f"for degree {d} , rmse error train data")
    rmse= np.sqrt(metrics.mean_squared_error(y, y_pred))
    yd.append(rmse)
    print(rmse)

    #  Predict test data

    x_test = PolynomialFeatures(degree=d, include_bias=False).fit_transform(xt)
    y_pred_test = model.predict(x_test)
    print(f"for degree {d}, accuracy for test data")
    rmse2 =np.sqrt(metrics.mean_squared_error(yt, y_pred_test))
    y2d.append(rmse2)
    print(rmse2)


# # Plot the bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis).for train data
plt.bar(xd,yd)
plt.title("Multivariate Polynomial Regression")
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE value")
plt.show()

# Plot the bar graph of RMSE (y-axis) vs different values of degree of the polynomial (x-axis).for test data

plt.bar(xd,y2d)
plt.title("Multivariate Polynomial Regression")
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE value")
plt.show()

# best degree 2
# 4
# plot the required scatter plot
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
    # Step 3: Create a model and fit it
model = LinearRegression().fit(x_, y)
x_test = PolynomialFeatures(degree=2, include_bias=False).fit_transform(xt)
y_pred_test = model.predict(x_test)

plt.scatter(test_target, y_pred_test)
plt.title("Univariate Polynomial regression ")
plt.xlabel("Actual Rings")
plt.ylabel("Predicted rings")
plt.show()








