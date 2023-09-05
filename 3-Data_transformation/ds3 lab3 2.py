import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics


# define a function for projection
def proj(evec, x):
    mvec1 = evec[0] ** 2 + evec[1] ** 2
    dotp = x[0]*evec[0] + x[1]*evec[1]
    a1x= (dotp*evec[0])/mvec1
    a1y = (dotp*evec[1])/mvec1
    return a1x,a1y


# enter mean values = 0,0
a,b =map(int,input("Enter 2 space separated values:").split())
# plotting the scatter given mean and cov
mean = [0,0]
cov = [[13,-3], [-3, 5]]
x= pd.DataFrame(np.random.multivariate_normal(mean, cov, 1000))
x1 = x[0]
x2 = x[1]

# scatter Plot
plt.scatter(x1,x2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
# print(np.corr(x1,x2))

# Find eigen values and eigen vector
evalue, evect = np.linalg.eig(cov)
print(f"Eigen Value of data {evalue}")
print(f"Eigen Vector of data {evect}")

vec1 = evect[0]
vec2= evect[1]
print(vec1,vec2)

# Plotting the eigen direction in the previous scatter plot
xpos = [0,0]
ypos = [0,0]
x_direct = [0.9486, -0.3162]
y_direct = [0.3162, 0.9486]

plt.scatter(x1,x2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.quiver(xpos, ypos, x_direct, y_direct,
         scale = 5)
plt.show()

# finding x and y components of projections of the data towards eigen direction
a1x= []
a1y=[]
for i in range(len(x1)):
    ax,ay = proj(vec1, x.iloc[i,:])
    a1x.append(ax)
    a1y.append(ay)

# Plotting the projections for eigen vector 1
plt.scatter(x1,x2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.quiver(xpos, ypos, x_direct, y_direct,scale = 5)
plt.scatter(a1x,a1y, color = "red")
plt.show()

# plotting the projections for eigen vector 2
a2x= []
a2y =[]
for i in range(len(x1)):
    ax, ay = proj(vec2, x.iloc[i, :])
    a2x.append(ax)
    a2y.append(ay)

plt.scatter(x1,x2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.quiver(xpos, ypos, x_direct, y_direct,scale = 5)

plt.scatter(a2x,a2y, color = "red")
plt.show()

# Reconstruct the data
x2x=[]
x2y =[]
for i in range(len(a2x)):
    x2x.append(a1x[i]+a2x[i])
    x2y.append(a1y[i]+a2y[i])

# finding error
error=[]
for i in range(len(a2x)):
    error.append(round((((x1[i]-x2x[i])**2 + (x2[i]- x2y[i])**2)**0.5),3))
print(sum(error))



