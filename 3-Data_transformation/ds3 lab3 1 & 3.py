import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
# obtaining the data
data1 = pd.read_csv("pima-indians-diabetes.csv", sep=",")
data = pd.DataFrame(data1)
attributes = list(data.columns)
counts =data.isna().sum()

data2 = data.copy(deep = True)

# Reolacement of outliers by median
for name in attributes[0:8]:
    q1 = data[name].quantile(0.25)
    q3 = data[name].quantile((0.75))
    med = np.median(data[name])
    IQR = q3-q1
    # lower and upper limit
    ll = q1 - 1.5*IQR
    ul = q3 + 1.5*IQR
    # checking outlier condition and replacing it with the median
    for i in range(len(data[name])):
        if (ll>data[name][i]) or (ul<data[name][i]):
            data.loc[i, name] = med
# Min max normalization
for name in attributes[0:8]:
    min1 = min(data2[name])
    max1 = max(data2[name])
    print(name,max1, min1)
    for i in range(len(data2[name])):
        data2.loc[i, name] = ((data2.loc[i, name] - min1) / (max1 - min1)) * (7) + 5
    min2 = min(data2[name])
    max2 = max(data2[name])
    print(name, max2, min2)

# Z score Normalization
dataZ= data.copy(deep = True)
del dataZ["class"]
for name in attributes[0:8]:
    mean1 = data[name].mean()
    std1= statistics.stdev(data[name])
    print(name,mean1, std1)
    for i in range(len(dataZ[name])):
        dataZ.loc[i, name] =( data.loc[i, name] - mean1)/std1
    mean2 = dataZ[name].mean()
    std2 = statistics.stdev(dataZ[name])
    print(mean2, std2)

# ques 3
print("Question3")
# Performing PCA on Standardized Dataframe"""

# using in =built function for PCA analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(dataZ)
x_pca = pca.transform(dataZ)
prin_DF = pd.DataFrame(data=x_pca, columns=['principal component 1', 'principal component 2'])
print(dataZ.shape)

print(prin_DF)


prin_DF.plot.scatter('principal component 1', 'principal component 2', s=20)
plt.tight_layout()
plt.show()
print(f'Explained variation per principal component: {pca.explained_variance_}')

# find eigen values n eigen vectors
from numpy.linalg import eig
cov = np.cov(x_pca.T)
v, w = np.linalg.eig(cov)
print("eigenvalues are:", v )
print("eigen vectors", w)

cov = np.cov(dataZ.T)
evalue, evect= np.linalg.eig(cov)
print(evalue)
evalue = sorted(evalue, reverse= True)
plt.bar(np.arange(1, 9), evalue)
plt.xlabel("Eigenvalues")
plt.ylabel("Magnitude of eigenvalues")
plt.title("Eigenvalues")
plt.show()

# c
# Reconstruction error
error = {}
for i in range(1, 9):
    pca = PCA(n_components=i)
    x = pca.fit_transform(dataZ)
    x_new = pca.inverse_transform(x)
    d = pd.DataFrame(data=x)
    d_rec = pd.DataFrame(data=x_new, columns=attributes[:8])
    error[i] = np.linalg.norm(d_rec - dataZ, None)
    print(round(d.cov(), 3))
print(round(dataZ.cov(), 3))
plt.plot(error.keys(), error.values())
plt.xlabel("l")
plt.ylabel("Reconstruction error")
plt.show()

# d Give the covariance matrix for the original data (8-dimensional). Compare the
# covariance matrix for the original data (8-dimensional) with that of the covariance matrix for 8-dimensional representation
# obtained using PCA with l = 8.
pca = PCA(n_components=8)
pca.fit(dataZ)
x_pca_8 = pca.transform(dataZ)
prin_DF = pd.DataFrame(data=x_pca_8, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
cov1 = np.cov(dataZ.T)
cov2 = np.cov(x_pca_8.T)
print("First covariance matrix", cov1)
print("second covariance matrix", cov2)



