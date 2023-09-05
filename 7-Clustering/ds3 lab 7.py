import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
# import the data
data = pd.read_csv("Iris.csv")
# drop the last column
df = data.iloc[:,:4]


# 1
# Performing PCA - buliding a model
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)
x_pca = pca.transform(df)
DF = pd.DataFrame(data=x_pca, columns=['principal component 1', 'principal component 2'])

# the result dataframe is called DF here

# 1b
# Plot Eigen value- import required module
from numpy.linalg import eig
v, w = np.linalg.eig(df.corr().to_numpy())
labels = [1, 2, 3, 4]
plt.bar(labels, [round(i, 3) for i in v])
plt.xlabel('Components')
plt.ylabel('Eigen Values')
plt.title('Eigen Values vs Components')
plt.tight_layout()
plt.show()
print(v)

# 1b plot the data
plt.scatter(DF['principal component 1'] , DF['principal component 2'])
plt.xlabel("principal component 1")
plt.ylabel('principal component 1')
plt.show()

# 2 a , b ,c
# Apply K-means (K=3) clustering on the reduced data, Here reduced data is DF

from sklearn.cluster import KMeans
K = 3
# applying the model , fitting and predicting .
kmeans = KMeans(n_clusters=K)
kmeans.fit(DF)
kmeans_prediction = kmeans.predict(DF)
data_with_clusters = DF.copy()
data_with_clusters['Clusters'] = kmeans_prediction


a= kmeans.cluster_centers_

# 2 a
plt.scatter(data_with_clusters['principal component 1'],data_with_clusters['principal component 2'],c=data_with_clusters['Clusters'],cmap='rainbow')
# Plot the centers
plt.scatter([a[i][0] for i in range(3)], [a[i][1] for i in range(3)], label='cluster centres')
plt.xlabel("principal component 1")
plt.ylabel('principal component 1')
plt.show()

# 2 b  Distortion Measure
b = kmeans.inertia_
print("Distortion measure for K=3 ",b)

from sklearn import metrics
from scipy.optimize import linear_sum_assignment

# 2 c
# Compute Purity Score Function
def purity_score(y_true, y_pred):
 # compute contingency matrix (also called confusion matrix)
 contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
 #print(contingency_matrix)
 # Find optimal one-to-one mapping between cluster labels and true labels
 row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
 # Return cluster accuracy
 return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)


h = purity_score(data["Species"], kmeans_prediction )
print("purity score - " ,h)


# 3
# forming the list of values k must take
Klist = [2, 3, 4, 5, 6 , 7]

# Initialing the array of Distortion mes=asure and Purity Scores
Dmeasure=[]
Pscores=[]

# Applying K mean Clustering for different values of K
for i in Klist:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(DF)
    kmeans_prediction = kmeans.predict(DF)
    data_with_clusters = DF.copy()
    data_with_clusters['Clusters'] = kmeans_prediction

    # Collecting  distortion measure and purity score
    b = kmeans.inertia_
    Dmeasure.append(b)

    h = purity_score(data["Species"], kmeans_prediction)
    Pscores.append(h)

plt.plot(Klist,Dmeasure)
plt.xlabel("Values of K")
plt.ylabel("Distotion Measure")
plt.show()


print("Purity Score", Pscores)


# 4 a form GMM for k =3
from sklearn.mixture import GaussianMixture
K = 3

# forming the model
gmm = GaussianMixture(n_components = 3)
# print(DF)
gmm.fit(DF)
GMM_prediction = gmm.predict(DF)
# print(GMM_prediction)
centre_gmm = gmm.means_
# adding the prediccted avlues and plot the graph
data_with_GMM = DF.copy()
data_with_GMM['Clusters'] = GMM_prediction
plt.scatter(data_with_GMM['principal component 1'],data_with_GMM['principal component 2'],c=data_with_GMM['Clusters'],cmap='rainbow')
plt.scatter([centre_gmm[i][0] for i in range(3)], [centre_gmm[i][1] for i in range(3)], label='cluster centres')
plt.xlabel('PRINCIPLE COMPONENT 1')
plt.ylabel('PRINCIPLE COMPONENT 2')
plt.show()

# the total data log likelihood at the last iteration of the GMM
data_with_GMM = data_with_GMM.drop(['Clusters'], axis=1)
print('The distortion measure for k =3 is', round(gmm.score(data_with_GMM) * len(data_with_GMM), 3))

# Purity Score
Pscoregmm = purity_score(data["Species"], GMM_prediction)
print("purity score after examples are assigned to clusters",Pscoregmm)

# 5
# Initialing the array of Distortion mes=asure and Purity Scores
Dmeasuregmm=[]
Pscoresgmm=[]
for i in Klist:

    # form model . fit model and evaluate predictions
    gmm = GaussianMixture(n_components=i)
    # print(DF)
    gmm.fit(DF)
    GMM_prediction = gmm.predict(DF)
    # print(GMM_prediction)
    data_with_GMM = DF.copy()
    data_with_GMM['Clusters'] = GMM_prediction
    loglike = gmm.lower_bound_
    Dmeasuregmm.append(loglike)

    # distortion method
    data_with_GMM = data_with_GMM.drop(['Clusters'], axis=1)
    print(f'The distortion measure for k ={i} is', round(gmm.score(data_with_GMM) * len(data_with_GMM), 3))

    # Purity score
    Pscoregmm = purity_score(data["Species"], GMM_prediction)
    print("purity score after examples are assigned to clusters", Pscoregmm)

    h = purity_score(data["Species"] , GMM_prediction)
    Pscoresgmm.append(h)

#     plotting distortion Measure
plt.plot(Klist,Dmeasuregmm)
plt.xlabel("Values of K")
plt.ylabel("Distotion Measure")
plt.show()


print("Purity Score", Pscoresgmm)


# 6 DBSCAN
from sklearn.cluster import DBSCAN

# form model and predictions

dbscan_model=DBSCAN(eps=1, min_samples=4).fit(DF)
DBSCAN_predictions = dbscan_model.labels_
data_with_clusters_DBSCAN = DF.copy()
data_with_clusters_DBSCAN['Clusters'] = DBSCAN_predictions

# plot eps=1, min_samples=4
plt.scatter(data_with_clusters['principal component 1'],data_with_clusters['principal component 2'],c=data_with_clusters_DBSCAN['Clusters'],cmap='rainbow')
plt.title(f'Data Points for Epsilon=1 and Minimum Samples=4')
plt.xlabel('PRINCIPLE COMPONENT 1')
plt.ylabel('PRINCIPLE COMPONENT 2')
plt.show()
h = purity_score(data["Species"], DBSCAN_predictions)
print("eps=1, min_samples=4",h)



dbscan_model=DBSCAN(eps=1, min_samples=10).fit(DF)
DBSCAN_predictions = dbscan_model.labels_
data_with_clusters_DBSCAN = DF.copy()
data_with_clusters_DBSCAN['Clusters'] = DBSCAN_predictions
plt.scatter(data_with_clusters['principal component 1'],data_with_clusters['principal component 2'],c=data_with_clusters_DBSCAN['Clusters'],cmap='rainbow')
plt.title(f'Data Points for Epsilon=1 and Minimum Samples=10')
plt.xlabel('PRINCIPLE COMPONENT 1')
plt.ylabel('PRINCIPLE COMPONENT 2')
plt.show()
h = purity_score(data["Species"], DBSCAN_predictions)
print("eps=1, min_samples=10",h)


dbscan_model=DBSCAN(eps=5, min_samples=4).fit(DF)
DBSCAN_predictions = dbscan_model.labels_
data_with_clusters_DBSCAN = DF.copy()
data_with_clusters_DBSCAN['Clusters'] = DBSCAN_predictions
plt.scatter(data_with_clusters['principal component 1'],data_with_clusters['principal component 2'],c=data_with_clusters_DBSCAN['Clusters'],cmap='rainbow')
plt.title(f'Data Points for Epsilon=5 and Minimum Samples=4')
plt.xlabel('PRINCIPLE COMPONENT 1')
plt.ylabel('PRINCIPLE COMPONENT 2')
plt.show()
h = purity_score(data["Species"], DBSCAN_predictions)
print("eps=5, min_samples=4-",h)


dbscan_model=DBSCAN(eps=5, min_samples=10).fit(DF)
DBSCAN_predictions = dbscan_model.labels_
data_with_clusters_DBSCAN = DF.copy()
data_with_clusters_DBSCAN['Clusters'] = DBSCAN_predictions
plt.scatter(data_with_clusters['principal component 1'],data_with_clusters['principal component 2'],c=data_with_clusters_DBSCAN['Clusters'],cmap='rainbow')
plt.title(f'Data Points for Epsilon=5 and Minimum Samples=10')
plt.xlabel('PRINCIPLE COMPONENT 1')
plt.ylabel('PRINCIPLE COMPONENT 2')
plt.show()
h = purity_score(data["Species"], DBSCAN_predictions)
print("eps=5, min_samples=10",h)






