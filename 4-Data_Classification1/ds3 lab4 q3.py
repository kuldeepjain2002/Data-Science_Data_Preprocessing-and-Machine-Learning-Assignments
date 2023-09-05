import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data= pd.read_csv("SteelPlateFaults-2class2_edited_final.csv")
attributes = list(data.columns)
def Cal_likelihood(data, att_name, att_value,Y, label):
    attributes = list(data.columns)
    data = data[data[Y]==label]
    mean, std = data[att_name].mean(),data[att_name].std()
    # print(att_name, mean, std)
    px_y = (1/(np.sqrt(2*np.pi)*std)) *np.exp(-((att_value-mean)**2 / (2 * std**2 )))
    return px_y

def cal_Priority(data, Y):
    classes = sorted(list(data[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(data[data[Y] == i]) / len(data))
    return prior

def naive_bayes_gaussian(df,X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = cal_Priority(df, Y)


    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= Cal_likelihood(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]


        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)

train, test = train_test_split(data, test_size=.2, random_state=41)
X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values
Y_pred = naive_bayes_gaussian(train, X=X_test, Y="Class")

from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))

data2 = pd.read_csv("SteelPlateFaults-2class.csv")
dk = data2.groupby(["Class"])
# mean od attributes after breaking them by class
df0 = data2[data2["Class"]==0]
df1 = data2[data2["Class"]==1]

for i in attributes[:-1]:
    print(i, round(df0[i].mean(),3), round(df1[i].mean(), 3))


# covariance matrix
cov0 = df0.iloc[:, :-1].cov()
print(cov0)
cov1 = df1.iloc[:, :-1].cov()


cov0.to_excel("cov0.xlsx")
cov1.to_excel("cov1.xlsx")
