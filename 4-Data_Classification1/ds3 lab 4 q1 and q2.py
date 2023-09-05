import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data= pd.read_csv("SteelPlateFaults-2class.csv")
X = data.iloc[:, :-1]
X_label = data.iloc[:, -1]

# forming test and train datas
[X_train, X_test, X_label_train, X_label_test] = train_test_split(X, X_label, test_size=0.3, random_state=42, shuffle=True)
X_train.to_csv("SteelPlateFaults-train.csv")
X_test.to_csv("SteelPlateFaults-test.csv")

print(data)


# import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
arr= [1,3,5]

# applying KNN foe different values of k using loop
for i in arr:
    classifier = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
    classifier.fit(X_train, X_label_train)
    # from sklearn.neighbors import KNeighborsClassifier
    # classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix,accuracy_score
    # confusion matrix
    cm = confusion_matrix(X_label_test, y_pred)
    print(f"confusion matrix for i = {i}")
    print(cm)
    # accuracy
    ac = accuracy_score(X_label_test, y_pred)
    print(f"Accuracy matrix for i = {i}")

    print(ac)

print("_______________q2________________")

# forming normalized data of train and test data separately
normalized_train = X_train.copy()
normalized_test= X_test.copy()

# normalization formula
for column in normalized_train.columns:
    normalized_train[column] = (normalized_train[column] - X_train[column].min()) / (X_train[column].max() - X_train[column].min())
for column in normalized_test.columns:
    normalized_test[column] = (normalized_test[column] - X_train[column].min()) / (X_train[column].max() - X_train[column].min())

print(normalized_train)

# convert into csv file
normalized_train.to_csv("SteelPlateFaults-train-normalised.csv")
normalized_test.to_csv("SteelPlateFaults-test-normalised.csv")

# KNN for normalized data
from sklearn.neighbors import KNeighborsClassifier

arr = [1, 3, 5]
for i in arr:
    classifier = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
    classifier.fit(normalized_train, X_label_train)
    # from sklearn.neighbors import KNeighborsClassifier
    # classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

    y_pred = classifier.predict(normalized_test)
    from sklearn.metrics import confusion_matrix, accuracy_score
    # confusion matrix
    cm = confusion_matrix(X_label_test, y_pred)
    print(f"confusion matrix for i = {i}")
    print(cm)
    # accuracy
    ac = accuracy_score(X_label_test, y_pred)
    print(f"Accuracy matrix for i = {i}")

    print(ac)