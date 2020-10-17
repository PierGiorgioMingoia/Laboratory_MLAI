# KNN

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adspy_shared_utilities import plot_two_class_knn
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Step 1 load Wine dataset
title = "Wine k-NN"
wine = datasets.load_wine()
X = wine.data[:, :2]
y = wine.target
accuracy_value = []
# Step 3 split into train validation and test sets 5:2:3

# using np
# train, validate, test = np.split(shuffle(X, random_state=0), [int(.6 * len(y)), int(.8 * len(y))])

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.14, random_state=1)  # 0.14* 0.7 = 0.2

# Step 4 apply k-nn

# for i in range(1, 9, 2):
# create knn classifier
# knn = neighbors.KNeighborsClassifier(n_neighbors=i)
# knn.fit(X_train, y_train)

# fit the classifier to train data

print(wine.feature_names[0])
print(wine.feature_names[1])

# plot_two_class_knn(X_train, y_train, 1, 'uniform', X_test, y_test)

# test model
# print(knn.predict(X_val)[0:5])

# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = 0.02  # step size
cmap_light = ListedColormap(['#FFFFAA', '#AAFFAA', '#AAAAFF', '#EFEFEF'])
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF', '#000000'])
plot_symbol_size = 50
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = knn.predict(X_test)
# Z = Z.reshape(xx.shape)
# plt.figure()
# map color
# cmap_light = ListedColormap(['orange', 'cyan', 'darkblue'])
# plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.show()

for i in range(1, 3, 2):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # plot training data
    plt.scatter(X_train[:, 0], X_train[:, 1], s=plot_symbol_size, c=y_train, cmap=cmap_bold, edgecolor='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FFFF00', label='class 0')
    patch1 = mpatches.Patch(color='#000000', label='class 1')
    patch2 = mpatches.Patch(color='#0000FF', label='class 2')
    plt.legend(handles=[patch0, patch1, patch2])
    plt.xlabel(wine.feature_names[0])
    plt.ylabel(wine.feature_names[1])
    plt.title(title)
    plt.show()
    test_score = knn.score(X_test, y_test)
    accuracy_value.append(test_score)

clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))


#print(accuracy_value)

#plt.plot([1, 3, 5, 7], accuracy_value)
#plt.show()
