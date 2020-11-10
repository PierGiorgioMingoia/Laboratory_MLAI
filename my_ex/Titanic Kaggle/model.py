# Data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
# Machine Learning
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Acquisition of Data
train_df = pd.read_csv('./Data/train.csv')
test_df = pd.read_csv('./Data/test.csv')
combine = [train_df, test_df]

data_set = train_df[["Pclass", "Sex", ]].copy()
data_set["Sex"] = data_set["Sex"].map({'male': 0, 'female': 1})
print(data_set)
data_set_label = train_df["Survived"]
d_l = data_set_label.to_numpy()
d = data_set.to_numpy()
print(d)
X_train, X_val, y_train, y_val = train_test_split(d, d_l, test_size=.3)

X_test = test_df[["Pclass", "Sex"]].copy()
X_test["Sex"] = X_test["Sex"].map({'male': 0, 'female': 1})
X_test.to_numpy()
# y_test = test_df["Survived"]
# define

h = 0.02  # step size
cmap_light = ListedColormap(['#FFFFAA', '#AAFFAA', '#AAAAFF', '#EFEFEF'])
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF', '#000000'])
plot_symbol_size = 50


def plot_data_and_decision_boundaries(data, x_train, y_train, clf, k):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    # plot training data
    plt.scatter(x_train[:, 0], x_train[:, 1], s=plot_symbol_size, c=y_train, cmap=cmap_bold, edgecolor='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())y

    patch0 = mpatches.Patch(color='#FFFF00', label='class 0')
    patch1 = mpatches.Patch(color='#000000', label='class 1')
    patch2 = mpatches.Patch(color='#0000FF', label='class 2')
    plt.legend(handles=[patch0, patch1, patch2])
    title = "K = {}".format(k)
    plt.title(title)
    plt.show()


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
#plot_data_and_decision_boundaries(d, X_train, y_train, knn, 5)
score = knn.score(X_val, y_val)
test_predicted = knn.predict(X_test)

print(score)