import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# Step 1 load and Split data
wine = datasets.load_wine()
X = wine.data[:, :2]
y = wine.target
attributes = [wine.feature_names[0], wine.feature_names[1]]
accuracy_value = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.14)  # 0.14* 0.7 = 0.2

# Step 2 train and test model
c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for c in c_list:
    clf = svm.SVC(kernel='linear', C=c, decision_function_shape="ovr")  # Linear Kernel
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    accuracy_value.append(score)
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of linear SVC C = {}'.format(c))
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel(attributes[1])
    ax.set_xlabel(attributes[0])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()

print(accuracy_value)


def plot_evaluation_of_C(cs):
    x = c_list
    y = cs
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=12)
    plt.xticks(c_list)
    plt.xscale("log")
    plt.title("Accuracy with respect to C")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.show()


plot_evaluation_of_C(accuracy_value)
