import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


# plot decision region
def plot_decision_regions(X, y, classifier, X_test, y_test, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        color = cmap(idx)
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=color,
                    marker=markers[idx], label=cl)

    plt.scatter(X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                alpha=1.0,
                linewidths=1,
                marker='o',
                s=55, label='test set')


# Step 1 load and Split data
wine = datasets.load_wine()
X = wine.data[:, :2]
y = wine.target
attributes = [wine.feature_names[0], wine.feature_names[1]]
accuracy_value = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.14)  # 0.14* 0.7 = 0.2

c_list = [0.01, 1, 100]
gamma_list = [0.1, 1, 10]

# Create a SVC classifier using an RBF kernel
svm = svm.SVC(kernel='rbf', random_state=0, gamma=10000, C=100)
# Train the classifier
svm.fit(X_train, y_train)

# Visualize the decision boundaries
plot_decision_regions(X_train, y_train, svm, X_test, y_test)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
