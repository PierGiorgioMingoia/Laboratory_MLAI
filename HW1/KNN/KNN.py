import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, metrics
from sklearn.model_selection import train_test_split

# define

h = 0.02  # step size
cmap_light = ListedColormap(['#FFFFAA', '#AAFFAA', '#AAAAFF', '#EFEFEF'])
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF', '#000000'])
plot_symbol_size = 50


def plot_data_and_decision_boundaries(data,x_train,y_train, clf, k):
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
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FFFF00', label='class 0')
    patch1 = mpatches.Patch(color='#000000', label='class 1')
    patch2 = mpatches.Patch(color='#0000FF', label='class 2')
    plt.legend(handles=[patch0, patch1, patch2])
    plt.xlabel(attributes[0])
    plt.ylabel(attributes[1])
    title = "K = {}".format(k)
    plt.title(title)
    plt.show()


def plot_evaluation_of_k(ks):
    x = list
    y = ks
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=12)
    plt.title("Accuracy with respect to K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()


# Step 1 load Wine dataset

wine = datasets.load_wine()
# Step 2 select only two attributes
X = wine.data[:, :2]
y = wine.target
attributes = [wine.feature_names[0], wine.feature_names[1]]
accuracy_value = []
# Step 3 split into train, validation, test set 5:2:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.14)  # 0.14* 0.7 = 0.2
# Step 4 for k = [1,3,5,7]
list = [1, 3, 5, 7]

for k in list:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    # train knn
    knn.fit(X_train, y_train)
    # apply
    y_test_predicted = knn.predict(X_test)
    # plot data and decision bound
    plot_data_and_decision_boundaries(X, X_train,y_train, knn, k)

    # evaluate
    a = knn.score(X_val, y_val)
    accuracy_value.append(a)
    print("Accuracy:", a)

plot_evaluation_of_k(accuracy_value)
