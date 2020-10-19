import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

Cs = [0.01, 1, 10, 100, 1000]
gammas = [0.0001, 0.001, 0.1, 1, 10, 100, 10000]


def svc_param_selection(X, y):
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.cv_results_['mean_test_score']


wine = datasets.load_wine()
X = wine.data[:, :2]
y = wine.target
attributes = [wine.feature_names[0], wine.feature_names[1]]
accuracy_value = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.14)  # 0.14* 0.7 = 0.2

matrix = []
for c in Cs:
    for gamma in gammas:
        # Create a SVC classifier using an RBF kernel
        clf = svm.SVC(kernel='rbf', random_state=0, C=c, gamma=gamma)
        # Train the classifier
        clf.fit(X_train, y_train)
        score = clf.score(X_val, y_val)
        matrix.append(score)

matrix = np.array(matrix)
print(matrix)

scores = matrix.reshape(5,7)

"""
# Create a SVC classifier using an RBF kernel
svm = svm.SVC(kernel='rbf', random_state=0, C=100, gamma=10000)
# Train the classifier
svm.fit(X_train, y_train)
best, matrix = svc_param_selection(X_train, y_train)
scores = matrix.reshape(5, 7)
"""
print(len(matrix))

fig, ax = plt.subplots(figsize=(8, 6))
plt.xticks(np.arange(len(gammas)), gammas, rotation=45)
plt.yticks(np.arange(len(Cs)), Cs)
ax.matshow(scores, cmap=plt.cm.Blues)
# Set the ticks and ticklabels for all axes

for i in range(len(gammas)):
    for j in range(len(Cs)):
        c = "%.2f" % scores[j, i]
        ax.text(i, j, c, va='center', ha='center')

plt.show()

