import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
#print(iris.data)
X = iris.data[:,:2]
#print(X)

wine =datasets.load_wine()
print(wine.data)
X = wine.data[:,:2]
print(X)
