from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

x, y = make_blobs(100, 2, random_state=2, centers=2, cluster_std=1.5)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='RdBu')
# plt.show()

mask_l0 = y == 0
mask_l1 = y == 1

# print(mask_l1, mask_l0)

mean_l0 = x[mask_l0].mean(0)
mean_l1 = x[mask_l1].mean(0)
var_l0 = x[mask_l0].var(0)
var_l1 = x[mask_l1].var(0)
print("Mean l0: {}, l1: {}".format(mean_l0, mean_l1))
print("Var l0: {}, l1: {}".format(var_l0, var_l1))

from scipy.stats import norm
import math
import numpy as np

dist_l0_f0, dist_l0_f1 = norm(mean_l0[0], math.sqrt(var_l0[0])), norm(mean_l0[1], math.sqrt(var_l0[1]))
dist_l1_f0, dist_l1_f1 = norm(mean_l1[0], math.sqrt(var_l1[0])), norm(mean_l1[1], math.sqrt(var_l1[1]))


def prob_l0(point):
    return dist_l0_f0.pdf(point[0]) * dist_l0_f1.pdf(point[1])


def prob_l1(point):
    return dist_l1_f0.pdf(point[0]) * dist_l1_f1.pdf(point[1])


def predict_p(point):
    result = prob_l0(point) / prob_l1(point)
    if result > 1:
        return 0
    return 1


def predict_set(points):
    points = [predict_p(point) for point in points]
    return np.array(points)


Xnew = [-6, -14] + [14, 18] * np.random.rand(2000, 2)
ynew = predict_set(Xnew)

# print(ynew)

plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()

plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
# plt.show()

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x, y)

print("GNB mean:", model.theta_)
print("Mean l0: {}. li{}".format(mean_l0, mean_l1))

ynew = model.predict(Xnew)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='RdBu')
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)

plt.show()