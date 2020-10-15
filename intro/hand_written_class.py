import sklearn
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

X, y = load_digits(return_X_y=True)
print(X, y)
print(X.shape)

fig = plt.figure(figsize=(10., 4.))
grid = ImageGrid(fig, 111, nrows_ncols=(2, 5), axes_pad=0.1)

for idx, ax in enumerate(grid):
    ax.imshow(X[idx].reshape(8, 8), cmap='gray')
plt.show()

from sklearn.preprocessing import StandardScaler

X_train, y_train = X[:-200], y[:-200]
X_test, y_test = X[-200:], y[-200:]

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

print(model.predict(X_test[6].reshape(1, -1)))
plt.imshow(scaler.inverse_transform(X_test[6]).reshape(8, 8), cmap='gray')

plt.show()

print("Accuracy:", model.score(X_test, y_test))
