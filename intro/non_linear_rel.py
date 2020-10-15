from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

x = np.array([2, 3, 4])
poly = PolynomialFeatures(4, include_bias=False)
arr = poly.fit_transform(x[:, None])

print(arr)
n_points = 100
x = np.random.rand(n_points)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.rand(n_points)
plt.scatter(x=x, y=y)
# plt.show()

poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
poly_model.fit(x.reshape(-1, 1), y)

xfit = np.linspace(0, 10, 1000)
yfit = poly_model.predict(xfit.reshape(-1, 1))

plt.scatter(x, y)
lim = plt.axis()

plt.plot(xfit, yfit, color='red');
plt.axis(lim)

plt.show()
