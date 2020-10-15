import numpy as np
import matplotlib.pyplot as plt

# Ex 1

n_points = 100
x = 10 * np.random.rand(n_points)
y = 2 * x - 5 + np.random.rand(100)

# plt.scatter(x, y)
# plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(x.reshape(-1, 1), y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit.reshape(-1, 1))

plt.scatter(x, y)
plt.plot(xfit, yfit, color='red')
plt.show()

print("Model slope: {:.4f}".format(model.coef_[0]))
print("Model intercept: {:.4f}".format(model.intercept_))

