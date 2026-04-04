import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

dir = Path(__file__).resolve().parent
data_path = dir / "data"/ "clean_weather.csv"
print(dir)

data = pd.read_csv(data_path)
data= data.fillna(data.mean(numeric_only=True))

print(data.head())
print(data.shape)

data.plot.scatter("tmax", "tmax_tomorrow")
plt.plot([30, 120], [30, 120], "green")
# plt.show()

lr = LinearRegression()
lr.fit(data[["tmax"]], data["tmax_tomorrow"])

fig, ax = plt.subplots()
data.plot.scatter("tmax", "tmax_tomorrow", ax=ax)
ax.plot(data["tmax"], lr.predict(data[["tmax"]]), "red")
print(f"Wegiht: {lr.coef_[0]:.2f}")
print(f"Bias {lr.intercept_:.2f}")
plt.show()


y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse = mean_squared_error(y_true, y_pred)

print("Mean Squared Error:", mse)

loss = lambda w, y: ((w * 80 + 11.99) - y) ** 2
gradient = lambda w, y: ((w * 80 + 11.99) - y) * 2
y = 81

ws = np.arange(-1, 3, .05)

losses = loss(ws, y)
gradients = gradient(ws, y)


# plt.scatter(ws, losses)
plt.scatter(ws, gradients)

plt.scatter
plt.plot(1.25, gradient(1.25, y), 'ro')
plt.show()


ws = np.arange(-4000, 100, 100)
losses = loss(ws , y)

plt.scatter(ws, losses)
plt.plot(1, loss(1, y), 'ro')
new_weight = 1 - gradient(1, y) * 80
plt.plot(new_weight, loss(new_weight, y), 'go')
plt.show()


ws = np.arange(-.5, 1.5, .1)
losses = loss(ws, y)

plt.scatter(ws, losses)

plt.plot(1, loss(1, y), 'ro')
lr = 2e-5
new_weight = 1 - lr * gradient(1, y) * 80
plt.plot(new_weight, loss(new_weight, y), 'go')
plt.show()