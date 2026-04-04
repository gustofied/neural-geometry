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

PREDICTORS = ["tmax", "tmin", "rain"]
TARGET = "tmax_tomorrow"

np.random.seed(0)
train, valid, test = data.iloc[:int(.7*len(data))], data.iloc[int(.7*len(data)):int(.85*len(data))], data.iloc[int(.85*len(data)):]
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in [train, valid, test]]

import math

def init_params(predictors):
    np.random.seed(0)
    weights = np.random.rand(predictors, 1)
    biases = np.ones((1, 1))
    return [weights, biases]

init_params(3)

def forward(params, x):
    weights, biases = params
    prediction = x @ weights + biases
    return prediction

def mse(actual, predicted):
    return np.mean((actual-predicted) ** 2)

def mse_grad(actual, predicted):
    return predicted - actual

def backward(params, x, lr, grad):
    # x1 * g, x2 * 2, x3 * g
    w_grad = (x.T / x.shape[0]) @ grad
    b_grad = np.mean(grad, axis=0)

    params[0] -= w_grad * lr
    params[1] -= b_grad * lr

    return params

lr = 1e-4
epochs = 10000

params = init_params(train_x.shape[1])

for i in range(epochs):
    predictions = forward(params, train_x)
    grad = mse_grad(train_y, predictions)

    params = backward(params, train_x, lr, grad)

    if i % 1000 == 0:
        predictions = forward(params, valid_x)
        valid_loss = mse(valid_y, predictions)

        print(f"Epoch {i} loss: {valid_loss}")

