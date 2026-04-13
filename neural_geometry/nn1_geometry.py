"""
nn1_geometry.py — ReLU geometry

Radial-band dataset, 2-class softmax classifier built from scratch in NumPy.
"""
import numpy as np
from nn1data import make_radial_bands


class ReLU:
    def forward(self, x):
        self.x_in = np.copy(x)
        return np.clip(x, 0, None)

    def backward(self, grad):
        return np.where(self.x_in > 0, grad, 0)


class Softmax:
    def forward(self, x):
        exp = np.exp(x - x.max(axis=1, keepdims=True))
        self.y_out = exp / exp.sum(axis=1, keepdims=True)
        return self.y_out

    def backward(self, grad):
        return self.y_out * (grad - (grad * self.y_out).sum(axis=1)[:, None])


class CrossEntropy:
    def forward(self, x, y):
        self.x_in = x.clip(min=1e-8, max=None)
        self.y_in = y
        return (np.where(y == 1, -np.log(self.x_in), 0)).sum(axis=1)

    def backward(self):
        return np.where(self.y_in == 1, -1 / self.x_in, 0)


class Linear:
    def __init__(self, n_in, n_out):
        self.weights = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        self.biases  = np.zeros(n_out)

    def forward(self, x):
        self.x_in = x
        return x @ self.weights + self.biases

    def backward(self, grad):
        self.grad_b = grad.mean(axis=0)
        self.grad_w = (self.x_in[:, :, None] @ grad[:, None, :]).mean(axis=0)
        return grad @ self.weights.T


class Model:
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost   = cost

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        return self.cost.forward(self.forward(x), y)

    def backward(self):
        grad = self.cost.backward()
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)


def train(model, X, y, lr, nb_epoch, batch_size=64):
    Y   = np.eye(int(y.max()) + 1)[y]
    rng = np.random.default_rng(0)

    for epoch in range(nb_epoch):
        perm     = rng.permutation(len(X))
        X_s, Y_s = X[perm], Y[perm]

        running_loss = 0.
        for i in range(0, len(X_s), batch_size):
            xb, yb = X_s[i:i+batch_size], Y_s[i:i+batch_size]
            running_loss += model.loss(xb, yb).sum()
            model.backward()
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.weights -= lr * layer.grad_w
                    layer.biases  -= lr * layer.grad_b

        if (epoch + 1) % 200 == 0:
            loss = running_loss / len(X_s)
            acc  = (np.argmax(model.forward(X), axis=1) == y).mean()
            print(f"Epoch {epoch+1}/{nb_epoch}  loss={loss:.4f}  acc={acc:.4f}")


def build():
    np.random.seed(42)
    X, y, _ = make_radial_bands(
        n_samples=1600, band_radii=(0.55, 1.05, 1.55, 2.05),
        band_width=0.12, xy_noise=0.02, seed=42,
    )
    net = Model([Linear(2, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 2), Softmax()],
                CrossEntropy())
    train(net, X, y, lr=0.05, nb_epoch=2000)
    print(f"Final accuracy: {(np.argmax(net.forward(X), axis=1) == y).mean():.4f}")
    return net, X, y
