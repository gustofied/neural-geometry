import numpy as np

class ReLU():
    def forward(self, x):
        self.x_in = np.copy(x)
        return np.clip(x,0,None)

    def backward(self, grad):
        return np.where(self.x_in>0,grad,0) 


class Sigmoid():
    def forward(self, x):
        self.y_out = np.exp(x) / (1. + np.exp(x))
        return self.y_out

    def backward(self, grad):
        return self.y_out * (1. - self.y_out) * grad

class Softmax():
    def forward(self, x):
        exp = np.exp(x)
        self.y_out = exp / exp.sum(axis=1)[:, None]
        return self.y_out

    def backward(self, grad):
        return self.y_out * (grad - (grad * self.y_out).sum(axis=1)[:, None])

class CrossEntropy():
    def forward(self, x, y):
        self.x_in = x.clip(min=1e-8, max=None)
        self.y_in = y
        return (np.where(y == 1, -np.log(self.x_in), 0)).sum(axis=1)

    def backward(self):
        return np.where(self.y_in == 1, -1 / self.x_in, 0)