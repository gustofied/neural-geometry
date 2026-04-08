import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class ReLU():
    def forward(self, x):
        self.x_in = np.copy(x)
        return np.clip(x,0,None)

    def backward(self, grad):
        return np.where(self.x_in>0,grad,0) 


class Sigmoid():
    def forward(self, x):
        self.y_out = np.exp(x) / (1. + np.exp(x)) # writing this term the more common way with diviining on 1 + np.exp(-x) will mitigaate the issue of overlfow when im doing np.exp(x) as it can become very largey
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

class Linear():
    def __init__(self, n_in, n_out):
        self.weights = np.random.randn(n_in,n_out) * np.sqrt(2/n_in)
        self.biases = np.zeros(n_out)

    def forward(self, x):
        self.x_in = x
        return x @ self.weights + self.biases

    def backward(self, grad):
        self.grad_b = grad.mean(axis=0)
        self.grad_w = (self.x_in[:,:,None] @ grad[:,None,:]).mean(axis=0)
        return grad @ self.weights.T


class Model():
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def loss(self,x,y):
        return self.cost.forward(self.forward(x),y)

    def backward(self):
        grad = self.cost.backward()
        for i in range(len(self.layers)-1,-1,-1):
            grad = self.layers[i].backward(grad)


net = Model([Linear(2, 20), ReLU(), Linear(20, 2), Softmax()], CrossEntropy())

def train(model,lr,nb_epoch,data):
    for epoch in range(nb_epoch):
        running_loss = 0.
        num_inputs = 0
        for mini_batch in data:
            inputs,targets = mini_batch
            num_inputs += inputs.shape[0]
            #Forward pass + compute loss
            running_loss += model.loss(inputs,targets).sum()
            #Back propagation
            model.backward()
            #Update of the parameters
            for layer in model.layers:
                if type(layer) == Linear:
                    layer.weights -= lr * layer.grad_w
                    layer.biases -= lr * layer.grad_b
        print(f'Epoch {epoch+1}/{nb_epoch}: loss = {running_loss/num_inputs}')

# data

X, y = make_moons(n_samples=1000, noise=0.1)

Y = np.zeros((len(y), 2))
Y[np.arange(len(y)), y] = 1

def make_batches(X, Y, batch_size=64):
    data = []
    for i in range(0, len(X), batch_size):
        data.append((X[i:i+batch_size], Y[i:i+batch_size]))
    return data

data = make_batches(X, Y)

train(net, lr=0.1, nb_epoch=50, data=data)