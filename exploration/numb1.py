import timeit
import numpy as np
from numba import njit


@njit
def forward_relu(W1, b1, W2, b2, x):
    h = np.maximum(0, x @ W1.T + b1)
    return h @ W2.T + b2


def forward_relu_numpy(W1, b1, W2, b2, x):
    h = np.maximum(0, x @ W1.T + b1)
    return h @ W2.T + b2


def bench(label, fn, args, n_runs=1000):
    fn(*args)  # warmup / compile
    t = timeit.timeit(lambda: fn(*args), number=n_runs)
    print(f"  {label:<12} {t / n_runs * 1000:.3f} ms/call")


if __name__ == "__main__":
    configs = [
        (1000, 2, 64),
        (10000, 2, 256),
    ]

    for n_points, d_in, d_hidden in configs:
        W1 = np.random.randn(d_hidden, d_in)
        b1 = np.random.randn(d_hidden)
        W2 = np.random.randn(1, d_hidden)
        b2 = np.random.randn(1)
        x = np.random.randn(n_points, d_in)
        args = (W1, b1, W2, b2, x)

        print(f"points={n_points} hidden={d_hidden}")
        bench("numpy", forward_relu_numpy, args)
        bench("numba", forward_relu, args)
        print()
