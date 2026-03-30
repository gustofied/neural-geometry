import timeit
import numpy as np


def forward_relu(W1, b1, W2, b2, x):
    h = np.maximum(0, x @ W1.T + b1)
    return h @ W2.T + b2


def setup_network(d_in=2, d_hidden=64, d_out=1):
    W1 = np.random.randn(d_hidden, d_in)
    b1 = np.random.randn(d_hidden)
    W2 = np.random.randn(d_out, d_hidden)
    b2 = np.random.randn(d_out)
    return W1, b1, W2, b2


def time_forward(n_points=1000, d_in=2, d_hidden=64, d_out=1, n_runs=1000):
    W1, b1, W2, b2 = setup_network(d_in, d_hidden, d_out)
    x = np.random.randn(n_points, d_in)
    t = timeit.timeit(lambda: forward_relu(W1, b1, W2, b2, x), number=n_runs)
    return t / n_runs


def time_activation_patterns(n_points=1000, d_in=2, d_hidden=64, n_runs=1000):
    W1 = np.random.randn(d_hidden, d_in)
    b1 = np.random.randn(d_hidden)
    x = np.random.randn(n_points, d_in)
    def get_patterns():
        return (x @ W1.T + b1) > 0
    t = timeit.timeit(get_patterns, number=n_runs)
    return t / n_runs


if __name__ == "__main__":
    configs = [
        (1000, 2, 16),
        (1000, 2, 64),
        (1000, 2, 256),
        (10000, 2, 64),
        (10000, 2, 256),
    ]

    print("Forward pass timings")
    print(f"{'points':>8} {'d_in':>5} {'hidden':>7} {'ms/call':>10}")
    print("-" * 34)
    for n_points, d_in, d_hidden in configs:
        t = time_forward(n_points, d_in, d_hidden)
        print(f"{n_points:>8} {d_in:>5} {d_hidden:>7} {t*1000:>10.3f}")

    print()
    print("Activation pattern timings")
    print(f"{'points':>8} {'d_in':>5} {'hidden':>7} {'ms/call':>10}")
    print("-" * 34)
    for n_points, d_in, d_hidden in configs:
        t = time_activation_patterns(n_points, d_in, d_hidden)
        print(f"{n_points:>8} {d_in:>5} {d_hidden:>7} {t*1000:>10.3f}")
