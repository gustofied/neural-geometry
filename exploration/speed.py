import timeit
import numpy as np
from numba import njit


# forward pass through a 2-layer ReLU net, three ways
# pure python to show what numpy is actually saving you from

N       = 200
D_IN    = 2
D_H     = 32

W1 = np.random.randn(D_H, D_IN).astype(np.float64)
b1 = np.random.randn(D_H).astype(np.float64)
W2 = np.random.randn(D_H, D_H).astype(np.float64)
b2 = np.random.randn(D_H).astype(np.float64)
X  = np.random.randn(N, D_IN).astype(np.float64)


def forward_python(W1, b1, W2, b2, x):
    N, D = x.shape
    H    = W1.shape[0]
    h    = [[max(0.0, sum(x[i, k] * W1[j, k] for k in range(D)) + b1[j])
             for j in range(H)] for i in range(N)]
    out  = [[sum(h[i][k] * W2[j, k] for k in range(H)) + b2[j]
             for j in range(H)] for i in range(N)]
    return out


def forward_numpy(W1, b1, W2, b2, x):
    h = np.maximum(0, x @ W1.T + b1)
    return h @ W2.T + b2


@njit
def forward_numba(W1, b1, W2, b2, x):
    N, D = x.shape
    H    = W1.shape[0]
    h    = np.empty((N, H))
    for i in range(N):
        for j in range(H):
            s = b1[j]
            for k in range(D):
                s += x[i, k] * W1[j, k]
            h[i, j] = s if s > 0 else 0.0
    out = np.empty((N, H))
    for i in range(N):
        for j in range(H):
            s = b2[j]
            for k in range(H):
                s += h[i, k] * W2[j, k]
            out[i, j] = s
    return out


if __name__ == "__main__":
    print(f"forward pass  —  {N} samples, {D_IN}→{D_H}→{D_H}\n")

    # python
    t_py = timeit.timeit(lambda: forward_python(W1, b1, W2, b2, X), number=5) / 5
    print(f"  python   {t_py * 1000:8.2f} ms")

    # numpy
    t_np = timeit.timeit(lambda: forward_numpy(W1, b1, W2, b2, X), number=500) / 500
    print(f"  numpy    {t_np * 1000:8.4f} ms   {t_py / t_np:6.0f}x faster than python")

    # numba — compile first
    forward_numba(W1, b1, W2, b2, X)
    t_nb = timeit.timeit(lambda: forward_numba(W1, b1, W2, b2, X), number=500) / 500
    print(f"  numba    {t_nb * 1000:8.4f} ms   {t_py / t_nb:6.0f}x faster than python  "
          f"  {t_np / t_nb:.1f}x vs numpy")
