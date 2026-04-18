import timeit
import numpy as np
from numba import njit


# --- benchmark 1: forward pass, three ways ---
# dense matmul favors numpy because it calls optimized BLAS under the hood

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


# --- benchmark 2: ReLU region labeling on a grid ---
# per-pixel gate pattern computation, loop-heavy, numba territory

GRID = 600
H    = 32

W_r = np.random.randn(H, 2).astype(np.float64)
b_r = np.random.randn(H).astype(np.float64)

xs = np.linspace(-2.5, 3.0, GRID)
ys = np.linspace(-2.0, 2.5, GRID)


def regions_numpy(W, b, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    pre = pts @ W.T + b
    gates = (pre > 0).astype(np.uint32)
    shifts = (1 << np.arange(H, dtype=np.uint32))
    return (gates * shifts).sum(axis=1).reshape(len(ys), len(xs))


@njit
def regions_numba(W, b, xs, ys):
    out = np.empty((len(ys), len(xs)), dtype=np.uint32)
    H = W.shape[0]
    for iy in range(len(ys)):
        y = ys[iy]
        for ix in range(len(xs)):
            x = xs[ix]
            code = np.uint32(0)
            for j in range(H):
                if b[j] + W[j, 0] * x + W[j, 1] * y > 0:
                    code |= np.uint32(1 << j)
            out[iy, ix] = code
    return out


if __name__ == "__main__":

    # benchmark 1: forward pass
    print(f"forward pass, {N} samples, {D_IN} > {D_H} > {D_H}\n")

    t_py = timeit.timeit(lambda: forward_python(W1, b1, W2, b2, X), number=5) / 5
    print(f"  python   {t_py * 1000:8.2f} ms")

    t_np = timeit.timeit(lambda: forward_numpy(W1, b1, W2, b2, X), number=500) / 500
    print(f"  numpy    {t_np * 1000:8.4f} ms   {t_py / t_np:6.0f}x faster than python")

    forward_numba(W1, b1, W2, b2, X)
    t_nb = timeit.timeit(lambda: forward_numba(W1, b1, W2, b2, X), number=500) / 500
    print(f"  numba    {t_nb * 1000:8.4f} ms   {t_py / t_nb:6.0f}x faster than python  "
          f"  {t_np / t_nb:.1f}x vs numpy")

    # benchmark 2: region grid
    print(f"\nregion grid, {GRID}x{GRID}, {H} hidden units\n")

    t_np2 = timeit.timeit(lambda: regions_numpy(W_r, b_r, xs, ys), number=20) / 20
    print(f"  numpy    {t_np2 * 1000:8.3f} ms")

    regions_numba(W_r, b_r, xs, ys)
    t_nb2 = timeit.timeit(lambda: regions_numba(W_r, b_r, xs, ys), number=20) / 20
    print(f"  numba    {t_nb2 * 1000:8.3f} ms   {t_np2 / t_nb2:.1f}x vs numpy")
