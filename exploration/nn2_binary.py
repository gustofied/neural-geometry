"""
nn2_binary.py — toy adaptation inspired by Kristiadi et al. 2020

Binary logistic classifier (single sigmoid output) on two Gaussian blobs.
Diagonal last-layer Laplace approximation (LLLA).

  1. prior_std sweep — pick smallest std with a visible effect
  2. MAP vs LLLA confidence maps
  3. 1D x-axis confidence probe
"""
import numpy as np
import matplotlib.pyplot as plt


# ── palette (matches nn1_geometry) ───────────────────────────────────────
BG         = "#0c0c10"
FG         = "#b8b8c8"
BOUNDARY_C = "#e8004a"
C0         = "#b87c00"
C1         = "#1a7a8a"


# ── datasets ──────────────────────────────────────────────────────────────
def make_blobs(n=300, seed=42):
    rng = np.random.default_rng(seed)
    X0  = rng.normal([-3.0, 0.0], 0.45, (n // 2, 2))
    X1  = rng.normal([ 3.0, 0.0], 0.45, (n // 2, 2))
    X   = np.vstack([X0, X1])
    y   = np.array([0] * (n // 2) + [1] * (n // 2), dtype=int)
    return X, y


# ── layers ────────────────────────────────────────────────────────────────
class ReLU:
    def forward(self, x):
        self.x_in = x.copy()
        return np.clip(x, 0, None)
    def backward(self, grad):
        return np.where(self.x_in > 0, grad, 0)

class Sigmoid:
    def forward(self, x):
        self.y_out = 1.0 / (1.0 + np.exp(-x))
        return self.y_out
    def backward(self, grad):
        return self.y_out * (1.0 - self.y_out) * grad

class Linear:
    def __init__(self, n_in, n_out):
        self.weights = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.biases  = np.zeros(n_out)
    def forward(self, x):
        self.x_in = x
        return x @ self.weights + self.biases
    def backward(self, grad):
        self.grad_b = grad.mean(axis=0)
        self.grad_w = (self.x_in[:, :, None] @ grad[:, None, :]).mean(axis=0)
        return grad @ self.weights.T

class BinaryCrossEntropy:
    def forward(self, p, y):
        self.p = np.clip(p, 1e-7, 1.0 - 1e-7)
        self.y = y
        return -(y * np.log(self.p) + (1.0 - y) * np.log(1.0 - self.p))
    def backward(self):
        return (self.p - self.y) / (self.p * (1.0 - self.p))

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
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


def make_net():
    np.random.seed(0)
    return Model(
        [Linear(2, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 1), Sigmoid()],
        BinaryCrossEntropy()
    )


# ── training ──────────────────────────────────────────────────────────────
def train(model, X, y, lr=0.005, nb_epoch=3000, batch_size=32):
    Y   = y[:, None]
    rng = np.random.default_rng(1)
    for epoch in range(nb_epoch):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            model.loss(X[idx], Y[idx])
            model.backward()
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.weights -= lr * layer.grad_w
                    layer.biases  -= lr * layer.grad_b
    p   = model.forward(X).flatten()
    acc = ((p > 0.5) == y.astype(bool)).mean()
    print(f"  trained — acc={acc:.4f}")


# ── helpers ───────────────────────────────────────────────────────────────
def get_features(model, x):
    for layer in model.layers[:-2]:
        x = layer.forward(x)
    return x

def _make_grid(X, margin=5.0, h=0.05):
    xc, yc = X[:, 0].mean(), X[:, 1].mean()
    xx, yy = np.meshgrid(np.arange(xc - margin, xc + margin, h),
                         np.arange(yc - margin, yc + margin, h))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]

def _clean_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=9, fontweight="bold", fontfamily="DejaVu Sans")
    ax.tick_params(colors="#666666", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#222222")

def _pca2d(Z):
    """Project (N, D) matrix down to (N, 2) via PCA — no sklearn needed."""
    Z  = Z - Z.mean(axis=0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ Vt[:2].T

def _forward_to_logit(model, x):
    """Run all layers except the final Sigmoid — returns raw scalar logit."""
    for layer in model.layers[:-1]:
        x = layer.forward(x)
    return x.flatten()


# ── diagonal Laplace last layer ───────────────────────────────────────────
class BayesianLastLayer:
    """Diagonal Laplace on Linear(64,1).
    κ_i = p_i(1−p_i), H_j = 1/σ²_prior + Σ_i κ_i·φ_ij²"""
    def __init__(self, model, X_train, prior_std=1.0):
        last = model.layers[-2]
        self.W_mean = last.weights.copy()   # (64, 1)
        self.b_mean = last.biases.copy()    # (1,)

        phi   = get_features(model, X_train)
        p     = model.forward(X_train)
        kappa = p * (1.0 - p)

        W_prec     = (1.0 / prior_std**2) + (phi**2).T @ kappa
        self.W_std = 1.0 / np.sqrt(W_prec)
        b_prec     = (1.0 / prior_std**2) + float(kappa.sum())
        self.b_std = 1.0 / np.sqrt(b_prec)

    def sample(self, phi, n_samples=256, rng=None):
        """Returns (n_samples, N) of sigmoid probabilities."""
        if rng is None:
            rng = np.random.default_rng(0)
        out = np.zeros((n_samples, phi.shape[0]))
        for s in range(n_samples):
            W      = rng.normal(self.W_mean, self.W_std)
            b      = rng.normal(self.b_mean, self.b_std)
            logit  = (phi @ W + b).flatten()
            out[s] = 1.0 / (1.0 + np.exp(-logit))
        return out


# ── plots ─────────────────────────────────────────────────────────────────
def plot_prior_sweep(model, X, y, prior_stds=(0.3, 1.0, 3.0, 10.0)):
    """One row of LLLA confidence maps for each prior_std — pick by eye."""
    xx, yy, grid = _make_grid(X)
    phi = get_features(model, grid)

    fig, axes = plt.subplots(1, len(prior_stds), figsize=(14, 3.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("LLLA prior_std sweep  —  pick smallest with a visible effect",
                 color=FG, fontsize=9, fontweight="bold")

    for ax, s in zip(axes, prior_stds):
        bll   = BayesianLastLayer(model, X, prior_std=s)
        p_bay = bll.sample(phi, n_samples=128).mean(axis=0)
        conf  = np.maximum(p_bay, 1.0 - p_bay).reshape(xx.shape)
        bound = (p_bay > 0.5).reshape(xx.shape).astype(float)

        cf = ax.contourf(xx, yy, conf, levels=40, cmap="Blues", vmin=0.5, vmax=1.0)
        ax.contour(xx, yy, bound, levels=[0.5], colors="black", linewidths=1.0)
        ax.scatter(X[:, 0], X[:, 1],
                   c=[C0 if yi == 0 else C1 for yi in y],
                   s=10, edgecolors="white", linewidths=0.2, alpha=0.8, zorder=3)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
        _clean_ax(ax, f"σ_prior = {s}")

    plt.tight_layout()
    plt.show()


def plot_confidence_maps(model, bll, X, y, n_samples=256):
    """MAP vs LLLA side by side. Each panel uses its own decision boundary."""
    xx, yy, grid = _make_grid(X)
    phi = get_features(model, grid)

    p_map     = model.forward(grid).flatten()
    map_conf  = np.maximum(p_map, 1.0 - p_map).reshape(xx.shape)
    map_bound = (p_map > 0.5).reshape(xx.shape).astype(float)

    stack     = bll.sample(phi, n_samples=n_samples)
    p_bay     = stack.mean(axis=0)
    bay_conf  = np.maximum(p_bay, 1.0 - p_bay).reshape(xx.shape)
    bay_bound = (p_bay > 0.5).reshape(xx.shape).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    for ax, conf, bound, title in zip(
            axes,
            [map_conf,  bay_conf],
            [map_bound, bay_bound],
            ["MAP  CONFIDENCE", "LLLA  CONFIDENCE"]):
        cf = ax.contourf(xx, yy, conf, levels=50, cmap="Blues", vmin=0.5, vmax=1.0)
        ax.contour(xx, yy, bound, levels=[0.5], colors="black", linewidths=1.2)
        ax.scatter(X[:, 0], X[:, 1],
                   c=[C0 if yi == 0 else C1 for yi in y],
                   s=14, edgecolors="white", linewidths=0.3, alpha=0.85, zorder=3)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        _clean_ax(ax, title)

    plt.tight_layout()
    plt.show()


def plot_1d_probe(model, bll, x_range=(-9.0, 9.0), n_points=400, n_samples=256):
    """Confidence along the x-axis (y=0).
    Line = mean, band = ±3σ, both of per-sample confidence across posterior samples."""
    xs  = np.linspace(*x_range, n_points)
    pts = np.c_[xs, np.zeros_like(xs)]

    p_map    = model.forward(pts).flatten()
    map_conf = np.maximum(p_map, 1.0 - p_map)

    phi    = get_features(model, pts)
    stack  = bll.sample(phi, n_samples=n_samples)   # (S, N)
    conf_s = np.maximum(stack, 1.0 - stack)          # (S, N) — confidence per sample
    bay_conf = conf_s.mean(axis=0)                   # mean of per-sample confidence
    bay_std  = conf_s.std(axis=0)                    # std  of per-sample confidence

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("confidence along x-axis  (y = 0)", color=FG,
                 fontsize=10, fontweight="bold")

    axes[0].plot(xs, map_conf, color=C1, linewidth=1.5)
    axes[0].axhline(0.5, color=FG, linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_ylim(0.45, 1.02)
    _clean_ax(axes[0], "MAP")
    axes[0].set_xlabel("x", color="#666666", fontsize=8)

    axes[1].plot(xs, bay_conf, color=C1, linewidth=1.5)
    axes[1].fill_between(xs,
                         np.clip(bay_conf - 3 * bay_std, 0.5, 1.0),
                         np.clip(bay_conf + 3 * bay_std, 0.5, 1.0),
                         color=C1, alpha=0.2)
    axes[1].axhline(0.5, color=FG, linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1].set_ylim(0.45, 1.02)
    _clean_ax(axes[1], "LLLA  (mean ± 3σ across posterior samples)")
    axes[1].set_xlabel("x", color="#666666", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_feature_space(model, X, y):
    """Input space (left) vs PCA of 64D penultimate features (right).
    Shows what space the last-layer Gaussian is actually operating in."""
    phi  = get_features(model, X)     # (N, 64)
    proj = _pca2d(phi)                # (N, 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("input space vs penultimate feature space (PCA)",
                 color=FG, fontsize=9, fontweight="bold")

    colors = [C0 if yi == 0 else C1 for yi in y]

    axes[0].scatter(X[:, 0], X[:, 1], c=colors, s=14,
                    edgecolors="white", linewidths=0.3, alpha=0.85)
    _clean_ax(axes[0], "INPUT SPACE")
    axes[0].set_xlabel("x₁", color="#888888", fontsize=8)
    axes[0].set_ylabel("x₂", color="#888888", fontsize=8)

    axes[1].scatter(proj[:, 0], proj[:, 1], c=colors, s=14,
                    edgecolors="white", linewidths=0.3, alpha=0.85)
    _clean_ax(axes[1], "PENULTIMATE FEATURES  (PCA 2D)")
    axes[1].set_xlabel("PC1", color="#888888", fontsize=8)
    axes[1].set_ylabel("PC2", color="#888888", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_logit_surface(model, X, margin=5.0, h=0.06):
    """3D surface of the MAP logit (before sigmoid). Shows the piecewise-linear
    score the last-layer Gaussian is sampling slopes over."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
    xc, yc = X[:, 0].mean(), X[:, 1].mean()
    xx, yy = np.meshgrid(np.arange(xc - margin, xc + margin, h),
                         np.arange(yc - margin, yc + margin, h))
    grid   = np.c_[xx.ravel(), yy.ravel()]
    z      = _forward_to_logit(model, grid).reshape(xx.shape)
    z_clip = np.clip(z, -8, 8)

    norm       = plt.Normalize(vmin=-8, vmax=8)
    facecolors = plt.cm.coolwarm(norm(z_clip[:-1, :-1]))
    facecolors[..., 3] = 0.9

    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor(BG)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.plot_surface(xx, yy, z_clip, facecolors=facecolors,
                    linewidth=0.2, antialiased=True)
    ax.set_title("MAP logit surface", color=FG,
                 fontsize=9, fontweight="bold", pad=8)
    ax.set_zlabel("logit", fontsize=7)
    ax.tick_params(colors="#888888", labelsize=6)
    for attr in ("xaxis", "yaxis", "zaxis"):
        getattr(ax, attr).pane.fill = False
        getattr(ax, attr).pane.set_edgecolor("#dddddd")
    ax.set_zlim(-8, 8)
    plt.tight_layout()
    plt.show()


# ── run ───────────────────────────────────────────────────────────────────
PRIOR_STD = 1.0   # ← choose after inspecting plot_prior_sweep

X, y = make_blobs()
net  = make_net()
train(net, X, y)

plot_prior_sweep(net, X, y)
bll = BayesianLastLayer(net, X, prior_std=PRIOR_STD)
plot_confidence_maps(net, bll, X, y)
plot_1d_probe(net, bll)

# optional geometry views — uncomment to enable
# plot_feature_space(net, X, y)
# plot_logit_surface(net, X)
