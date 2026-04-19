import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
        self.grad_w = self.x_in.T @ grad / self.x_in.shape[0]
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
    def logits(self, x):
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return x.flatten()
    def loss(self, x, y):
        return self.cost.forward(self.forward(x), y)
    def backward(self):
        grad = self.cost.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

def make_blobs(n=300, seed=42):
    rng = np.random.default_rng(seed)
    X0  = rng.normal([-3.0, 0.0], 0.45, (n // 2, 2))
    X1  = rng.normal([ 3.0, 0.0], 0.45, (n // 2, 2))
    X   = np.vstack([X0, X1])
    y   = np.array([0] * (n // 2) + [1] * (n // 2), dtype=int)
    return X, y


def make_net():
    np.random.seed(0)
    return Model(
        [Linear(2, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 1), Sigmoid()],
        BinaryCrossEntropy()
    )


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
    print(f"  trained, acc={acc:.4f}")


def get_features(model, x):
    for layer in model.layers[:-2]:
        x = layer.forward(x)
    return x

class LastLayerLaplace:
    """Diagonal Gaussian over the final linear layer.
    Precision per weight: 1/sigma_prior^2 + sum_i kappa_i * phi_ij^2
    where kappa_i = p_i(1 - p_i) is the logistic curvature."""

    def __init__(self, model, X_train, prior_std=1.0):
        last = model.layers[-2]
        self.W_mean = last.weights.copy()
        self.b_mean = last.biases.copy()

        phi   = get_features(model, X_train)
        p     = model.forward(X_train)
        kappa = p * (1.0 - p)

        W_prec     = (1.0 / prior_std**2) + (phi**2).T @ kappa
        self.W_std = 1.0 / np.sqrt(W_prec)
        b_prec     = (1.0 / prior_std**2) + float(kappa.sum())
        self.b_std = 1.0 / np.sqrt(b_prec)

    def sample_logits(self, phi, n_samples=256, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        out = np.zeros((n_samples, phi.shape[0]))
        for s in range(n_samples):
            W = rng.normal(self.W_mean, self.W_std)
            b = rng.normal(self.b_mean, self.b_std)
            out[s] = (phi @ W + b).flatten()
        return out

    def sample_probs(self, phi, n_samples=256, rng=None):
        logits = self.sample_logits(phi, n_samples=n_samples, rng=rng)
        return 1.0 / (1.0 + np.exp(-logits))


PRIOR_STD = 1.0

def build(prior_std=PRIOR_STD):
    X, y = make_blobs()
    net  = make_net()
    train(net, X, y)
    llla = LastLayerLaplace(net, X, prior_std=prior_std)
    return net, llla, X, y

def _make_grid(X, margin=5.0, h=0.05):
    xc, yc = X[:, 0].mean(), X[:, 1].mean()
    xx, yy = np.meshgrid(np.arange(xc - margin, xc + margin, h),
                         np.arange(yc - margin, yc + margin, h))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]

def compute_fields(model, llla, X, h=0.05, n_samples=256):
    xx, yy, grid = _make_grid(X, h=h)
    phi = get_features(model, grid)

    p_map    = model.forward(grid).flatten()
    map_conf = np.maximum(p_map, 1.0 - p_map).reshape(xx.shape)
    map_pred = (p_map > 0.5).reshape(xx.shape).astype(float)

    probs    = llla.sample_probs(phi, n_samples=n_samples)
    p_bay    = probs.mean(axis=0)
    bay_conf = np.maximum(p_bay, 1.0 - p_bay).reshape(xx.shape)
    bay_pred = (p_bay > 0.5).reshape(xx.shape).astype(float)

    return {
        "xx": xx, "yy": yy,
        "map_prob": p_map.reshape(xx.shape),
        "map_conf": map_conf, "map_pred": map_pred,
        "bay_prob": p_bay.reshape(xx.shape),
        "bay_conf": bay_conf, "bay_pred": bay_pred,
    }

BG         = "#07070d"
FG         = "#c0c0d0"

PINK_GLOW  = "#ff4fa3"
PINK_NEON  = "#ff0f7b"
PINK_HOT   = "#ff7cc4"

CLASS0     = "#c88040"
CLASS1     = "#40a0b8"

_CONF_CMAP = LinearSegmentedColormap.from_list("conf", [
    (0.00, "#ff5eb0"),
    (0.12, "#ff207f"),
    (0.25, "#8a184d"),
    (0.45, "#24101a"),
    (1.00, "#07070d"),
])

def _clean_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=10, fontweight="bold", pad=10,
                 fontfamily="monospace")
    ax.tick_params(colors="#444450", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1a24")
        sp.set_linewidth(0.5)


def _class_colors(y):
    return [CLASS0 if yi == 0 else CLASS1 for yi in y]


def _scatter_data(ax, X, y, s=10, alpha=0.9):
    ax.scatter(X[:, 0], X[:, 1], c=_class_colors(y),
               s=s, edgecolors="none", alpha=alpha, zorder=5)


def _neon_boundary(ax, xx, yy, pred):
    ax.contour(xx, yy, pred, levels=[0.5], colors=PINK_GLOW, linewidths=2.0, alpha=0.08)
    ax.contour(xx, yy, pred, levels=[0.5], colors=PINK_NEON, linewidths=0.7, alpha=0.75)

def plot_confidence_maps(fields, X, y):
    """MAP vs LLLA confidence side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    xx, yy = fields["xx"], fields["yy"]

    for ax, conf, pred, title in [
        (axes[0], fields["map_conf"], fields["map_pred"], "MAP CONFIDENCE"),
        (axes[1], fields["bay_conf"], fields["bay_pred"], "LLLA CONFIDENCE"),
    ]:
        cf = ax.contourf(xx, yy, conf, levels=50, cmap=_CONF_CMAP,
                         vmin=0.5, vmax=1.0, alpha=0.95)
        _neon_boundary(ax, xx, yy, pred)
        _scatter_data(ax, X, y, s=6, alpha=0.68)
        cb = plt.colorbar(cf, ax=ax, fraction=0.032, pad=0.03)
        cb.ax.tick_params(colors="#333340", labelsize=6)
        cb.outline.set_edgecolor("#101018")
        _clean_ax(ax, title)

    plt.tight_layout()
    return fig


def plot_1d_probe(model, llla, x_range=(-9.0, 9.0), n_points=400, n_samples=256):
    """Confidence along x-axis (y=0), slicing through the data gap
    and into far-field space where MAP stays overconfident."""
    xs  = np.linspace(x_range[0], x_range[1], n_points)
    pts = np.c_[xs, np.zeros_like(xs)]

    p_map    = model.forward(pts).flatten()
    map_conf = np.maximum(p_map, 1.0 - p_map)

    phi      = get_features(model, pts)
    probs    = llla.sample_probs(phi, n_samples=n_samples)
    conf_s   = np.maximum(probs, 1.0 - probs)
    bay_conf = conf_s.mean(axis=0)
    bay_std  = conf_s.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(BG)

    axes[0].plot(xs, map_conf, color=PINK_GLOW, linewidth=1.8, alpha=0.08)
    axes[0].plot(xs, map_conf, color=PINK_NEON, linewidth=0.9, alpha=0.85)
    axes[0].axhline(0.5, color=FG, linewidth=0.6, linestyle="--", alpha=0.3)
    axes[0].set_ylim(0.45, 1.02)
    _clean_ax(axes[0], "MAP")
    axes[0].set_xlabel("x", color="#444450", fontsize=8, fontfamily="monospace")

    axes[1].fill_between(xs,
                         np.clip(bay_conf - 3 * bay_std, 0.5, 1.0),
                         np.clip(bay_conf + 3 * bay_std, 0.5, 1.0),
                         color=PINK_GLOW, alpha=0.10)
    axes[1].plot(xs, bay_conf, color=PINK_GLOW, linewidth=1.8, alpha=0.08)
    axes[1].plot(xs, bay_conf, color=PINK_NEON, linewidth=0.9, alpha=0.85)
    axes[1].axhline(0.5, color=FG, linewidth=0.6, linestyle="--", alpha=0.3)
    axes[1].set_ylim(0.45, 1.02)
    _clean_ax(axes[1], "LLLA")
    axes[1].set_xlabel("x", color="#444450", fontsize=8, fontfamily="monospace")

    for ax in axes:
        ax.axvspan(-4.0, -2.0, color=CLASS0, alpha=0.05)
        ax.axvspan( 2.0,  4.0, color=CLASS1, alpha=0.05)

    plt.tight_layout()
    return fig


def plot_prior_sweep(model, X, y, prior_stds=(0.3, 1.0, 3.0, 10.0)):
    """LLLA confidence for different prior scales."""
    xx, yy, grid = _make_grid(X)
    phi = get_features(model, grid)

    fig, axes = plt.subplots(1, len(prior_stds), figsize=(14, 3.5))
    fig.patch.set_facecolor(BG)

    for ax, s in zip(axes, prior_stds):
        llla  = LastLayerLaplace(model, X, prior_std=s)
        p_bay = llla.sample_probs(phi, n_samples=128).mean(axis=0)
        conf  = np.maximum(p_bay, 1.0 - p_bay).reshape(xx.shape)
        pred  = (p_bay > 0.5).reshape(xx.shape).astype(float)

        cf = ax.contourf(xx, yy, conf, levels=40, cmap=_CONF_CMAP,
                         vmin=0.5, vmax=1.0, alpha=0.95)
        _neon_boundary(ax, xx, yy, pred)
        _scatter_data(ax, X, y, s=8)
        plt.colorbar(cf, ax=ax, fraction=0.032, pad=0.03).ax.tick_params(
            colors="#333340", labelsize=5)
        _clean_ax(ax, f"\u03c3_prior = {s}")

    plt.tight_layout()
    return fig


def _save(fig, path, dpi=200):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.1)
    print(f"  saved {path}")


def run_all(save=False):
    net, llla, X, y = build()
    fields = compute_fields(net, llla, X)

    assets = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    figs = {
        "confidence": plot_confidence_maps(fields, X, y),
        "probe":      plot_1d_probe(net, llla),
        "sweep":      plot_prior_sweep(net, X, y),
    }

    if save:
        for name, fig in figs.items():
            _save(fig, os.path.join(assets, f"bayes_{name}.png"))

    for fig in figs.values():
        fig.show()

    plt.show()


if __name__ == "__main__":
    run_all(save="--save" in sys.argv)
