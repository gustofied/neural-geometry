import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from data import make_radial_bands


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
        return self.y_out * (grad - (grad * self.y_out).sum(axis=1, keepdims=True))


class CrossEntropy:
    def forward(self, x, y):
        self.x_in = np.clip(x, 1e-8, 1.0)
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
        self.grad_w = self.x_in.T @ grad / self.x_in.shape[0]
        return grad @ self.weights.T

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
        return x

    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def loss(self, x, y):
        return self.cost.forward(self.forward(x), y)

    def backward(self):
        grad = self.cost.backward()
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)

def one_hot(y, n_classes=None):
    if n_classes is None:
        n_classes = int(y.max()) + 1
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


def train(model, X, y, lr, nb_epoch, batch_size=64):
    Y   = one_hot(y)
    rng = np.random.default_rng(0)
    loss_history = []
    acc_history  = []

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

        epoch_loss = running_loss / len(X_s)
        epoch_acc  = (model.predict(X) == y).mean()
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{nb_epoch}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

    return loss_history, acc_history

def build():
    np.random.seed(42)
    X, y, _ = make_radial_bands(
        n_samples=1600, band_radii=(0.55, 1.05, 1.55, 2.05),
        band_width=0.12, xy_noise=0.02, seed=42,
    )
    net = Model([Linear(2, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 2), Softmax()],
                CrossEntropy())
    loss_history, acc_history = train(net, X, y, lr=0.05, nb_epoch=2000)
    final_loss = loss_history[-1]
    final_acc  = (net.predict(X) == y).mean()
    print(f"Final accuracy: {final_acc:.4f}  loss: {final_loss:.4f}")
    return net, X, y, loss_history, acc_history

def _make_grid(X, h=0.008):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def _activation_mask(model, x, relu_indices):
    model.forward(x)
    masks = [model.layers[i].x_in > 0 for i in relu_indices]
    return np.concatenate(masks, axis=1)


def _region_ids(mask):
    _, ids = np.unique(mask, axis=0, return_inverse=True)
    return ids


def compute_geometry(model, X, h=0.008):
    xx, yy, grid = _make_grid(X, h)
    probs = model.predict_proba(grid)
    preds = np.argmax(probs, axis=1).reshape(xx.shape)
    conf  = probs.max(axis=1).reshape(xx.shape)

    r1_ids = _region_ids(_activation_mask(model, grid, [1]))
    r2_ids = _region_ids(_activation_mask(model, grid, [3]))
    jt_ids = _region_ids(_activation_mask(model, grid, [1, 3]))

    return {
        "xx": xx, "yy": yy, "grid": grid,
        "preds": preds, "conf": conf,
        "r1": r1_ids.reshape(xx.shape),
        "r2": r2_ids.reshape(xx.shape),
        "joint": jt_ids.reshape(xx.shape),
        "n1": len(np.unique(r1_ids)),
        "n2": len(np.unique(r2_ids)),
        "nj": len(np.unique(jt_ids)),
    }


BG         = "#07070d"
FG         = "#c0c0d0"

PINK_GLOW  = "#ff4fa3"   
PINK_NEON  = "#ff0f7b"  
PINK_HOT   = "#ff7cc4"  

CLASS0     = "#ff9e00"
CLASS1     = "#00c8e8"
EDGE_BRIGHT  = "#f3f1ea"  
EDGE_SOFT    = "#c9c4ba"  

REGION_MUTED = [
    "#143041",
    "#26183a",
    "#341b25",
    "#2a2d14",
    "#3a2414",
    "#18232d",
]

REGION_JOINT = [
    "#2e7ea0",
    "#5a4fd1",
    "#b04bd1",
    "#2faea1",
    "#c98a3b",
    "#7dad43",
    "#c4577c",
    "#5c86c9",
]

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
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a1a24")
        spine.set_linewidth(0.5)


def _class_colors(y):
    return [CLASS0 if yi == 0 else CLASS1 for yi in y]


def _scatter_data(ax, X, y, s=10, alpha=0.9):
    ax.scatter(X[:, 0], X[:, 1], c=_class_colors(y),
               s=s, edgecolors="none", alpha=alpha, zorder=5)


def _neon_boundary(ax, xx, yy, preds):
    ax.contour(xx, yy, preds, colors=PINK_GLOW, linewidths=4.0, alpha=0.10)
    ax.contour(xx, yy, preds, colors=PINK_GLOW, linewidths=2.4, alpha=0.20)
    ax.contour(xx, yy, preds, colors=PINK_NEON, linewidths=1.0, alpha=0.95)
    ax.contour(xx, yy, preds, colors=PINK_HOT,  linewidths=0.45, alpha=0.65)


def _region_mesh(ax, xx, yy, Z, palette=None):
    if palette is None:
        palette = REGION_MUTED
    n = int(Z.max()) + 1
    colors = [palette[i % len(palette)] for i in range(n)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n, 1), n)
    ax.pcolormesh(xx, yy, Z, cmap=cmap, norm=norm, rasterized=True)

def plot_training_curves(loss_history, acc_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor(BG)

    epochs = np.arange(len(loss_history))

    ax1.fill_between(epochs, loss_history, 0, color=PINK_NEON, alpha=0.10)
    ax1.plot(epochs, loss_history, color=PINK_GLOW, linewidth=2.0, alpha=0.10)
    ax1.plot(epochs, loss_history, color=PINK_NEON, linewidth=0.95, alpha=0.95)
    _clean_ax(ax1, "LOSS")

    ax2.fill_between(epochs, acc_history, 0, color=CLASS1, alpha=0.035)
    ax2.plot(epochs, acc_history, color=CLASS1, linewidth=0.95, alpha=0.90)
    ax2.set_ylim(0.0, 1.02)
    _clean_ax(ax2, "ACCURACY")

    plt.tight_layout()
    return fig


def plot_decision_boundary(geo, X, y):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    _neon_boundary(ax, geo["xx"], geo["yy"], geo["preds"])
    _scatter_data(ax, X, y)
    _clean_ax(ax, "DECISION BOUNDARY")
    plt.tight_layout()
    return fig


def plot_regions(geo, X, y):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor(BG)
    xx, yy = geo["xx"], geo["yy"]

    for ax, key, n, label, pal, lw, la, edge in [
        (axes[0], "r1",    geo["n1"], "LAYER 1", REGION_MUTED, 0.70, 0.80, "#f3f1ea"),
        (axes[1], "r2",    geo["n2"], "LAYER 2", REGION_MUTED, 0.30, 0.38, "#d8d3c7"),
        (axes[2], "joint", geo["nj"], "JOINT",   REGION_JOINT, 0.18, 0.22, "#b8b3c8"),
    ]:
        _region_mesh(ax, xx, yy, geo[key], pal)
        ax.contour(xx, yy, geo[key], colors=edge, linewidths=lw, alpha=la)
        _clean_ax(ax, f"{label}  ({n})")

    _neon_boundary(axes[3], xx, yy, geo["preds"])
    _scatter_data(axes[3], X, y, s=6)
    _clean_ax(axes[3], "DECISION BOUNDARY")

    plt.tight_layout()
    return fig


def plot_radial_probe(model, n_radii=8, r_max=5.0, n_points=300):
    angles = np.linspace(0, 2 * np.pi, n_radii, endpoint=False)
    radii  = np.linspace(0.0, r_max, n_points)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(BG)

    for theta in angles:
        pts    = np.c_[radii * np.cos(theta), radii * np.sin(theta)]
        logits = model.logits(pts)
        conf   = model.predict_proba(pts).max(axis=1)
        diff   = logits[:, 1] - logits[:, 0]

        axes[0].plot(radii, diff, color=CLASS1, linewidth=1.8, alpha=0.06)
        axes[0].plot(radii, diff, color=CLASS1, linewidth=0.7, alpha=0.32)
        axes[1].plot(radii, conf, color=CLASS1, linewidth=1.8, alpha=0.06)
        axes[1].plot(radii, conf, color=CLASS1, linewidth=0.7, alpha=0.32)

    for r in (0.55, 1.05, 1.55, 2.05):
        axes[0].axvline(r, color=PINK_NEON, linewidth=0.6, linestyle="--", alpha=0.35)
        axes[1].axvline(r, color=PINK_NEON, linewidth=0.6, linestyle="--", alpha=0.35)

    _clean_ax(axes[0], "LOGIT DIFFERENCE vs RADIUS")
    _clean_ax(axes[1], "CONFIDENCE vs RADIUS")
    axes[0].set_xlabel("radius", color="#444450", fontsize=8, fontfamily="monospace")
    axes[1].set_xlabel("radius", color="#444450", fontsize=8, fontfamily="monospace")
    axes[1].set_ylim(0.4, 1.02)

    plt.tight_layout()
    return fig


def plot_confidence_map(geo, X, y):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    cf = ax.contourf(geo["xx"], geo["yy"], geo["conf"],
                     levels=50, cmap=_CONF_CMAP, vmin=0.5, vmax=1.0, alpha=0.95)
    ax.contour(geo["xx"], geo["yy"], geo["conf"],
               levels=[0.55, 0.6, 0.7, 0.8, 0.9],
               colors=PINK_NEON, linewidths=0.3, alpha=0.15)
    cb = plt.colorbar(cf, ax=ax, fraction=0.032, pad=0.03)
    cb.ax.tick_params(colors="#333340", labelsize=6)
    cb.outline.set_edgecolor("#101018")  # ty:ignore[call-non-callable]
    _scatter_data(ax, X, y, s=6, alpha=0.7)
    _clean_ax(ax, "SOFTMAX CONFIDENCE")
    plt.tight_layout()
    return fig


def _save(fig, path):
    import os as os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.1)
    print(f"  saved {path}")


def run_all(save=False):
    import os as _os
    net, X, y, loss_history, acc_history = build()
    geo = compute_geometry(net, X)

    assets = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "assets")

    figs = {
        "training":   plot_training_curves(loss_history, acc_history),
        "radial":     plot_radial_probe(net),
        "confidence": plot_confidence_map(geo, X, y),
        "regions":    plot_regions(geo, X, y),
    }

    if save:
        for name, fig in figs.items():
            _save(fig, _os.path.join(assets, f"relu_{name}.png"))

    for fig in figs.values():
        fig.show()

    plt.show()


if __name__ == "__main__":
    import os, sys
    run_all(save="--save" in sys.argv)
