"""
relu_train.py — partition evolution during training

Saves snapshots of the joint activation partition at checkpoint epochs.
Early on the plane has a few large linear cells; over time the partition
becomes finer and the radial decision boundary emerges from many straight cuts.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from data import make_radial_bands
from relu import (
    ReLU, Softmax, CrossEntropy, Linear, Model,
    one_hot, _make_grid, _activation_mask, _region_ids,
)


# ── palette ──────────────────────────────────────────────────────────────
BG = "#07070d"
FG = "#c0c0d0"

FILL_PALETTE = [
    "#3a8cb0", "#7060d0", "#c050c0", "#50c8c0",
    "#d0a040", "#80c040", "#d06080", "#6898d0",
    "#50b080", "#b07050", "#9080e0", "#c0a070",
    "#60b0a0", "#a06090", "#80b060", "#7080b0",
]

EDGE      = "#e8e4dc"
EDGE_SOFT = "#b0a898"
BOUNDARY  = "#ff0f7b"


# ── checkpoints ──────────────────────────────────────────────────────────
CHECKPOINTS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000]


def train_with_snapshots(model, X, y, lr=0.05, nb_epoch=2000, batch_size=64,
                         checkpoints=None, grid_h=0.012):
    if checkpoints is None:
        checkpoints = CHECKPOINTS

    Y   = one_hot(y)
    rng = np.random.default_rng(0)
    xx, yy, grid = _make_grid(X, h=grid_h)

    snapshots = []

    def snapshot(epoch):
        jt_ids = _region_ids(_activation_mask(model, grid, [1, 3]))
        logits = model.logits(grid)
        preds  = np.argmax(logits, axis=1).reshape(xx.shape)
        return {
            "epoch": epoch,
            "joint": jt_ids.reshape(xx.shape),
            "preds": preds,
            "n_regions": len(np.unique(jt_ids)),
            "xx": xx, "yy": yy,
        }

    if 0 in checkpoints:
        snapshots.append(snapshot(0))

    for epoch in range(1, nb_epoch + 1):
        perm     = rng.permutation(len(X))
        X_s, Y_s = X[perm], Y[perm]

        for i in range(0, len(X_s), batch_size):
            xb, yb = X_s[i:i+batch_size], Y_s[i:i+batch_size]
            model.loss(xb, yb)
            model.backward()
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.weights -= lr * layer.grad_w
                    layer.biases  -= lr * layer.grad_b

        if epoch in checkpoints:
            acc = (model.predict(X) == y).mean()
            print(f"  epoch {epoch:4d}  acc={acc:.4f}  ", end="")
            snap = snapshot(epoch)
            print(f"regions={snap['n_regions']}")
            snapshots.append(snap)

    return snapshots


# ── rendering ────────────────────────────────────────────────────────────
def render_frame(snap, X, y, figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xx, yy = snap["xx"], snap["yy"]
    Z      = snap["joint"]
    preds  = snap["preds"]
    n      = int(Z.max()) + 1

    # region fills
    colors = [FILL_PALETTE[i % len(FILL_PALETTE)] for i in range(n)]
    cmap   = ListedColormap(colors)
    norm   = BoundaryNorm(np.arange(-0.5, n, 1), n)
    ax.pcolormesh(xx, yy, Z, cmap=cmap, norm=norm, rasterized=True)

    # region edges: white, weight depends on density
    edge_lw    = max(0.15, 0.6 - n * 0.00005)
    edge_alpha = max(0.20, 0.7 - n * 0.00005)
    edge_color = EDGE if n < 2000 else EDGE_SOFT
    ax.contour(xx, yy, Z, colors=edge_color, linewidths=edge_lw, alpha=edge_alpha)

    # decision boundary: thin pink
    ax.contour(xx, yy, preds, colors=BOUNDARY, linewidths=0.8, alpha=0.85)

    # data points (very subtle)
    c = ["#b08068" if yi == 0 else "#6898a8" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=c, s=3, alpha=0.4, edgecolors="none", zorder=5)

    # epoch label
    ax.text(0.03, 0.97, f"epoch {snap['epoch']}",
            transform=ax.transAxes, color=FG, fontsize=9,
            fontfamily="monospace", fontweight="bold",
            va="top", ha="left", alpha=0.7)
    ax.text(0.03, 0.92, f"{snap['n_regions']} regions",
            transform=ax.transAxes, color=FG, fontsize=7,
            fontfamily="monospace", va="top", ha="left", alpha=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")

    plt.tight_layout(pad=0.2)
    return fig


def render_strip(snapshots, X, y, indices=None):
    """Side-by-side strip of selected checkpoints."""
    if indices is None:
        indices = [0, len(snapshots)//3, 2*len(snapshots)//3, -1]
    selected = [snapshots[i] for i in indices]

    fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 5))
    fig.patch.set_facecolor(BG)

    for ax, snap in zip(axes, selected):
        ax.set_facecolor(BG)
        xx, yy = snap["xx"], snap["yy"]
        Z      = snap["joint"]
        preds  = snap["preds"]
        n      = int(Z.max()) + 1

        colors = [FILL_PALETTE[i % len(FILL_PALETTE)] for i in range(n)]
        cmap   = ListedColormap(colors)
        norm   = BoundaryNorm(np.arange(-0.5, n, 1), n)
        ax.pcolormesh(xx, yy, Z, cmap=cmap, norm=norm, rasterized=True)

        edge_lw    = max(0.15, 0.6 - n * 0.00005)
        edge_alpha = max(0.20, 0.7 - n * 0.00005)
        edge_color = EDGE if n < 2000 else EDGE_SOFT
        ax.contour(xx, yy, Z, colors=edge_color, linewidths=edge_lw, alpha=edge_alpha)
        ax.contour(xx, yy, preds, colors=BOUNDARY, linewidths=0.8, alpha=0.85)

        c = ["#b08068" if yi == 0 else "#6898a8" for yi in y]
        ax.scatter(X[:, 0], X[:, 1], c=c, s=2, alpha=0.3, edgecolors="none", zorder=5)

        ax.text(0.03, 0.97, f"epoch {snap['epoch']}",
                transform=ax.transAxes, color=FG, fontsize=9,
                fontfamily="monospace", fontweight="bold",
                va="top", ha="left", alpha=0.7)
        ax.text(0.03, 0.92, f"{snap['n_regions']} regions",
                transform=ax.transAxes, color=FG, fontsize=7,
                fontfamily="monospace", va="top", ha="left", alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_aspect("equal")

    plt.tight_layout(pad=0.3)
    return fig


# ── run ──────────────────────────────────────────────────────────────────
def run(save=False):
    np.random.seed(42)
    X, y, _ = make_radial_bands(
        n_samples=1600, band_radii=(0.55, 1.05, 1.55, 2.05),
        band_width=0.12, xy_noise=0.02, seed=42,
    )
    net = Model([Linear(2, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 2), Softmax()],
                CrossEntropy())

    snapshots = train_with_snapshots(net, X, y)

    assets = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    if save:
        os.makedirs(assets, exist_ok=True)
        for snap in snapshots:
            fig = render_frame(snap, X, y)
            path = os.path.join(assets, f"train_{snap['epoch']:04d}.png")
            fig.savefig(path, dpi=200, facecolor=BG,
                        bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            print(f"  saved {path}")

        # strip of 4 key moments
        strip = render_strip(snapshots, X, y, indices=[0, 3, 7, -1])
        path = os.path.join(assets, "train_strip.png")
        strip.savefig(path, dpi=200, facecolor=BG,
                      bbox_inches="tight", pad_inches=0.02)
        plt.close(strip)
        print(f"  saved {path}")
    else:
        strip = render_strip(snapshots, X, y, indices=[0, 3, 7, -1])
        strip.show()
        plt.show()


if __name__ == "__main__":
    import sys
    run(save="--save" in sys.argv)
