import numpy as np
from nn1data import make_radial_bands
import matplotlib.pyplot as plt


class ReLU():
    def forward(self, x):
        self.x_in = np.copy(x)
        return np.clip(x,0,None)

    def backward(self, grad):
        return np.where(self.x_in>0,grad,0) 

class Sigmoid():
    def forward(self, x):
        self.y_out = np.exp(x) / (1. + np.exp(x)) # wee changeup writing this term the more common way with diviining on 1 + np.exp(-x) will mitigaate the issue of overlfow when im doing np.exp(x) as it can become very largey
        return self.y_out

    def backward(self, grad):
        return self.y_out * (1. - self.y_out) * grad

class Softmax():
    def forward(self, x):
        exp = np.exp(x - x.max(axis=1, keepdims=True))  # subtract max for numerical stability
        self.y_out = exp / exp.sum(axis=1, keepdims=True)
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

net = Model([Linear(2, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 2), Softmax()], CrossEntropy())

def train(model, X, Y, y, lr, nb_epoch, batch_size=64):
    rng = np.random.default_rng(0)
    loss_history = []
    acc_history  = []

    for epoch in range(nb_epoch):
        perm = rng.permutation(len(X))
        X_s, Y_s = X[perm], Y[perm]

        running_loss = 0.
        for i in range(0, len(X_s), batch_size):
            xb, yb = X_s[i:i+batch_size], Y_s[i:i+batch_size]
            running_loss += model.loss(xb, yb).sum()
            model.backward()
            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.weights -= lr * layer.grad_w
                    layer.biases -= lr * layer.grad_b

        epoch_loss = running_loss / len(X_s)
        preds = np.argmax(model.forward(X), axis=1)
        epoch_acc = (preds == y).mean()

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{nb_epoch}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

    return loss_history, acc_history

X, y, Y = make_radial_bands(
    n_samples=1600,
    band_radii=(0.55, 1.05, 1.55, 2.05),
    band_width=0.12,
    xy_noise=0.02,
    seed=42,
)

loss_history, acc_history = train(net, X, Y, y, lr=0.05, nb_epoch=2000)

out = net.forward(X)
preds = np.argmax(out, axis=1)
print(f"Final accuracy: {(preds == y).mean():.4f}")

# dark geometric palette
BG           = "#0c0c10"
FG           = "#b8b8c8"
BOUNDARY_C   = "#e8004a"   # neon crimson — decision boundary
C0           = "#b87c00"   # amber — class 0 fill
C1           = "#1a7a8a"   # slate teal — class 1 fill
REGION_WIRE  = "#a0c0b0"   # cool mint — region boundary lines
REGION_COLORS = [
    "#2a5f6e", "#5c3f6a", "#6a5c20", "#3a4a5a",
    "#1e5c5c", "#6a4a28", "#304858", "#5a3848",
    "#485830", "#385068", "#604038", "#305848",
    "#504830", "#304a68", "#485a50", "#583a58",
    "#385848", "#503848", "#445838", "#384858",
]

def plot_training_curves(loss_history, acc_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor(BG)

    ax1.plot(loss_history, color=BOUNDARY_C, linewidth=1.2)
    ax1.set_facecolor(BG)
    ax1.set_title("LOSS", color=FG, fontsize=10, fontweight="bold", fontfamily="DejaVu Sans")
    ax1.tick_params(colors="#666666", labelsize=7)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#222222")

    ax2.plot(acc_history, color=C1, linewidth=1.2)
    ax2.set_facecolor(BG)
    ax2.set_title("ACCURACY", color=FG, fontsize=10, fontweight="bold", fontfamily="DejaVu Sans")
    ax2.tick_params(colors="#666666", labelsize=7)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#222222")

    plt.tight_layout()
    plt.show()



def plot_confidence_map(model, X, y, h=0.008):
    from matplotlib.colors import ListedColormap
    xx, yy, grid = _make_grid_fn(X, h)
    probs = model.forward(grid)
    conf = probs.max(axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    cf = ax.contourf(xx, yy, conf, levels=50, cmap="plasma", alpha=0.9)
    ax.contourf(xx, yy, conf, levels=[0.99, 1.01], colors=BOUNDARY_C, alpha=0.3)
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=FG, labelsize=7)
    ax.scatter(X[:, 0], X[:, 1], c=[C0 if yi == 0 else C1 for yi in y],
               s=8, edgecolors=BG, linewidths=0.1, alpha=0.6)
    _clean_ax(ax, "MAX-SOFTMAX CONFIDENCE")
    plt.tight_layout()
    plt.show()


def plot_radial_probe(model, n_radii=8, r_max=5.0, n_points=300):
    angles = np.linspace(0, 2 * np.pi, n_radii, endpoint=False)
    radii  = np.linspace(0.0, r_max, n_points)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(BG)

    for theta in angles:
        pts = np.c_[radii * np.cos(theta), radii * np.sin(theta)]
        logits = forward_to_logits(model, pts)
        probs  = model.forward(pts)
        conf   = probs.max(axis=1)
        diff   = logits[:, 1] - logits[:, 0]

        axes[0].plot(radii, diff,  color=REGION_WIRE, linewidth=0.8, alpha=0.6)
        axes[1].plot(radii, conf,  color=REGION_WIRE, linewidth=0.8, alpha=0.6)

    band_radii = (0.55, 1.05, 1.55, 2.05)
    for r in band_radii:
        axes[0].axvline(r, color=BOUNDARY_C, linewidth=0.6, linestyle="--", alpha=0.5)
        axes[1].axvline(r, color=BOUNDARY_C, linewidth=0.6, linestyle="--", alpha=0.5)

    for ax, title in zip(axes, ["LOGIT DIFFERENCE vs RADIUS", "CONFIDENCE vs RADIUS"]):
        ax.set_facecolor(BG)
        ax.set_title(title, color=FG, fontsize=10, fontweight="bold", fontfamily="DejaVu Sans")
        ax.tick_params(colors="#666666", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#222222")
        ax.set_xlabel("radius", color="#666666", fontsize=8)
    axes[1].set_ylim(0.4, 1.02)

    plt.tight_layout()
    plt.show()


def _make_wide_grid(X, margin=6.0, h=0.025):
    xc, yc = X[:, 0].mean(), X[:, 1].mean()
    xx, yy = np.meshgrid(np.arange(xc - margin, xc + margin, h),
                         np.arange(yc - margin, yc + margin, h))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]

def _make_grid_fn(X, h=0.008):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]

def _make_grid(X, h=0.008):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]

def _clean_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=10, fontweight="bold", pad=10,
                 fontfamily="DejaVu Sans")
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")
        spine.set_linewidth(0.6)

def plot_decision_boundary(model, X, y, h=0.008):
    from matplotlib.colors import ListedColormap
    xx, yy, grid = _make_grid(X, h)
    preds = np.argmax(model.forward(grid), axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    ax.contourf(xx, yy, preds, cmap=ListedColormap([C0, C1]), alpha=0.4)
    ax.contour(xx, yy, preds, colors=BOUNDARY_C, linewidths=1.5, alpha=1.0)
    ax.scatter(X[:, 0], X[:, 1], c=[C0 if yi == 0 else C1 for yi in y],
               s=12, edgecolors=BG, linewidths=0.2, alpha=0.85)
    _clean_ax(ax, "DECISION BOUNDARY")
    plt.tight_layout()
    plt.show()



def _region_colors(n):
    pool = (REGION_COLORS * ((n // len(REGION_COLORS)) + 1))
    step = max(1, len(REGION_COLORS) // 3)
    indices = [(i * step) % len(pool) for i in range(n)]
    return [pool[i] for i in indices]

def _region_mesh(ax, xx, yy, Z, alpha=0.75):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    n = int(Z.max()) + 1
    cmap = ListedColormap(_region_colors(n))
    norm = BoundaryNorm(np.arange(-0.5, n, 1), n)
    ax.pcolormesh(xx, yy, Z, cmap=cmap, norm=norm, alpha=alpha)

def _get_regions(model, grid, relu_idx):
    model.forward(grid)
    mask = model.layers[relu_idx].x_in > 0
    unique_patterns, region_ids = np.unique(mask, axis=0, return_inverse=True)
    return unique_patterns, region_ids

def plot_activation_regions(model, X, y=None, relu_idx=1, h=0.008):
    from matplotlib.colors import ListedColormap
    xx, yy, grid = _make_grid(X, h)
    unique_patterns, region_ids = _get_regions(model, grid, relu_idx)
    n = len(unique_patterns)

    Z = region_ids.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    _region_mesh(ax, xx, yy, Z)
    ax.contour(xx, yy, Z, colors=REGION_WIRE, linewidths=0.6, alpha=0.7)
    _clean_ax(ax, f"LAYER {relu_idx} REGIONS  ({n})")
    plt.tight_layout()
    plt.show()
    return unique_patterns, region_ids



def plot_decision_and_regions(model, X, y, h=0.008):
    from matplotlib.colors import ListedColormap
    xx, yy, grid = _make_grid(X, h)
    out = model.forward(grid)
    preds = np.argmax(out, axis=1).reshape(xx.shape)

    _, r1_ids = _get_regions(model, grid, relu_idx=1)
    _, r2_ids = _get_regions(model, grid, relu_idx=3)
    r1 = r1_ids.reshape(xx.shape)
    r2 = r2_ids.reshape(xx.shape)
    n1, n2 = len(np.unique(r1_ids)), len(np.unique(r2_ids))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)

    axes[0].contourf(xx, yy, preds, cmap=ListedColormap([C0, C1]), alpha=0.4)
    axes[0].contour(xx, yy, preds, colors=BOUNDARY_C, linewidths=1.5, alpha=1.0)
    axes[0].scatter(X[:, 0], X[:, 1], c=[C0 if yi == 0 else C1 for yi in y],
                    s=12, edgecolors=BG, linewidths=0.2, alpha=0.85)
    _clean_ax(axes[0], "DECISION BOUNDARY")

    _region_mesh(axes[1], xx, yy, r1)
    axes[1].contour(xx, yy, r1, colors=REGION_WIRE, linewidths=0.6, alpha=0.7)
    _clean_ax(axes[1], f"LAYER 1 REGIONS  ({n1})")

    _region_mesh(axes[2], xx, yy, r2)
    axes[2].contour(xx, yy, r2, colors=REGION_WIRE, linewidths=0.6, alpha=0.7)
    _clean_ax(axes[2], f"LAYER 2 REGIONS  ({n2})")

    plt.tight_layout()
    plt.show()



# joint activation regions — real piecewise-linear partition of the full network
def get_joint_regions(model, grid):
    model.forward(grid)
    mask1 = model.layers[1].x_in > 0   # (N, 64)
    mask2 = model.layers[3].x_in > 0   # (N, 64)
    joint_mask = np.concatenate([mask1, mask2], axis=1)  # (N, 128)
    unique_patterns, region_ids = np.unique(joint_mask, axis=0, return_inverse=True)
    return unique_patterns, region_ids

def plot_joint_activation_regions(model, X, h=0.008):
    from matplotlib.colors import ListedColormap
    xx, yy, grid = _make_grid(X, h)
    unique_patterns, region_ids = get_joint_regions(model, grid)
    n = len(unique_patterns)

    Z = region_ids.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    _region_mesh(ax, xx, yy, Z)
    ax.contour(xx, yy, Z, colors=REGION_WIRE, linewidths=0.5, alpha=0.7)
    _clean_ax(ax, f"JOINT REGIONS  ({n})")
    plt.tight_layout()
    plt.show()
    return unique_patterns, region_ids



def forward_to_logits(model, x):
    for layer in model.layers[:-1]:   # skip softmax
        x = layer.forward(x)
    return x

def get_features(model, x):
    """Penultimate representation — output of the last hidden ReLU, input to the final linear layer.
    This is the handoff point for Bayesian last-layer replacement."""
    for layer in model.layers[:-2]:   # skip final Linear + Softmax
        x = layer.forward(x)
    return x


def _softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class BayesianLastLayer:
    """Diagonal Laplace approximation over the final linear layer.

    Posterior variance per weight is data-dependent: inversely proportional
    to the curvature of the CE loss at the MAP solution. Features that are
    large and confidently-predicted directions get small variance; poorly
    constrained directions stay broad.
    """
    def __init__(self, model, X_train, prior_std=1.0):
        last = model.layers[-2]           # final Linear(64, 2)
        self.W_mean = last.weights.copy() # (64, 2)
        self.b_mean = last.biases.copy()  # (2,)

        phi   = get_features(model, X_train)  # (N, 64)
        probs = model.forward(X_train)        # (N, 2)

        # curvature of CE loss w.r.t. logit k at point i: p_ik * (1 - p_ik)
        kappa = probs * (1.0 - probs)         # (N, 2)

        # diagonal Hessian for W_{jk}: 1/sigma^2 + sum_i kappa_ik * phi_ij^2
        phi_sq      = phi ** 2                        # (N, 64)
        W_precision = (1.0 / prior_std**2) + phi_sq.T @ kappa  # (64, 2)
        self.W_std  = 1.0 / np.sqrt(W_precision)

        # diagonal Hessian for b_k: 1/sigma^2 + sum_i kappa_ik
        b_precision = (1.0 / prior_std**2) + kappa.sum(axis=0)  # (2,)
        self.b_std  = 1.0 / np.sqrt(b_precision)

    def predict_probs(self, phi, n_samples=64, rng=None):
        """phi: (N, 64)  →  (N, 2) averaged predictive probabilities."""
        if rng is None:
            rng = np.random.default_rng(0)
        probs = np.zeros((phi.shape[0], self.W_mean.shape[1]))
        for _ in range(n_samples):
            W = rng.normal(self.W_mean, self.W_std)
            b = rng.normal(self.b_mean, self.b_std)
            probs += _softmax(phi @ W + b)
        return probs / n_samples

    def predict_probs_samples(self, phi, n_samples=64, rng=None):
        """Returns the full (n_samples, N, 2) stack — needed for epistemic spread."""
        if rng is None:
            rng = np.random.default_rng(0)
        stack = np.zeros((n_samples, phi.shape[0], self.W_mean.shape[1]))
        for s in range(n_samples):
            W = rng.normal(self.W_mean, self.W_std)
            b = rng.normal(self.b_mean, self.b_std)
            stack[s] = _softmax(phi @ W + b)
        return stack

bll = BayesianLastLayer(net, X, prior_std=10.0)


def _entropy(p):
    p = np.clip(p, 1e-8, None)
    return -(p * np.log(p)).sum(axis=1)


def plot_map_vs_bayes_confidence(model, bll, X, y):
    xx, yy, grid = _make_wide_grid(X)
    phi = get_features(model, grid)

    map_probs = model.forward(grid)
    map_conf  = map_probs.max(axis=1).reshape(xx.shape)
    map_preds = np.argmax(map_probs, axis=1).reshape(xx.shape)

    bay_conf = bll.predict_probs(phi, n_samples=128).max(axis=1).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    for ax, conf, title in zip(axes,
                               [map_conf, bay_conf],
                               ["MAP  CONFIDENCE", "BAYESIAN LAST-LAYER  CONFIDENCE"]):
        cf = ax.contourf(xx, yy, conf, levels=50, cmap="Blues",
                         vmin=0.5, vmax=1.0, alpha=0.95)
        # MAP decision boundary on both panels — same frozen features, boundary unchanged
        ax.contour(xx, yy, map_preds, levels=[0.5], colors="black", linewidths=1.2)
        ax.scatter(X[:, 0], X[:, 1], c=[C0 if yi == 0 else C1 for yi in y],
                   s=8, edgecolors="white", linewidths=0.2, alpha=0.7, zorder=3)
        cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=FG, labelsize=7)
        _clean_ax(ax, title)

    plt.tight_layout()
    plt.show()


def plot_map_vs_bayes_entropy(model, bll, X):
    xx, yy, grid = _make_wide_grid(X)
    phi = get_features(model, grid)

    map_ent = _entropy(model.forward(grid)).reshape(xx.shape)
    bay_ent = _entropy(bll.predict_probs(phi, n_samples=128)).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    for ax, ent, title in zip(axes,
                              [map_ent, bay_ent],
                              ["MAP  PREDICTIVE ENTROPY", "BAYESIAN  PREDICTIVE ENTROPY"]):
        cf = ax.contourf(xx, yy, ent, levels=50, cmap="inferno", alpha=0.9)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=FG, labelsize=7)
        _clean_ax(ax, title)
    plt.tight_layout()
    plt.show()


def plot_map_vs_bayes_radial(model, bll, n_radii=8, r_max=12.0, n_points=400, n_samples=128):
    """Replicates Figure 5 of Kristiadi et al. 2020: confidence vs distance from origin.
    MAP: solid line. Bayesian: mean line + shaded ±3σ band (sigma of class-1 prob across samples)."""
    angles = np.linspace(0, 2 * np.pi, n_radii, endpoint=False)
    radii  = np.linspace(0.0, r_max, n_points)

    map_curves = []
    bay_means  = []
    bay_stds   = []

    for theta in angles:
        pts = np.c_[radii * np.cos(theta), radii * np.sin(theta)]
        map_curves.append(model.forward(pts).max(axis=1))

        phi   = get_features(model, pts)
        stack = bll.predict_probs_samples(phi, n_samples=n_samples)  # (S, N, 2)
        bay_means.append(stack.mean(axis=0).max(axis=1))             # (N,)
        bay_stds.append(stack[:, :, 1].std(axis=0))                  # (N,) — epistemic spread

    map_arr  = np.stack(map_curves)   # (n_radii, N)
    bay_m    = np.stack(bay_means)    # (n_radii, N)
    bay_s    = np.stack(bay_stds)     # (n_radii, N)

    # aggregate across angles: mean and ±3σ envelope
    map_mean = map_arr.mean(axis=0)
    map_std  = map_arr.std(axis=0)
    bay_mean = bay_m.mean(axis=0)
    bay_std  = bay_s.mean(axis=0)     # average epistemic spread across directions

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(BG)

    for ax, mean, std, title in zip(
            axes,
            [map_mean, bay_mean],
            [map_std,  bay_std],
            ["MAP  CONFIDENCE vs RADIUS", "BAYESIAN  CONFIDENCE vs RADIUS"]):
        ax.plot(radii, mean, color=C1, linewidth=1.4)
        ax.fill_between(radii,
                        np.clip(mean - 3 * std, 0, 1),
                        np.clip(mean + 3 * std, 0, 1),
                        color=C1, alpha=0.2)
        ax.axhline(0.5, color=FG, linewidth=0.8, linestyle="--", alpha=0.5)

        band_radii = (0.55, 1.05, 1.55, 2.05)
        for r in band_radii:
            ax.axvline(r, color=BOUNDARY_C, linewidth=0.6, linestyle="--", alpha=0.5)

        ax.set_facecolor(BG)
        ax.set_title(title, color=FG, fontsize=10, fontweight="bold", fontfamily="DejaVu Sans")
        ax.tick_params(colors="#666666", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#222222")
        ax.set_xlabel("radius", color="#666666", fontsize=8)
        ax.set_ylim(0.4, 1.02)

    plt.tight_layout()
    plt.show()



def plot_all(model, X, y, h=0.008):
    from matplotlib.colors import ListedColormap

    xx, yy, grid = _make_grid(X, h)
    out = model.forward(grid)
    preds = np.argmax(out, axis=1).reshape(xx.shape)

    _, r1_ids = _get_regions(model, grid, relu_idx=1)
    _, r2_ids = _get_regions(model, grid, relu_idx=3)
    _, jt_ids = get_joint_regions(model, grid)
    r1 = r1_ids.reshape(xx.shape)
    r2 = r2_ids.reshape(xx.shape)
    jt = jt_ids.reshape(xx.shape)
    n1, n2, nj = len(np.unique(r1_ids)), len(np.unique(r2_ids)), len(np.unique(jt_ids))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor(BG)

    axes[0].contourf(xx, yy, preds, cmap=ListedColormap([C0, C1]), alpha=0.4)
    axes[0].contour(xx, yy, preds, colors=BOUNDARY_C, linewidths=1.5, alpha=1.0)
    axes[0].scatter(X[:, 0], X[:, 1], c=[C0 if yi == 0 else C1 for yi in y],
                    s=10, edgecolors=BG, linewidths=0.2, alpha=0.85)
    _clean_ax(axes[0], "DECISION BOUNDARY")

    _region_mesh(axes[1], xx, yy, r1)
    axes[1].contour(xx, yy, r1, colors=REGION_WIRE, linewidths=0.5, alpha=0.7)
    _clean_ax(axes[1], f"LAYER 1  ({n1})")

    _region_mesh(axes[2], xx, yy, r2)
    axes[2].contour(xx, yy, r2, colors=REGION_WIRE, linewidths=0.5, alpha=0.7)
    _clean_ax(axes[2], f"LAYER 2  ({n2})")

    _region_mesh(axes[3], xx, yy, jt)
    axes[3].contour(xx, yy, jt, colors=REGION_WIRE, linewidths=0.5, alpha=0.7)
    _clean_ax(axes[3], f"JOINT  ({nj})")

    plt.tight_layout()
    plt.show()

plot_training_curves(loss_history, acc_history)
plot_all(net, X, y)
plot_map_vs_bayes_confidence(net, bll, X, y)
plot_map_vs_bayes_entropy(net, bll, X)
plot_map_vs_bayes_radial(net, bll)