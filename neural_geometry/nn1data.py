import numpy as np

def make_radial_bands(
    n_samples=1600,
    band_radii=(0.55, 1.05, 1.55, 2.05),
    band_width=0.12,
    xy_noise=0.02,
    seed=None,
):
    rng = np.random.default_rng(seed)

    n_bands = len(band_radii)
    counts = np.full(n_bands, n_samples // n_bands, dtype=int)
    counts[: n_samples % n_bands] += 1

    X_parts = []
    y_parts = []

    for i, (r, n_i) in enumerate(zip(band_radii, counts)):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_i)

        # Sample radius from a noisy band rather than a perfect circle
        rr = r + rng.normal(loc=0.0, scale=band_width, size=n_i)

        x1 = rr * np.cos(theta)
        x2 = rr * np.sin(theta)

        X_band = np.c_[x1, x2]

        if xy_noise > 0:
            X_band += rng.normal(loc=0.0, scale=xy_noise, size=X_band.shape)

        # Alternate labels by band: 0,1,0,1,...
        y_band = np.full(n_i, i % 2, dtype=int)

        X_parts.append(X_band)
        y_parts.append(y_band)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle so batches are mixed
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # One-hot targets for your current training code
    Y = np.zeros((len(y), 2), dtype=float)
    Y[np.arange(len(y)), y] = 1.0

    return X, y, Y