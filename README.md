#### Neural Geometry

Building intuition for neural networks through simple models, visualization, and graphics programming. For the full write-up, see [adamsioud.com/projects/neural-geometry](https://www.adamsioud.com/projects/neural-geometry.html). The project uses NumPy, Numba, Matplotlib, and OpenGL, and is also very much a teaching ground for those too.

---

##### Setup

```bash
uv sync
```

##### Run

```bash
uv run neural-geometry [command]
```

| command | description |
|---|---|
| `simple` | simple neural network |
| `speed` | python vs numpy vs numba, forward pass benchmark |
| `relu` | radial bands and activation regions |
| `bayesian` | MAP vs LLLA confidence maps |
| `relu-gl` | interactive linear regions |
| `bayes-gl` | confidence field and posterior boundaries |

---

##### simple

Two-class classifier built from scratch in NumPy, trained on the moons dataset. Inspired by [Sylvain Gugger's numpy neural net](https://sgugger.github.io/a-simple-neural-net-in-numpy.html).

---

##### speed

```
forward pass  —  200 samples, 2→32→32

  python      30.48 ms
  numpy       0.0232 ms    1316x faster than python
  numba       0.0560 ms     544x faster than python    0.4x vs numpy
```

Numba loses to NumPy here because the matmul is small enough that NumPy's BLAS kernel dominates. Numba compiles loops, but BLAS is hand-tuned assembly. At larger matrix sizes the gap closes and can flip.

---

##### relu-gl

Interactive OpenGL viewer for the linear regions a ReLU network creates. Move the mouse to highlight the region under the cursor. The decision boundary glows red. Pan with drag, zoom with scroll.

##### bayes-gl

OpenGL viewer comparing MAP and LLLA confidence side by side. The color encodes class (orange / teal) and saturation encodes conviction. MAP stays vivid everywhere, LLLA fades to grey far from training data. Sampled posterior decision boundaries are drawn as a pink fan: tight near the data, spreading out where the model is uncertain. The divider sweeps automatically.
