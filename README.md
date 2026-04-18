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

<kbd>simple</kbd> &nbsp; [neural_geometry/simple.py](neural_geometry/simple.py)

Two-class classifier built from scratch in NumPy. Inspired by [Sylvain Gugger's numpy neural net](https://sgugger.github.io/a-simple-neural-net-in-numpy.html).

<kbd>speed</kbd> &nbsp; [neural_geometry/speed.py](neural_geometry/speed.py)

Two benchmarks: a dense forward pass and a per-pixel ReLU region computation. The forward pass maps cleanly to matrix multiplies, so NumPy wins via BLAS. The region grid is loop-heavy per-pixel work, and Numba pulls ahead.

```
forward pass, 200 samples, 2 > 32 > 32

  python      30.91 ms
  numpy       0.0231 ms    1339x faster than python
  numba       0.0578 ms     535x faster than python    0.4x vs numpy

region grid, 600x600, 32 hidden units

  numpy      38.256 ms
  numba       6.952 ms     5.5x vs numpy
```

<kbd>relu-gl</kbd> &nbsp; [neural_geometry/gl1_geometry.py](neural_geometry/gl1_geometry.py)

Interactive OpenGL viewer for the linear regions a ReLU network creates. Move the mouse to highlight the region under the cursor. The decision boundary glows red. Pan with drag, zoom with scroll.

<kbd>bayes-gl</kbd> &nbsp; [neural_geometry/gl2_binary.py](neural_geometry/gl2_binary.py)

OpenGL viewer comparing MAP and LLLA confidence side by side. The color encodes class (orange / teal) and saturation encodes conviction. MAP stays vivid everywhere, LLLA fades to grey far from training data. Sampled posterior decision boundaries are drawn as a pink fan: tight near the data, spreading out where the model is uncertain. The divider sweeps automatically.
