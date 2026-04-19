<img src="assets/regions_header.png" width="100%">

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
| `speed` | forward pass and linear-region benchmark |
| `relu` | layerwise ReLU regions and decision boundary |
| `bayesian` | MAP vs last-layer Laplace uncertainty  |
| `regions` | OpenGL ReLU region viewer |

---

<kbd>simple</kbd> &nbsp; [neural_geometry/simple.py](neural_geometry/simple.py)

Two-class classifier built from scratch in NumPy. Inspired by [Sylvain Gugger's numpy neural net](https://sgugger.github.io/a-simple-neural-net-in-numpy.html).

<kbd>speed</kbd> &nbsp; [neural_geometry/speed.py](neural_geometry/speed.py)

Two benchmarks: a dense forward pass and a linear-region labeling pass over a 2D grid. NumPy wins on the dense, vectorized case; Numba pulls ahead on the loop-heavy grid pass. I'm using this experiment to get a better feel for where Numba actually helps in practice, especially in the kind of numerical and visualization code that shows up around projects like this. For a concise introduction, see [Python⇒Speed](https://pythonspeed.com/articles/numba-faster-python/).

```
forward pass, 200 samples, 2 → 32 → 32

  python      30.75 ms
  numpy       0.0209 ms    1470x faster than python
  numba       0.0576 ms     534x faster than python    0.4x vs numpy

activation-region map, 600×600, 32 hidden units

  numpy      35.828 ms
  numba       6.963 ms     5.1x vs numpy
```

<kbd>relu</kbd> &nbsp; [neural_geometry/relu.py](neural_geometry/relu.py)

A from-scratch ReLU classifier on a radial-band dataset, with visualizations of layerwise regions, their joint partition, and the final decision boundary. Successive ReLU layers partition the plane into increasingly fine linear regions, and the resulting decision boundary emerges from that composed structure.

<p align="center"><em>Layerwise regions, their joint partition, and the resulting decision boundary</em></p>

<table>
  <tr>
    <td align="center" valign="middle"><img src="assets/relu_layer1.png" width="100%"></td>
    <td align="center" valign="middle"><img src="assets/relu_layer2.png" width="100%"></td>
    <td align="center" valign="middle"><img src="assets/relu_joint.png" width="100%"></td>
    <td align="center" valign="middle"><img src="assets/relu_boundary.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center">Layer 1</td>
    <td align="center">Layer 2</td>
    <td align="center">Joint partition</td>
    <td align="center">Decision boundary</td>
  </tr>
</table>

<p align="center"><em>Radial probes through the trained network</em></p>

<table>
  <tr>
    <td align="center" valign="middle"><img src="assets/relu_logit.png" width="100%"></td>
    <td align="center" valign="middle"><img src="assets/relu_conf_radius.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center">Logit difference vs radius</td>
    <td align="center">Confidence vs radius</td>
  </tr>
</table>

Along a fixed ray, the logit difference changes piecewise linearly, with kinks where the activation pattern changes. Confidence drops near boundary crossings but stays high almost everywhere else, importantly including far from the data.

<kbd>bayesian</kbd> &nbsp; [neural_geometry/bayesian.py](neural_geometry/bayesian.py)

Binary ReLU classifier with a last-layer Laplace approximation, inspired by [Kristiadi et al. 2020](https://arxiv.org/abs/2002.10118). I wanted to understand what changes when only the last layer is treated probabilistically. The point-estimate model stays overconfident far from the training data, while the last-layer Laplace approximation pulls confidence back toward 0.5 in sparse regions.

<p align="center"><em>MAP vs last-layer Laplace confidence</em></p>

<table>
  <tr>
    <td align="center" width="24%"><img src="assets/bayes_map.png" width="100%"></td>
    <td align="center" width="24%"><img src="assets/bayes_llla.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center">MAP</td>
    <td align="center">LLLA</td>
  </tr>
</table>

<p align="center"><em>Confidence along a horizontal slice through the data gap</em></p>

<table>
  <tr>
    <td align="center" width="24%"><img src="assets/bayes_probe_map.png" width="100%"></td>
    <td align="center" width="24%"><img src="assets/bayes_probe_llla.png" width="100%"></td>
  </tr>
  <tr>
    <td align="center">MAP</td>
    <td align="center">LLLA</td>
  </tr>
</table>

MAP quickly returns to near-1 confidence away from the data. The last-layer Laplace approximation relaxes toward 0.5 and shows wider predictive spread in regions with little or no training data.

<kbd>regions</kbd> &nbsp; [neural_geometry/gl_regions.py](neural_geometry/gl_regions.py)

Live OpenGL viewer that trains a ReLU network and renders its joint activation regions in real time. As training progresses, the network reorganizes the plane into linear regions, while the decision boundary settles into a curved shape built from local linear pieces. I leaned a bit stylized here so the partition reads more clearly in motion; the point is to watch the regions reorganize and the decision boundary gradually lock into place.

<img src="assets/regions.gif" width="100%">