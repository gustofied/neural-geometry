import os
import sys
import typer

app = typer.Typer(help="neural-geometry visualizations")

SCRIPTS = {
    "simple":   ("simple neural network",                            "neural_geometry/simple.py"),
    "speed":    ("forward pass and linear-region benchmark",            "neural_geometry/speed.py"),
    "relu":     ("layerwise ReLU regions and decision boundary",     "neural_geometry/relu.py"),
    "bayesian": ("MAP vs last-layer Laplace uncertainty",             "neural_geometry/bayesian.py"),
    "train":    ("live training partition viewer",                   "neural_geometry/gl_train.py"),
    "relu-gl":  ("interactive linear regions",                       "neural_geometry/gl_relu.py"),
    "bayes-gl": ("confidence field and posterior boundaries",        "neural_geometry/gl_bayesian.py"),
}

def _run(name: str):
    path = os.path.join(os.path.dirname(__file__), SCRIPTS[name][1])
    script_dir = os.path.dirname(path)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    with open(path) as f:
        code = f.read()
    exec(compile(code, path, "exec"), {"__file__": path, "__name__": "__main__"})

for _name, (_label, _) in SCRIPTS.items():
    def _make(name=_name, label=_label):
        @app.command(name=name, help=label)
        def _cmd():
            _run(name)
    _make()

if __name__ == "__main__":
    app()
