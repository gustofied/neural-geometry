import os
import sys
import typer

app = typer.Typer(help="neural-geometry visualizations")

SCRIPTS = {
    "simple":   ("simple neural network",                            "exploration/nn1.py"),
    "speed":    ("python vs numpy vs numba a simple forward pass benchmark","exploration/speed.py"),
    "relu":     ("radial bands and activation regions",              "neural_geometry/nn1_geometry.py"),
    "bayesian": ("MAP vs LLLA confidence maps",                      "neural_geometry/nn2_binary.py"),
    "relu-gl":  ("interactive linear regions",                       "neural_geometry/gl1_geometry.py"),
    "bayes-gl": ("confidence field and posterior boundaries",        "neural_geometry/gl2_binary.py"),
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
