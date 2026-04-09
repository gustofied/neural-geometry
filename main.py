"""
neural-geometry — project entry point

Run with:  uv run main.py
"""
import sys
import os

SCRIPTS = {
    "1": ("nn1  — ReLU geometry: radial bands, activation regions, confidence map",
          "exploration/nn1_geometry.py"),
    "2": ("nn2  — paper reproduction: binary LLLA vs MAP (Kristiadi et al. 2020)",
          "exploration/nn2_binary.py"),
}

def main():
    print("\nneural-geometry\n" + "─" * 40)
    for key, (label, _) in SCRIPTS.items():
        print(f"  [{key}]  {label}")
    print("─" * 40)

    choice = input("select: ").strip()

    if choice not in SCRIPTS:
        print(f"unknown option: {choice!r}")
        sys.exit(1)

    _, path = SCRIPTS[choice]
    full_path = os.path.join(os.path.dirname(__file__), path)

    with open(full_path) as f:
        code = f.read()

    # run the selected script with its own __file__ so relative imports work
    exec(compile(code, full_path, "exec"), {"__file__": full_path, "__name__": "__main__"})


if __name__ == "__main__":
    main()
