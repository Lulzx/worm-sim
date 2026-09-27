# Small-circuit fitting tutorial

[Open the notebook](fit-small-circuit.ipynb) for a complete synthetic fit using
WormSim's native Rust reference. It generates a three-neuron teacher recording,
fits one perturbed chemical strength, checks a separate pulse, and plots both
responses. This is an engineering demonstration, not biological validation.

Requires Git, a Rust toolchain compatible with edition 2024, and
[uv](https://docs.astral.sh/uv/). From the repository root, execute all cells and
save the rendered notebook without changing the JAX fitting environment:

```sh
uv run --no-project --python 3.12 \
  --with nbclient --with nbformat --with ipykernel --with matplotlib==3.11.2 \
  python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

source = Path('tutorials/fit-small-circuit.ipynb')
notebook = nbformat.read(source, as_version=4)
nbformat.validate(notebook)
NotebookClient(
    notebook, timeout=180, kernel_name='python3',
    resources={'metadata': {'path': str(Path.cwd())}},
).execute()
nbformat.write(notebook, 'runs/tutorial-small-executed.ipynb')
PY
```

The cells check loss reduction, independently recompute the exported MSEs, and
verify that frozen parameters remain unchanged. Outputs under
`runs/tutorial-small/` include the full JSON artifact, a source/input hash receipt,
and an SVG plot. The executed notebook remains under `runs/`; the checked-in
notebook has empty outputs so it does not present old results as a new run.

For interactive use, select a Python kernel with Matplotlib in your notebook
editor and run all cells from the repository root or `tutorials/`.
The Rust example also works without Python:

```sh
mkdir -p runs
cargo run --locked --release --example fit_small -- runs/small-fit.json
```

Omit the output argument to keep the original console-only example behavior.
For real-data fitting, use the [JAX backend guide](../backends/jax/README.md).
