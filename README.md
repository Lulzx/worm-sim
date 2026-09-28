# WormSim

A differentiable **C. elegans nervous-system simulator** for fitting
connectome-constrained models to neural recordings and testing their responses
to perturbations.

WormSim combines **Rust** for data import, compact storage, reproducible splits,
and independent scoring with **JAX, Diffrax, Equinox, and Optax** for model fitting.
The native Rust simulator also serves as a numerical reference.

This is an active research implementation of [the specification](SPEC.md).
Level 0 fitting works end to end, but a validated whole-worm model and the
specification’s scientific and performance targets remain open.

## Quick start

Requires a Rust toolchain. The repository includes the imported 302-neuron anatomy.

```sh
WORMSIM_COMMIT=$(git rev-parse HEAD) RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --examples --bins
mkdir -p runs
./target/release/wormsim simulate \
  data/c302-herm.wsc examples/aval-pulse.json runs/aval.wst
```

This runs a current-pulse experiment and writes a compressed trajectory plus a
`runs/aval.wst.json` provenance manifest. Use a `.json` output path for readable
voltage and fluorescence arrays. Default parameters are illustrative.

[Protocols](docs/PERTURBATIONS.md) support JSON and YAML, current and conductance
waveforms, voltage clamps, silencing, and ablation. Try
[`examples/aval-clamp.yaml`](examples/aval-clamp.yaml) in place of the pulse protocol.

To check the native implementation and run a small synthetic fitting example:

```sh
cargo test
cargo clippy --all-targets -- -D warnings
./target/release/examples/fit_small
```

For a step-by-step synthetic fit with loss and response plots, open the
[small-circuit tutorial](tutorials/fit-small-circuit.ipynb) and its
[execution instructions](tutorials/README.md).

## Model fitting

Install the pinned Python 3.12 environment with [uv](https://docs.astral.sh/uv/):

```sh
uv venv .venv-jax --python 3.12
uv pip install --python .venv-jax/bin/python -r backends/jax/requirements.txt
.venv-jax/bin/python -m unittest discover -s backends/jax -p 'test_*.py' -v
```

The [JAX backend guide](backends/jax/README.md) covers data preparation, checkpoint
replay, and population fitting. Fits use reverse-mode differentiation and Optax;
checkpoint selection and benchmark evaluation use the Rust scorer. The tested
Apple Silicon environment uses JAX on CPU. GPU performance targets are unverified.

## What is implemented

| Area | Current capabilities |
| --- | --- |
| Anatomy and storage | Canonical 302-neuron identities, sparse chemical and electrical connectivity, source hashes, lossless graph and trajectory codecs |
| Level 0 dynamics | Graded neurons, chemical synapses, gap junctions, calcium readout, tied parameters, and history-based initial-state inference |
| Integration | Native Euler/RK4; JAX reference-grid replay and separately tested Diffrax adaptive and implicit solvers |
| Experimental extensions | Neuromodulation, gap rectification, extra chemical edges, and plasticity; fitting, checkpoint reload, and Rust scoring checked on synthetic data |
| Spontaneous activity | 21 labeled animals, fixed animal splits, history-based forecasts, trivial/AR/LDS/GRU controls, and animal-bootstrap uncertainty |
| Perturbation responses | 3,166 Randi atlas trials, held-out stimulated-neuron splits, trace and pair-response scoring, and fitted Level 0/LDS comparisons |
| Reproducibility | Input hashes, explicit masks, fixed neuron ordering, run manifests, independent scorers, and numerical parity checks |

The [implementation ledger](docs/IMPLEMENTATION-STATUS.md) distinguishes working
features, experimental components, and unmet acceptance gates.

## Results and limitations

The immediate focus is [Level 0 training capacity](docs/LEVEL0-CAPACITY-PROTOCOL.md):
longer optimization, non-neutral initialization, and per-neuron observation gains.
A [1,000-update capacity test](docs/LEVEL0-LONG-RUN.md) improved training fit but
missed its 90% gate and showed late instability. Subsequent
[warm-start diagnostics](docs/CAPACITY-WARM-START.md) reproduced a coarse-step
Euler failure during preparation. A refined-step run completed, but captured
only 76.69% of the available training-response energy. A subsequent
[1,001-evaluation L-BFGS diagnostic](docs/CAPACITY-LBFGS.md) reached 78.02%,
with independent endpoint checks, but exhausted its budget without convergence
and still missed the 90% gate. A bounded
[curvature-history comparison](docs/CAPACITY-CURVATURE-HISTORY.md) raised capture
to 78.26%. A subsequent
[coordinate-scaling comparison](docs/CAPACITY-COORDINATE-SCALING.md) reached 78.63%
versus 78.42% for its control; both exhausted their budgets and missed the gate.
A [240-second preparation continuation](docs/CAPACITY-PREP240.md) reached 78.82%
with stable endpoint preparation gradients, but also exhausted its budget and
failed the gate. The [longer run](docs/CAPACITY-PREP240-LONG.md) reached **79.31%**
after 663 accepted updates, with stable endpoint audits. It stopped on tiny
loss changes while still missing the gradient tolerance and 90% capacity gate.
New dynamical features are paused. Existing validation/test results are
exploratory; a fresh confirmatory cohort remains outstanding.

The JAX backend reproduced the original five-update Level 0 fit through the
unchanged Rust scorer: test pair-response **AUROC 0.681**, with maximum held-out
prediction difference **2.09 × 10⁻¹³**. This establishes numerical reproduction;
it does not establish superiority over the LDS baseline. See the
[migration report](docs/JAX-MIGRATION.md) and
[atlas fitting comparison](docs/LEVEL0-ATLAS-CLASSIFICATION-FIT.md).

Long-horizon forecasting remains unresolved. The spontaneous-activity benchmark
has only three test animals, and its source signals use retrospective
whole-recording preprocessing. See the [forecasting results](docs/TASK2-FITTING.md)
and [preprocessing audit](docs/PREPROCESSING-AUDIT.md) before interpreting scores.

Time is measured in seconds; Level 0 voltages and currents use normalized units,
not calibrated mV/pA. Anatomy alone does not determine chemical signs. Synthetic
checks of solvers, gradients, or extensions are engineering evidence, not
biological validation. Higher-fidelity neurons, source-backed modulation maps,
body and sensory feedback, and full uncertainty and performance acceptance
remain unfinished.

## Documentation

- [Specification](SPEC.md), [implementation status](docs/IMPLEMENTATION-STATUS.md), and [benchmark contracts](docs/BENCHMARKS.md)
- [WormWideWeb data](docs/WORMWIDEWEB.md), [Randi atlas](docs/RANDI-ATLAS.md), and [molecular priors](docs/MOLECULAR-PRIORS.md)
- [Initial-state inference](docs/INITIAL-STATE.md), [JAX fitting](backends/jax/README.md), and [extended model checkpoints](docs/EXTENDED-FITTING.md)
- [Perturbation protocols](docs/PERTURBATIONS.md), [adaptive solvers](docs/ADAPTIVE-SOLVERS.md), [neuromodulation](docs/NEUROMODULATION.md), and [gap rectification](docs/GAP-RECTIFICATION.md)
- [Performance and storage formats](docs/PERFORMANCE.md), [experimental Taichi backend](docs/TAICHI.md), and [upstream review](docs/UPSTREAM-REVIEW.md)

## License

[MIT](LICENSE). Imported sources retain their provenance and upstream notices.
