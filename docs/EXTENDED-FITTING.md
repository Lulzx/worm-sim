# Fitting and replaying extended JAX models

The coupling modules and optional Diffrax solver settings now have a population
fit and checkpoint path. Rust remains responsible for data validation, split
validation, and trace scoring. JAX evaluates the configured dynamics and gradients.
The integration is checked on a small synthetic dataset; it is not a new biological
fit or a claim that extensions improve existing benchmark results.

## Boundary and checkpoint format

`external_atlas plan` exports only trial IDs, stimulus targets, time grids, neuron
names, response-score names, topology, and lineage metadata for the requested
validation/test partition. It does not export observed fluorescence, confidence
weights, behavior, or response-label values. The Python fitting CLI only requests
validation plans and uses the existing Rust training-statistics export for losses.
It hashes input data files for provenance but does not decode held-out observations.

A saved model is a separate `wormsim-jax-atlas` schema-version-1 envelope containing:

- `base_model`: the native Level 0 parameters, initialization, fit configuration,
  epoch, and data/split lineage.
- `configuration`: versioned extension declarations and optional adaptive solver
  settings (`solver: null` retains reference-grid Euler), plus optional
  [multirate settings](MULTIRATE.md) for a held-concentration coarse grid.
- `extension_parameters`: the fitted arrays, each with explicit shape and flattened
  values. Declarations supply initial values; these arrays supply learned values.

Reload validates parameter families, shapes, finite values, inactive coordinates,
and module/topology compatibility. Native `AtlasModel` deserialization rejects the
envelope and unknown top-level fields; it cannot silently simulate an extended
checkpoint as a plain native model.

All current modules are supported together: modulation, gap rectification, extra
chemical connections, and plasticity. Type parameters unused in a plasticity mode
are frozen, including under AdamW. Extra-edge L1 cost is added once to the complete
training objective, independently of trial count. Other extension parameters are
trainable; biological priors and group-specific learning rates remain future work.

`fit_extensions.py` reuses the Optax fitting loop, saves every candidate, reloads
its complete parameters to generate predictions, and invokes the Rust scorer.
Selection minimizes validation trace MSE, with the earliest epoch breaking ties.
Prediction metadata includes the exact checkpoint file SHA-256. The Rust boundary
checks this binding and declared training/selection lineage, then calls the common
benchmark evaluator. It does not independently reproduce extension dynamics or
prove that a submitter followed its declared training procedure.

## Reproduce the synthetic pipeline

Run from the repository root after installing the [JAX environment](../backends/jax/README.md):

```sh
WORMSIM_COMMIT=$(git rev-parse HEAD) cargo build --release \
  --example external_atlas --example export_external_atlas_fixture \
  --example export_atlas_training
.venv-jax/bin/python -m unittest discover -s backends/jax \
  -p 'test_external_smoke.py' -v
```

The test exports a three-neuron synthetic dataset with held-out stimulus groups,
trains two Optax updates with all four extension modules, Tsit5, and multirate
concentration updates enabled,
selects using Rust scores, reloads the selected model in a separate Python
process, and verifies identical predictions. It checks Rust MSE against direct
NumPy arithmetic, rejects a changed checkpoint paired with old predictions, and
rejects the JAX envelope at the native model-loading boundary. CI builds the Rust
executables and runs this test; standalone Python discovery explicitly skips it
if those executables are absent.

For a persistent run (all output locations must be new):

```sh
mkdir -p runs
./target/release/examples/export_external_atlas_fixture runs/extension-fixture
./target/release/examples/export_atlas_training \
  runs/extension-fixture/graph.wsc runs/extension-fixture/data.json \
  runs/extension-fixture/split.json runs/extension-fixture/model.json \
  runs/extension-training.json
.venv-jax/bin/python backends/jax/fit_extensions.py \
  --model runs/extension-fixture/model.json \
  --configuration backends/jax/examples/extensions-synthetic.json \
  --graph-json runs/extension-fixture/graph.json \
  --graph runs/extension-fixture/graph.wsc --data runs/extension-fixture/data.json \
  --split runs/extension-fixture/split.json --training runs/extension-training.json \
  --scorer target/release/examples/external_atlas --output runs/extension-fit
```

The example configuration names synthetic neurons. It is not a biological map and
must not be applied unchanged to c302. Real-data use requires predeclared source-backed
or explicitly exploratory candidates and type assignments, not test-selected maps.

For inference, `predict_extensions.py --checkpoint ... --graph ... --plan ...
--output ...` uses a Rust-exported plan and the saved envelope. `external_atlas
score GRAPH DATA SPLIT CHECKPOINT validation|test PREDICTIONS NEW_DIRECTORY` applies
the common trace evaluator. The saved predictions also use the existing schema
accepted by `evaluate_atlas_predictions` for pair scores and clustered uncertainty.
Test scoring is a separate post-selection action, not part of the fit runner.

## Remaining requirements

Optimizer resume, nonzero-epoch continuation, per-extension prior families,
source-backed assignments, native perturbation integration, and actual biological
comparisons remain open. Initial-state inference for spontaneous activity and
higher-fidelity neuron models are not introduced by this atlas fitting path.
Adaptive settings are serialized and trainable dynamics differentiate through the
library solver; this does not establish full-network adaptive fitting performance
or scientific acceptance. Existing negative benchmark results remain unchanged.
