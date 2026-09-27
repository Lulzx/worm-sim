# JAX fitting backend: migration gate

Rust remains the data-import, split and scoring authority. New fitting work moves
to JAX, Diffrax, Equinox and Optax; the Rust numerical core remains a reference.
This backend implements Level 0 dynamics, reverse AD and a population fit runner
with independent Rust validation scoring. The original five-update fit has been
reproduced; see [migration results](../../docs/JAX-MIGRATION.md). This is numerical
reproduction, not a new biological result.

```sh
uv venv .venv-jax --python 3.12
uv pip install --python .venv-jax/bin/python -r backends/jax/requirements.txt
.venv-jax/bin/python -m unittest discover -s backends/jax -p 'test_*.py' -v
.venv-jax/bin/python backends/jax/replay.py \
  --model runs/level0-atlas-classification-fit/selected.json \
  --graph runs/c302-audit.json --data runs/randi-data.json \
  --reference runs/level0-atlas-classification-fit/test-predictions.json \
  --output runs/jax-level0-replay
# Use the existing Rust scorer without changes:
target/release/examples/evaluate_atlas_predictions \
  data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json \
  runs/randi-pairs.json runs/jax-level0-replay/predictions.json test \
  runs/jax-level0-replay-evaluation
```

The dependency file pins the complete tested Python 3.12 environment. Computation
uses float64. Apple Silicon currently uses the standard JAX CPU backend; GPU
speed is not assumed. The compatibility path deliberately retains the original
Euler grid, positive transforms, tied parameters, unforced preparation, shared
input kernel and relative calcium readout. Diffrax `StepTo` follows the actual
floating-point reference step accumulation. Adaptive and implicit integration are available through the separate
[extended fitting path](../../docs/EXTENDED-FITTING.md), with explicit solver settings.

Equinox carries the sparse topology. JAX scatters implement chemical and gap
currents. Diffrax supplies the integrator and recursive checkpointed adjoint;
JAX differentiates composed losses, including through preparation. The tests
compare with independent NumPy dynamics, check batched targets, compare reverse
gradients with finite differences across all parameter families, and apply an
Optax Adam step. No new hand-written differentiation or optimization algorithm
is used. Batched execution is supported by `jax.vmap`; large-batch timing and
memory use have not been measured.

The replay CLI recomputes every output value from the frozen checkpoint. The
reference artifact supplies trial/output schema and a post-hoc numerical check;
its predicted values are never inputs to the dynamics. It records input and
output hashes, package/device versions, source revision, dirty-worktree state and
per-target elapsed time (including compilation for the first target). Raw data
and large generated artifacts remain under ignored `runs/`.

The training objective, frozen masks, training-statistics export, Rust checkpoint
validation/selection and original five-update trajectory reproduction are complete.
Preparation reuse, optimizer resume and additional configuration parity checks
remain. Reproducing this fit does not fix underfitting or establish superiority
over the LDS baseline.

Further planned library reuse: Jaxley for suitable
conductance-based cells after equation/unit audits; dynamax for LDS inference;
NumPyro or sbi for the relevant inference tasks after identifying the actual
posterior/likelihood requirements. Library availability is not implementation or
acceptance evidence. Rust scorers, fixed splits and provenance contracts remain
unchanged.

Primary API references: [Diffrax adjoints](https://docs.kidger.site/diffrax/api/adjoints/),
[Equinox transformations](https://docs.kidger.site/equinox/api/transformations/),
[JAX installation and device support](https://docs.jax.dev/en/latest/installation.html).

## Training objective migration

`examples/export_atlas_training.rs` validates the checkpoint, Rust split and pair
evidence, then exports only training sufficient statistics, ordered pair labels
and sign probabilities. It refuses to overwrite an existing export. The export
is bound to the exact checkpoint SHA-256. `objective.py` composes MSE, optional
pair BCE/correlation and priors in JAX. Target contributions retain the original
confidence/sample weighting; pair losses normalize over all eligible training
pairs; priors are added once per full batch. Frozen coordinates use stop-gradient.
`check_training.py` streams reverse gradients across targets without retaining
all trajectories, and can compare one Optax Adam update against frozen Rust
checkpoint epoch one. This check does not select a new model or read test scores.

```sh
WORMSIM_COMMIT=$(git rev-parse HEAD) cargo run --release --example export_atlas_training -- \
  data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json \
  runs/level0-atlas-classification-fit/epoch-0.json runs/jax-training-v2.json runs/randi-pairs.json
.venv-jax/bin/python backends/jax/check_training.py \
  --model runs/level0-atlas-classification-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/jax-training-v2.json \
  --reference-next runs/level0-atlas-classification-fit/epoch-1.json \
  --output runs/jax-first-update.json
```

The epoch-one comparison starts from zero optimizer moments and checks the
existing Adam settings (global gradient clip 10, beta1 .9, beta2 .999, epsilon
1e-8). It is not an optimizer resume implementation. Preparation is currently
repeated per target; sharing that work remains. The population fit runner is described below.

## Population fit runner

`fit.py` starts from an exported epoch-zero checkpoint, retains Optax moments
across updates, writes every candidate in the Rust `AtlasModel` schema, and calls
`score_atlas_checkpoint` for validation-only scoring. Selection minimizes Rust
validation MSE with earliest-epoch tie breaking. The Python runner never reads
held-out fluorescence or test scores. Source and input hashes (including the
Rust scorer binary) are stored in a manifest. Output directories must be new;
nonzero-epoch starts are rejected because optimizer resume is not yet supported.
Adam/AdamW and constant/cosine schedules use Optax. Frozen coordinates receive no
updates or decoupled decay. The original initialization, priors and split lineage
remain in each checkpoint.

```sh
WORMSIM_COMMIT=$(git rev-parse HEAD) cargo build --release --example score_atlas_checkpoint
.venv-jax/bin/python backends/jax/fit.py \
  --model runs/level0-atlas-classification-fit/epoch-0.json \
  --graph-json runs/c302-audit.json --graph data/c302-herm.wsc \
  --data runs/randi-data.json --split data/randi-neuron-split.json \
  --training runs/jax-training-v2.json \
  --scorer target/release/examples/score_atlas_checkpoint \
  --output runs/jax-level0-classification-fit
```

This runner tests migration under the frozen configuration; it does not tune on
the repeatedly inspected test cohort. Tests cover multiple optimizer updates,
frozen coordinates under AdamW, earliest-epoch selection ties and cosine rates.
The original five-update fit and final Rust-scored test AUROC have been reproduced
([receipt](../../docs/jax-trained-test-replay.json)). Other configurations still
require their own numerical checks.

Training export schema 2 also carries native chemical/gap topology; the JAX
loader rejects a graph with different endpoints or weights, even if names match.

## Adaptive solver API

The optional `Adaptive` settings enable Diffrax Tsit5 or implicit Kvaerno5 with
PID control and reverse differentiation through preparation and all response
intervals. The default Euler reproduction path and population runner remain
unchanged. See [adaptive solver semantics and checks](../../docs/ADAPTIVE-SOLVERS.md),
including why implicit stages require interval-local input values.

## Slow modulation API

`Modulation` supplies sparse release/receptor maps and extra concentration states
for the Level 0 engine. It supports named parameter tying, independent
species/compartment channels, and gain/leak/chemical-weight effects with automatic
gradients. Maps require explicit provenance; the included example is synthetic.
See [equations, API and acceptance limits](../../docs/NEUROMODULATION.md).

## Rectifying gaps

`GapRectification` adds a conservative, nonnegative voltage-dependent conductance
on explicitly selected anatomical pairs. Zero asymmetry preserves the baseline;
source provenance and canonical pair orientation are required. See the
[law, API and limitations](../../docs/GAP-RECTIFICATION.md).

## Optional off-connectome connections

`DarkEdges` declares a budgeted sparse set of extra chemical candidates with
physical-strength L1 regularization. The low-level Level 0 API supports their
dynamics and reverse gradients. See [configuration and limits](../../docs/DARK-EDGES.md);
the [extended fitter](../../docs/EXTENDED-FITTING.md) preserves these parameters
and submits predictions to the Rust scorer.

## Short-term plasticity

The optional `Plasticity` module adds type-tied depression and facilitation to
anatomical or declared extra chemical edges, sharing states by source/type.
[Equations, configuration, checks, and integration limits](../../docs/PLASTICITY.md)
distinguish this rate-adapted mechanism from calibrated worm biology.

## Extended population fitting

`fit_extensions.py` trains declared coupling modules and saves complete JAX
checkpoint envelopes. `predict_extensions.py` reloads them against observation-free
Rust plans; Rust scores each validation candidate. See the
[reproduction guide and boundaries](../../docs/EXTENDED-FITTING.md).

## Multirate modulation

Optional [held-concentration splitting](../../docs/MULTIRATE.md) updates modulation
on a declared coarse grid while Diffrax integrates fast states. Its settings
survive fitting/checkpoint reload; tests check convergence and gradients.

## Adjoint configuration

Adaptive settings support an explicit recursive checkpoint budget or optional
continuous adjoints with separate backward tolerances. See
[configuration, numerical checks, and limits](../../docs/ADJOINTS.md).

## Group learning rates

The extended configuration supports [learning-rate multipliers](../../docs/GROUP-LEARNING-RATES.md)
for native parameter types, exact tied groups, and module arrays. Zero freezes a
coordinate through gradient clipping and AdamW decay; checkpoint reload preserves
these settings.
