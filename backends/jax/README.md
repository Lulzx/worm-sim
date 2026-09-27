# JAX fitting backend: migration gate

Rust remains the data-import, split and scoring authority. New fitting work moves
to JAX, Diffrax, Equinox and Optax; the Rust numerical core remains a reference.
This backend currently implements frozen Level 0 forward migration and reverse
AD, not a replacement population fit runner or a new biological result.

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
floating-point reference step accumulation. Switching to adaptive or implicit
integration will be a separate numerical comparison.

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

The training objective, frozen-coordinate masks and Rust training-statistics
export are now implemented (see below). Remaining migration gates: preparation
reuse across targets, validation-only
selection, checkpoint round trips, optimizer trajectory comparison and a full
reproduction of fitting. Matching saved predictions alone does not reproduce
training, fix underfitting or establish superiority over the LDS baseline.

Planned library reuse: Diffrax adaptive/stiff integration; Jaxley for suitable
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
  runs/level0-atlas-classification-fit/epoch-0.json runs/jax-training.json runs/randi-pairs.json
.venv-jax/bin/python backends/jax/check_training.py \
  --model runs/level0-atlas-classification-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/jax-training.json \
  --reference-next runs/level0-atlas-classification-fit/epoch-1.json \
  --output runs/jax-first-update.json
```

The epoch-one comparison starts from zero optimizer moments and checks the
existing Adam settings (global gradient clip 10, beta1 .9, beta2 .999, epsilon
1e-8). It is not an optimizer resume implementation. Preparation is currently
repeated per target; sharing that work and a full population fit runner remain.
