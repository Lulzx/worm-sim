# Level 0 atlas fitting components

The atlas uses baseline-relative fluorescence. A response readout is now available
in the existing exact discrete-Euler adjoint:

```
prediction_i(t) = offset_i + gain_i * calcium_scale_i * (calcium_i(t) - calcium_i(0))
```

Zero offsets enforce a zero initial response. The initial calcium value is a
model variable, not an observed held-out response. Its derivative includes the
subtracted baseline term. Gain and calcium-scale derivatives likewise use the
calcium difference. Existing absolute-fluorescence forecasting and state inference
keep their previous behavior. Tests compare every raw parameter and initial-state
coordinate, log gains, offsets and current-row derivatives to finite differences.

This is a latent baseline-relative observation model. The gain must absorb the
experimental ΔF/F normalization; there is no claim that model calcium at time
zero equals the experimental ten-second baseline mean. Nor does the current input
constitute a calibrated conversion of optical power to membrane current.

## Exact response aggregation

For a deterministic model with the same initial state, stimulus and readout for
all trials of a stimulated identity, trial-level squared error has sufficient
statistics: the confidence-weighted mean response, accumulated weight, and
within-group residual variance. `bench::atlas_training::aggregate` constructs these
from training trials only. The original MSE equals mean-trace MSE plus the saved
irreducible MSE. Gradients are unchanged. Across target groups, weight by original
`sample_weight` to recover a globally sample-weighted objective.

This aggregation applies to deterministic MSE response fitting only. It must not
be used for trial-conditioned initial states/inputs or Gaussian likelihood fitting.
The LDS training run continues to use its full individual-trial likelihood.
Positive-confidence traces with missing samples are rejected, not discarded;
all accepted real atlas response windows currently contain finite samples.
Zero-confidence traces contribute no weight. Grids must match within each target
and start at stimulation time zero. The aggregated recording's confidence field
encodes relative loss weights, explicitly not calibrated identity confidence.
It is never an evaluation recording or an animal-level observation.

Analytical tests verify original-versus-aggregated MSE and its derivative with
unequal confidence and variable trial responses. Missing-data rejection and exact
training membership are checked. The population runner below uses these
components; its measured comparison against the linear baseline remains pending.

## Native workload measurement

`examples/benchmark_level0_atlas.rs` aggregates the training partition, then
measures one full reverse-mode gradient for the lexically first training target
on all 302 latent cells. It uses the existing forecast defaults, zero readout
offsets and a declared assumed exponentially decaying current. It also compares
Euler step sizes 0.01 and 0.005 seconds on that initial model. Timing boundaries,
all source identities and the original-versus-mean-trace MSE are recorded.
This is a workload check, not a fitted biological result.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example benchmark_level0_atlas
target/release/examples/benchmark_level0_atlas data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json runs/level0-atlas-workload.json
```

The [measured workload receipt](level0-atlas-workload.json), source
`40aed58`, contains 2,842 training trials aggregated into 161 stimulated-target
groups. On the local Apple M4 Pro, one full 302-neuron response gradient took
0.11185 seconds. Halving dt from 0.01 to 0.005 s changed predictions by at most
4.22e−5 on this initial model. This single target does not establish population
throughput or numerical convergence after fitting.

## First population fit protocol

`bench::atlas_level0` now fits tied raw neuron/synapse parameters and a shared
nonnegative current kernel (softplus coordinates). All 302 cells use a shared
fixed initial state generated once from the declared forecast defaults. This
first protocol does not estimate trial-specific pre-stimulation states. The
relative readout has fixed unit gains/zero offsets; the model's positive calcium
scale remains among the tied learned parameters.

Each full-batch update sums exact response adjoints over the training target
aggregates, weighted by original observed sample weight. The retained within-group
variance restores the original training MSE. Raw-parameter shrinkage, graph sign
priors and kernel-coordinate shrinkage are added before Adam (global gradient norm
cap 10, beta1 0.9, beta2 0.999, epsilon 1e−8). Unknown graph signs remain neutral
0.5 priors; suffix L/R sharing is a declared assumption, not an annotation.

`configs/level0-atlas-first-fit.json` fixes five updates at learning rate 0.01,
Euler dt 0.01 s, 39 input lags, and each prior strength 0.01. The current starts
at 0.2 exp(−t/2) on the 0.5 s grid; its final unused row is zero. It represents an
uncalibrated effective drive, not a measured optical waveform. All targets,
including held-out targets, receive the same learned kernel.

Minimum validation pooled MSE selects among initialization and all five updates;
ties retain the earlier checkpoint. No test fluorescence enters fitting or
prediction. A test replaces all test outcomes and verifies identical tied
parameters, shared kernel, initial state, validation selection and predictions.
Every candidate is saved, and the selected model gets a validation-only half-step
check. The shared initial-state restriction and possible mismatch between its
calcium baseline and experimental baseline remain scientific limitations.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example fit_level0_atlas
target/release/examples/fit_level0_atlas data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json configs/level0-atlas-first-fit.json runs/level0-atlas-first-fit
```

This implements the first nonlinear fit runner; measured population results and
comparison against the linear atlas baseline remain pending. The test partition
has already been inspected for the linear baseline, so subsequent results are
exploratory comparisons on that fixed benchmark, not a fresh confirmatory cohort.
