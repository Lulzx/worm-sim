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
components; its first measured comparison is reported below.

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

The first measured population result and linear-baseline comparison are below.
The test partition
has already been inspected for the linear baseline, so subsequent results are
exploratory comparisons on that fixed benchmark, not a fresh confirmatory cohort.

## First completed fit: negative comparison

Source `2a510efbc972e0b60732f6291ced59bed69d5d16` completed the five-update
protocol. Validation MSE selected **epoch 1**, with 6,781 nominal fitted parameters.
Epochs 2–5 did not improve validation MSE. The selected model uses the same fixed
initial state as initialization. Summed epoch timers include gradients and
validation, but exclude data loading, aggregation and checkpoint writing.

| Test metric | Level 0 | Shared-kernel LDS |
|---|---:|---:|
| Pooled MSE | 0.0489985 | 0.0473448 |
| Mean trace correlation | 0.0239803 | 0.0452857 |
| Published pair-detection AUROC | 0.4177484 | 0.6861866 |

Both models have all 12,588 test trace correlations defined. The zero-response
MSE is 0.0500357; modest improvement over that control does not meet the required
linear-baseline comparison. Pair classification uses the same fixed absolute-area
score, eligibility and 1,758 pairs for both models.

The [paired comparison](atlas-first-comparison.json) reports Level 0 minus LDS:

- MSE difference +0.001654, target-cluster interval [+0.000413, +0.002655];
  recording-cluster interval [+0.001161, +0.002227]. Both favor the LDS.
- Correlation difference −0.02131, target interval [−0.03875, +0.00300];
  recording interval [−0.03495, −0.00985]. The target interval includes zero.
- AUROC difference −0.26844, target interval [−0.32737, −0.20375], favoring LDS.

These are paired, conditional, separate marginal cluster analyses with the
previously stated crossed-dependence and previously inspected-cohort limitations.
Level 0's standalone AUROC interval is [0.34658, 0.48453]. We do not invert the
ranking using this test outcome or call below-chance ranking a success.

Halving the selected model's integration dt changed validation MSE by about
1.2e−9 and predictions by at most 5.68e−5. This check does not establish full
convergence, but the observed baseline gap is not explained by this dt comparison.

The [saved-output audit](level0-atlas-fit-audit.json) independently recomputes trace
MSE/correlation, response-area scores and direct pairwise AUROC, checks selected
checkpoint identity/declared lineage, and hashes artifacts. It does not independently
replay nonlinear dynamics or optimization. The [uncertainty receipt](level0-atlas-fit-uncertainty.json)
preserves model/evaluator sources and coverage. The shared evaluator reproduces
the earlier LDS pair predictions, pair report and both bootstrap results exactly.

```sh
python3 scripts/audit_level0_atlas.py --output runs/level0-atlas-fit-audit.json
```

The first Level 0 atlas fit therefore fails the required baseline comparison.
Further work should diagnose validation-only response dynamics, particularly the
fixed common initial state and stimulus-independent drift, before expanding compute
or tuning against test outcomes. Source-backed signs/classes, calibrated drive,
and a declared training-pair classification objective are still missing. Task 1
biological acceptance and the complete specification remain open.

## Validation drift diagnostic

`diagnose_atlas_drift` keeps a checkpoint frozen and evaluates three validation-only
signals: its original driven response, the same initial state and parameters with
zero input current, and their difference. The three use identical observations,
trace scoring and pair ranking. It reports the initial unforced state derivative
and weighted mean squares, including the cross term; these components are not
orthogonal variance fractions. Subtracting drift here is a diagnostic, not a
newly fitted model or evidence of held-out improvement.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example diagnose_atlas_drift
target/release/examples/diagnose_atlas_drift data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json runs/randi-pairs.json runs/level0-atlas-first-fit/selected.json runs/level0-atlas-selected-drift
target/release/examples/diagnose_atlas_drift data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json runs/randi-pairs.json runs/level0-atlas-first-fit/epoch-0.json runs/level0-atlas-initial-drift
```
