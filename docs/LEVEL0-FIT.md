# First Level 0 population fit

**Preprocessing qualification:** the source traces are whole-recording z-scores.
These results concern retrospective processed-signal prediction, not an end-to-end
causal forecast. See the [content-bound audit](PREPROCESSING-AUDIT.md).

This experiment fits shared Level 0 dynamics and affine fluorescence readouts to
the 360 windows from all 15 training animals. The 72 validation windows select an
epoch; test predictions are generated only after selection. It uses the same
content-bound dataset, animal split and common scorer as the controls.

The fixed first-run configuration is
[configs/level0-first-fit.json](../configs/level0-first-fit.json): two epochs,
learning rate 0.005, seed 42, Euler dt=0.01 s, eight history-inference updates per
window and explicit parameter/readout shrinkage. These choices precede this fit's
test evaluation. The test animals have already been inspected in earlier baseline
experiments, so subsequent comparisons are exploratory, not fresh confirmation.

## Algorithm

1. Calibrate the affine readout from training animals only, as documented in
   [INITIAL-STATE.md](INITIAL-STATE.md).
2. Initialize membrane/calcium time constants to 2 s and synaptic time constant to
   0.2 s, with rest and threshold at zero. These are declared initialization
   assumptions, not biological estimates. The other raw values use model defaults.
3. For each training window, infer all 906 initial state values from its 10-second
   history under current parameters. Forecast for 30 seconds without clamping.
4. Differentiate the forecast loss through the exact discrete Euler steps, holding
   the inferred state at the origin fixed. Update shared dynamics and readout
   offsets/log gains with Adam and global gradient-norm clipping at 10.
5. Evaluate each epoch and the initial candidate on validation animals. Select
   maximum arithmetic mean macro-neuron R² at 1, 10 and 30 seconds. Ties retain
   the earlier candidate; if training hurts validation, epoch zero wins visibly.

This is a conditional-gradient training procedure. It does **not** differentiate
through the state-inference optimizer or account for the inferred state's response
to a parameter change. Every next training window re-infers its state under the
updated parameters. Forward and adjoint paths support the unforced Level 0 model;
perturbation protocols are not supported by this fitting objective yet.

All training windows are used in a deterministic SHA-256 order per epoch. Missing
future targets and the sample at the forecast origin are excluded from loss.
History remains available only to the initial-state estimator. Readout offsets
are optimized in training calibration-gain units and gains in log space. The
model's internal calcium scale is frozen at one to avoid adding a second learned
scale on top of the affine readout. The readout has two trained values per
training-observed neuron; training-unseen readouts keep their disclosed defaults.

## Sharing and priors

`parameters::TiedParameters` maps every raw parameter to a named group. An explicit
neuron-to-class map takes precedence over graph class metadata. The first-run
configuration enables exact matching terminal L/R suffix pairs when graph classes
are unannotated. Unmatched neurons stay separate. Chemical strength/sign groups
use ordered class/group pairs; gap groups use unordered pairs. Priors are raw-value
shrinkage around group initialization and Bernoulli sign cross entropy against the
graph's sign probabilities. Readout deviations shrink toward training calibration.

The c302 graph currently has no source-backed class or transmitter annotation.
Suffix pairing is a modeling assumption; it is not complete biological class
assignment. Every graph sign prior is 0.5, explicitly unknown. This experiment
therefore does not establish the requested CeNGEN transmitter/receptor-informed
priors. The sharing/prior mechanism accepts that evidence when supplied, but those
source-backed annotations remain outstanding.

The report counts trained dynamics/readout scalars separately from training-derived
calibration statistics and the 906 state variables inferred per trial. Calibration
statistics set initialization, parameter units and readout priors; final inference
uses the fitted affine readout directly. Parameter matching against other models
must disclose these inferred states and calibration assumptions too.

## Reproduction and evidence

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release
./target/release/wormsim level0-fit data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  configs/level0-first-fit.json runs/level0-first-fit.json
./target/release/wormsim level0-predict data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  runs/level0-first-fit.json test runs/level0-first-predictions.json
./target/release/wormsim bench-score data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  runs/level0-first-predictions.json test runs/level0-first-report.json
```

The fit writes a model snapshot and validation receipt for epoch zero and each
completed epoch. These are candidate artifacts, not optimizer-resume checkpoints.
A failure is reported rather than dropping a window. Snapshot selection is made
from validation scores only.

Tests check every raw dynamics/readout gradient against finite differences,
sharing-gradient accumulation and prior derivatives, use of all training windows,
and an end-to-end invariant: replacing test futures leaves fitted parameters,
validation scores, selected epoch and test predictions unchanged. Passing these
checks does not prove useful forecasting, biological parameter recovery or adequate
Euler accuracy on the real fitted dynamics. The stable latent LDS baseline,
source-backed sign/class annotations and further numerical checks remain required.

## First-run outcome: training failed validation

The [committed-source receipt](level0-first-fit-receipt.json) records all 720
training updates, the three candidate evaluations, selected test scores, animal
bootstrap intervals and a validation-only timestep sensitivity check.

| Candidate | Validation 1 s R² | 10 s R² | 30 s R² |
| --- | ---: | ---: | ---: |
| Initial candidate (selected) | −0.00380 | −0.01107 | −0.01142 |
| Epoch 1 | −0.04811 | −0.03645 | −0.05398 |
| Epoch 2 | −0.00563 | −0.03593 | −0.05271 |

Thus the trained network did not earn selection. The selected initial candidate's
test scores are −0.00333 / −0.01968 / −0.02239. Its marginal animal-bootstrap 95%
intervals are [−0.02130, −0.00333], [−0.03750, −0.01205], and
[−0.03924, −0.01133]. These are not results of a successfully trained dynamics
model. Epoch zero includes a training-calibrated readout and per-window state
inference, but no population gradient updates.

The declared trainable capacity is 7,040 scalars: 6,742 shared dynamics values and
298 readout values. The 298 calibration statistics are separately reported. Since
epoch zero won, the capacity count must not be read as 7,040 successfully estimated
biological parameters. The original run's terminal message called this count
“fitted”; the CLI wording is corrected to “trainable”.

The complete fit, including initial and per-epoch validation, took approximately
328 seconds on the M4 Pro CPU. This measured workload does not justify a backend
rewrite. Halving the selected candidate's inference/forecast step from 0.01 to
0.005 seconds changed validation R² by +0.0000573 / +0.000000932 /
+0.000000000087 at 1/10/30 seconds. This narrow check supports the reported selected
scores' timestep stability; it does not validate all future trained dynamics.

More useful is the validation history diagnostic: macro-neuron reconstruction R²
is **0.64545 at the first observed sample but only 0.000514 at the forecast origin**.
The ten-second prefix is available to inference, yet the fitted initial condition
loses its reconstruction by the time prediction starts. This diagnoses a failure
of this inference/model combination, not proof that the data are unpredictable.
A filtering-based origin-state estimate and a stable latent LDS should be compared
on validation before further attempts at long-horizon biological claims. Source
preprocessing causality/units and biological sign/class annotations also remain
open limitations.

To reproduce the scored receipt from the selected artifact:

```sh
python3 scripts/score_level0_fit.py --receipt runs/level0-first-fit-receipt.json
```

The stable latent LDS comparison is now available in [LATENT-LDS.md](LATENT-LDS.md).
The history-filter comparison is described in [LEVEL0-FILTER.md](LEVEL0-FILTER.md).
Prediction now continues directly from the inferred state at the forecast origin;
the reconstructed prefix is retained separately.
