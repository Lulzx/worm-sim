# First Level 0 population fit

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
