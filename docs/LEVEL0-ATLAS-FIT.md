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
training membership are checked. Full nonlinear population fitting, learned
shared stimulation currents, validation selection and comparison against the
linear atlas baseline remain to be implemented and measured.

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
