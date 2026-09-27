# Level 0 history filtering

The first shooting-based fit lost almost all reconstruction by the forecast
origin. This experiment replaces shooting with an approximate extended Kalman
history filter, retaining the same population objective and parameter capacity.
It estimates all 906 voltage/calcium/gate states from the observed ten-second
prefix. Forecasts continue freely from the assimilated origin state.

The coupled nonlinear mean follows the Rust Level 0 dynamics. Uncertainty is
stored as one 3×3 block per neuron. Each Euler prediction retains the diagonal
blocks of the full linearized `F P Fᵀ`, including chemical and gap transmission
of uncertainty, then discards cross-neuron correlations. Calcium measurements
update each block with a Joseph-form covariance update. Missing measurements are
skipped; measurement variance is divided by identity confidence. This is a
block-projected approximation, not a full-network EKF or calibrated posterior.
The general filtering reference is [Särkkä, Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf).
Block projection, bounds and noise choices here are implementation assumptions.

The fixed [configuration](../configs/level0-filter-fit.json) uses initial
variances `[0.25, 0.04, 0.04]`, process variance rates per second
`[0.05, 0.005, 0.005]`, and observation variance `0.0025` in calcium coordinates.
These seven hyperparameters are declared before running this experiment, not
estimated biological noise. Calcium and gate posterior means are projected into
`[1e-8, 1-1e-8]`; covariance does not account for that projection. Diagnostics
record projection counts, prior innovations and posterior reconstruction error.
The filter adds no fitted dynamics/readout scalars. Its per-trial covariance
blocks accompany the 906 inferred point states. Shooting optimizer settings do
not apply when `method` is `block_ekf`.

Training still holds inferred origin states fixed when taking parameter gradients.
It does not differentiate through the filter. Unknown-neuron readouts, suffix
sharing and neutral sign priors have the same limitations as the first fit.
History reconstruction is assimilation evidence, not held-out forecast accuracy.

Tests compare analytic Jacobians to finite differences, projected covariance
against full-network linearized prediction, and scalar filtering against an
independent closed-form Kalman recurrence. End-to-end tests verify origin-state
carry and invariance to changed test futures. These checks do not prove biological
validity or forecasting success.

## Prespecified comparison

Compare epoch zero with archived shooting epoch zero on validation, verifying
identical dynamics/readouts first. Train two epochs on the same 15 animals and
select among epochs 0/1/2 by mean validation R² at 1/10/30 seconds. Score the
selected candidate with the common animal-bootstrap scorer and audit half-step
validation sensitivity. Test animals were inspected previously, so results are
exploratory. Source preprocessing causality remains an open limitation.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release
target/release/wormsim level0-fit data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  configs/level0-filter-fit.json runs/level0-filter-fit.json
python3 scripts/score_level0_fit.py --model runs/level0-filter-fit.json \
  --prefix runs/level0-filter --receipt runs/level0-filter-receipt.json
```

## Outcome: state inference improved, population fit still failed

The [controlled comparison](level0-inference-comparison.json) verifies identical
initial parameters, readouts, calibration and data/split hashes. Both prediction
paths ran from source `c706e62219f756e092c7b1fa9d421a03bec673ca`.

| Epoch-zero method | Origin reconstruction R² | Validation 1 s | 10 s | 30 s |
| --- | ---: | ---: | ---: | ---: |
| Shooting | 0.00051 | −0.00380 | −0.01107 | −0.01142 |
| Block filter | 0.94684 | 0.64628 | −0.00136 | −0.01142 |

Filtering preserves the observed prefix at the forecast origin and improves the
short forecast. It does not preserve useful information at 30 seconds with these
initial dynamics. The prefix reconstruction uses assimilated observations and
must not be presented as prediction accuracy.

The [fit and score receipt](level0-filter-receipt.json) retains all 720 updates:

| Candidate | Validation 1 s | 10 s | 30 s | Selection mean |
| --- | ---: | ---: | ---: | ---: |
| Epoch 0, selected | 0.64628 | −0.00136 | −0.01142 | 0.21117 |
| Epoch 1 | 0.66684 | 0.00107 | −0.04852 | 0.20647 |
| Epoch 2 | 0.67663 | 0.01160 | −0.06750 | 0.20691 |

Both trained epochs improve the short forecast but lose under the declared
three-horizon validation criterion. The selected artifact remains epoch zero;
7,040 is trainable capacity, not a count of successfully fitted dynamics values.
Training calibration contributes 298 separately reported statistics. Inference
continues to estimate 906 state values per trial.

| Selected candidate test horizon | R² | Marginal animal-bootstrap 95% interval |
| --- | ---: | --- |
| 1 s | 0.72163 | [0.65638, 0.73525] |
| 10 s | −0.00776 | [−0.02327, −0.00050] |
| 30 s | −0.02239 | [−0.03924, −0.01133] |

Both long-horizon intervals are below zero. The AR(1) control has higher point
scores at 1 and 10 seconds; the stable LDS has higher point scores at both long
horizons. These are descriptive comparisons, not paired significance tests.
There are only three test animals, and coverage/fallback assumptions differ:
Level 0 uses default readouts for training-unseen identities while those baselines
use persistence. No positive long-horizon or biological-identifiability claim is
supported by this experiment.

Halving dt from 0.01 to 0.005 seconds with weights fixed changes validation R² by
`+0.00011483 / +0.00006201 / +0.00000000712`. Numerical step refinement does not
explain the failure. The fit's candidate timing totals 209.24 seconds on the M4
Pro CPU (initial/per-epoch validation and training; excludes loading, final output
and separate scoring). This workload still does not call for backend expansion.

To reproduce the fixed-parameter comparison:

```sh
python3 scripts/compare_level0_inference.py
```

The next required comparator is the masked GRU, followed by equal access to
behavior channels. A subsequent Level 0 training change must explicitly address
its conditional-gradient approximation, dynamics/readout assumptions or biological
priors and be evaluated on validation before test scoring. Source preprocessing
causality and held-out generalization remain unresolved.
