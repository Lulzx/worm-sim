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
