# Coordinate-scaling comparison

The [fine directional probes](CAPACITY-DIRECTIONAL-CURVATURE.md) support derivative
agreement but show substantially different curvature in three gradient-aligned
parameter families. This motivates the [declared experiment](../configs/capacity-coordinate-scaling.json):
a fixed-budget comparison of identity coordinates with threshold scale 0.13 and
rest scale 0.16, keeping other scales at one. These heuristic factors are rounded
square roots of the local chemical-strength-to-family curvature ratios. They
are not a diagonal Hessian estimate or a guarantee of faster convergence.

Both arms use `theta = initial + scale * z`, start at `z=0`, and reset L-BFGS
history. Only active coordinates participate. The adapter returns raw model
parameters and multiplies raw gradients by scale for the optimizer chain rule.
Frozen coordinates remain exactly unchanged. The objective, dynamics, data,
solver, preparation and observation model remain identical.

Each arm has 201 actual evaluations, at most 200 accepted updates and 100
curvature pairs. The primary outcome is last accepted training MSE, with full
histories and terminal reasons. Both use the existing 90% capacity gate. Gradient
tolerance applies in optimizer coordinates and is therefore scale-dependent;
report raw and transformed endpoint norms separately. Neither budget exhaustion
nor optimizer termination alone proves adequate fitting.

```sh
for wormsim_scaling in identity curvature-v1; do
  .venv-jax/bin/python backends/jax/overfit_lbfgs.py \
    --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
    --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
    --warm-start runs/capacity-curvature100-eval201/last-accepted.json \
    --dt .005 --maxcor 100 --coordinate-scaling "$wormsim_scaling" \
    --max-evaluations 201 --max-iterations 200 \
    --output "runs/capacity-scaling-${wormsim_scaling}-eval201"
done
```

Require both initial losses to reproduce the same parent. Unlike the previous
history-length comparison, subsequent trajectories are expected to differ
immediately. Independently replay both endpoints and perform step/preparation,
residual calibration and stationarity checks after termination. No test data are
used. Concurrent elapsed times are not controlled performance measurements.

## Implementation checks

Seven optimizer tests pass, including a finite-difference check of the scaled
chain rule, anisotropic quadratic convergence, exact preservation of frozen
coordinates, invalid scales, hard evaluation limits and a synthetic neural
objective. Two stationarity-summary tests also pass. A
[real-data smoke](capacity-scaling-smoke.json) took four evaluations and one
accepted update, terminating at its declared iteration limit. Its initial MSE
exactly reproduced the parent; final MSE was 0.04369316702558147. Independent
NumPy replay agreed within floating-point precision. The smoke ran in the dirty
pre-commit worktree and is excluded from the declared comparison. Endpoint
stationarity output includes both raw and optimizer-coordinate gradients.

## Completed result

Both runs launched from clean commit `708f653`, reproduced the same parent loss,
and completed exactly 201 finite objective/gradient evaluations. Both terminated
with `evaluation_budget_exhausted`, not convergence. Their final checkpoints are
also their best accepted checkpoints. The
[full receipt](capacity-scaling-results.json) retains manifests, complete accepted
and trial histories, checkpoint hashes, and independent endpoint audits.

| Coordinates | Accepted updates | Final training MSE | Captured response energy | 90% gate |
| --- | ---: | ---: | ---: | --- |
| Identity | 185 | 0.043677208568 | 78.4191% | Fail |
| Threshold/rest scaled | 190 | 0.043656337684 | 78.6295% | Fail |

Scaling lowers MSE by 2.08709e-5 and adds 0.21043 percentage points of capture.
This is a small finite-budget improvement on a single paired continuation;
it does not establish generalization, convergence, or an adequate training fit.
The comparison does not independently replicate initialization or estimate
uncertainty across targets. CI for the implementation passed.

Independent NumPy replay reproduces both final losses. Halving the step changes
MSE by +6.94e-9 and +4.21e-8 for identity and scaled coordinates respectively,
with maximum prediction differences 0.00280 and 0.00291. Doubling preparation
changes MSE by +1.63e-6 and +1.70e-6, with maximum prediction differences 0.01321
and 0.01231. Quadrupling preparation leaves those longer-preparation losses
essentially unchanged. The ranking survives these controls, but the endpoints
are more sensitive to finite preparation than earlier fits; do not describe them
as exactly equilibrated. No-stimulus energy is 1.32e-6 and 1.11e-6 versus stimulated
energy 0.00777 and 0.00780. All 3,638 prepared chemical coefficients remain nonzero.

Raw gradient infinity norms are 0.000268 and 0.002215. In optimizer coordinates
they are 0.000268 and 0.000288, both above the 1e-9 tolerance. Lower loss did not
establish stationarity. Frozen positive-gain recalibration closes only 2.55% and
2.41% of the remaining zero-start-bound gap. Gains remain broad and reach about
1,042 and 1,040.

Retain scaling as an available optimizer option, without declaring it sufficient
to solve the underfit. New dynamical features remain frozen. Before another long
fit, account for finite-preparation sensitivity and choose a bounded optimization
or initialization diagnostic. This comparison is complete; the 90% capacity gate,
broader training replication, fresh holdout and scientific acceptance remain open.

A subsequent [preparation-gradient audit](CAPACITY-PREPARATION-GRADIENTS.md)
finds that doubling preparation reverses much of the raw gradient direction,
despite the small loss shift. Gradients stabilize from 240 to 480 seconds at
this endpoint. Address this before another continuation of the fit.
