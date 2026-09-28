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
