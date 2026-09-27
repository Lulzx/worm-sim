# Fixed-model quasi-Newton capacity comparison

The [refined-step Adam fit](CAPACITY-WARM-START.md#refined-step-outcome-completed-capacity-gate-still-failed)
completed but captured only 76.69% of available zero-start training response
energy. The next [declared comparison](../configs/capacity-lbfgs.json) changes
only optimization: SciPy L-BFGS-B, without box bounds, with fresh curvature
history. The original parameter transforms remain in effect.

Start from the same saved-950 parent as the Adam comparison, using ADAL/ADAR's
50 training trials, dt 0.005, 120-second preparation, and the same trace-only
objective and per-neuron gains. The maximum is **201 actual objective/gradient
evaluations**, matching Adam's initial evaluation plus 200 updates; accepted
iterations are capped at 200. This is an evaluation-budget comparison, not an
equal-wall-time claim. The 90% capacity threshold remains unchanged.

The [SciPy optimizer](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
supplies curvature updates and line search. Settings are maxcor 20, maxls 20,
ftol 1e-12 and gtol 1e-9. The small gradient tolerance avoids treating the current
low-gradient underfit as converged at initialization. These settings are a
training-only diagnostic choice, not a validated production optimizer.

`quasi_newton.py` flattens only active coordinates and reconstructs all frozen
values exactly. It receives the existing JAX objective and reverse gradients;
there is no new loss or manually derived gradient. Consecutive identical requests
reuse their evaluation. A separate hard counter enforces the evaluation budget,
even if the optimizer would request more work during line search.

Every trial's metrics and finiteness are logged. Only accepted iterates can
replace `best.json`; `last-accepted.json` is retained separately. A nonfinite
trial stops the diagnostic and retains finite failing parameters. It is not
silently retried with another solver. Exceptions raised inside the objective
propagate and leave the prior accepted artifacts, without a normal result.
Optimizer convergence, budget exhaustion, abnormal termination and the scientific
capacity gate are distinct fields. An abnormal or nonfinite termination cannot
pass the capacity gate. Native checkpoint `fit_config` is retained for model
compatibility; the manifest's `fitting_optimizer` controls this run.

Four adapter tests pass: a known quadratic optimum, frozen coordinates, hard
evaluation limits/nonfinite trials retaining the last accepted state, mask
validation, and the real two-neuron training objective. A real-data one-iteration
smoke used four evaluations and lowered MSE from 0.0438694094 to 0.0438690283.
Its checkpoint independently replays in NumPy within 6.94e-18 MSE. That smoke
is not the declared comparison and does not establish optimizer superiority.

```sh
.venv-jax/bin/python backends/jax/overfit_lbfgs.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json \
  --training runs/overfit-seed1-training.json \
  --warm-start runs/level0-capacity-adal-adar-1000-prep120/epoch-950.json \
  --dt 0.005 --max-evaluations 201 --max-iterations 200 \
  --output runs/capacity-lbfgs-eval201
```

Report the last accepted and best accepted MSE, actual budget used, terminal
reason, complete histories, independent NumPy replay, and step/preparation/drift
controls. No validation/test response values enter this comparison. Dynamical
features remain frozen and a fresh confirmatory holdout remains unsecured.
