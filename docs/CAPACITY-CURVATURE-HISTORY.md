# L-BFGS curvature-history comparison

The [1,001-evaluation fit](CAPACITY-LBFGS.md) reaches 78.02% training-response
capture and remains above the declared gradient tolerance. Its slow progress
motivates a bounded optimizer comparison, not new dynamics or a claim that
model capacity has been exhausted.

The [declaration](../configs/capacity-curvature-history.json) compares SciPy
L-BFGS-B `maxcor=20` with `maxcor=100`. This parameter controls retained curvature
correction pairs. Both arms start from the exact same audited final checkpoint
and reset optimizer history. The control is necessary because a curvature reset
itself changes the trajectory; neither arm continues the previous optimizer
state. The forward model, active parameters, objective, data, solver step and
preparation remain unchanged.

Each arm has 201 actual objective/gradient evaluations and at most 200 accepted
updates. The primary comparison is the last accepted training MSE at termination;
report terminal reasons, best checkpoints, complete histories and numerical
checks for both. The existing 90% gate remains unchanged. Matching early
trajectories and initial MSE are implementation checks, not evidence of scientific
success. More history is a hypothesis to test, not an assumed improvement.

```sh
for wormsim_corrections in 20 100; do
  .venv-jax/bin/python backends/jax/overfit_lbfgs.py \
    --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
    --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
    --warm-start runs/capacity-lbfgs-eval1001/last-accepted.json \
    --dt 0.005 --max-evaluations 201 --max-iterations 200 \
    --maxcor "$wormsim_corrections" \
    --output "runs/capacity-curvature${wormsim_corrections}-eval201"
done
```

The runs may execute concurrently; elapsed times then include resource contention
and are not a controlled speed comparison. Do not increase the budget or add
parameters in response to interim results. If both remain underfit, retain that
outcome and use it to choose the next optimization diagnostic.

The adapter's default remains 20, and manifests/results record the chosen value.
Five adapter tests cover both history settings, analytic quadratic fitting,
frozen coordinates, hard budgets, nonfinite trials, a real synthetic objective
and mask validation. Five warm-start/capacity tests also pass. A separate
one-iteration real-data smoke is an implementation check and is excluded from
the declared comparison.

The [real-data smoke receipt](capacity-curvature-smoke.json) records five
evaluations and one accepted update with `maxcor=100` in both manifest and
optimizer result. Initial MSE exactly reproduces the parent; the final checkpoint
replays exactly in NumPy. Its MSE change is only 4.17e-10 and is not evidence of
optimizer improvement. The smoke ran in the pre-commit worktree; the declared
arms must launch from the clean committed implementation.
