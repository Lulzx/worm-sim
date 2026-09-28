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

## Early comparison prefix verified

Both declared arms launched from clean source `dfc28ca`. The
[prefix receipt](capacity-curvature-prefix.json) verifies initialization plus the
first **20 accepted updates** and all **25 associated objective evaluations**.
All recorded scalar fields agree within absolute tolerance 1e-10, excluding only
elapsed time. Initial MSE reproduces the hashed parent. Manifests agree except
for process ID and the declared `maxcor` value; inputs, source hashes, training
subset, objective, solver and all other optimizer settings match.

The check reuses the tested `complete_rows` and recursive `compare_value`
helpers from `scripts/audit_lbfgs_prefix.py`, then limits the comparison to the
declared early interval. The receipt records snapshot byte lengths and hashes,
so those append-only journal prefixes can be recovered after the runs finish.
This establishes a common scalar trajectory before interpreting later history-
length differences. It does not establish bitwise optimizer-state equality or
optimizer superiority. Both endpoint results were still pending at the snapshot.

## Completed comparison

Both arms terminated at exactly 201 finite objective/gradient evaluations with
`evaluation_budget_exhausted`, not convergence. The last accepted checkpoint is
also the best accepted checkpoint in each arm. Full manifests, histories,
checkpoint hashes and all four endpoint audits are retained in the
[result receipt](capacity-curvature-results.json).

| Retained pairs | Accepted updates | Training MSE | Captured response energy | 90% gate |
| --- | ---: | ---: | ---: | --- |
| 20 | 180 | 0.043706493552 | 78.1239% | Fail |
| 100 | 191 | 0.043693169008 | 78.2582% | Fail |

Larger history lowers MSE by 1.33245e-5 and adds 0.13434 percentage points of
capture in this paired continuation. This is a small training-only improvement,
not evidence of generalization or a result across independent initializations.
Neither arm warrants declaring the small-target capacity gate complete.

Independent NumPy replay matches both endpoint MSEs exactly. Halving the solver
step changes MSE by +5.93e-9 and +2.55e-9 for 20 and 100 pairs respectively;
maximum prediction differences are 0.00232 and 0.00229. Doubling preparation
changes MSE by +8.55e-8 and +4.23e-7, with maximum prediction differences 0.00185
and 0.00423. Quadrupling preparation produces essentially the same losses as
doubling it. The larger-history endpoint therefore has greater residual
preparation sensitivity, although these loss shifts do not reverse the ranking.
No-stimulus prediction energy is 6.60e-9 and 1.35e-7, versus stimulated energy
0.00771 and 0.00778. All 3,638 prepared chemical coefficients remain nonzero.

Active raw-coordinate gradient infinity norms are 0.00324 and 0.00108, both
well above the declared 1e-9 tolerance. They are also larger than the parent's
0.000313 despite lower losses: loss improvement alone does not establish
stationarity. Frozen positive-gain recalibration closes only 2.71% and 2.59%
of the remaining zero-start-bound gap. Gains remain broad, reaching about
1,042 and 1,037 respectively.

Keep new dynamics frozen. More curvature history alone did not resolve the
underfit. The next diagnostic should investigate optimization conditioning or
initialization on training data, with a declared bounded budget, rather than
silently extending this comparison or interpreting it as a capacity limit.
No fresh holdout has been secured and no held-out scoring was performed here.

A [frozen directional-curvature diagnostic](CAPACITY-DIRECTIONAL-CURVATURE.md)
finds strong local sensitivity in threshold and rest coordinates. Its initial
step sizes are too large for derivative agreement, so a finer sweep is needed
before selecting coordinate scaling. No parameters were updated.
