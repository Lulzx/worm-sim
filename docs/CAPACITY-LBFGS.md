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


## Outcome: finite budget exhaustion, small improvement

At clean source `4031794`, the run used exactly 201 objective/gradient evaluations,
accepted 181 steps and exited normally at the adapter's hard evaluation limit.
Every trial was finite. The optimizer **did not converge**; budget exhaustion
is not reported as optimizer success. The last accepted iterate is also the best.
The [receipt](capacity-lbfgs-results.json) retains both complete histories,
checkpoint/input hashes, terminal status and independent audits.

| Optimizer | Evaluations | Accepted updates | Final MSE | Captured response energy |
| --- | ---: | ---: | ---: | ---: |
| Adam | 201 | 200 | 0.04384851 | 76.6920% |
| L-BFGS | 201 | 181 | 0.04380470 | 77.1337% |

L-BFGS improves MSE by 4.38e-5 and capture by **0.442 percentage points**. Both
miss the unchanged 90% gate. This is one training-only comparison from one parent,
not a held-out benefit or proof of optimizer superiority across initializations.
The final accepted step was recorded after 1,196.48 seconds; this elapsed time
includes compilation and is not a controlled machine performance benchmark.

Independent NumPy replay matches MSE exactly. Halving the step changes MSE by
2.22e-8 (maximum prediction change 0.00129); doubling preparation changes MSE
by 4.20e-9. Quadrupling preparation adds negligible change. Unstimulated prediction
energy is 2.20e-12 versus 0.007808 with stimulation. All 3,638 prepared chemical
coupling coefficients remain nonzero. Frozen positive per-neuron recalibration
would give 0.04373860, closing only 2.91% of the remaining gap. Learned gains
span 0.0286–1,050.76 and have no physiological interpretation.

The model remains underfit and this optimizer is not converged. The next bounded
diagnostic increases the evaluation budget while holding all other settings
fixed. It must preserve the distinction between reproducing the old trajectory,
continuing optimization with reconstructed curvature history, and an independent
restart. No new dynamics or held-out model selection is justified by this result.

The [longer declaration](../configs/capacity-lbfgs-long.json) fixes 1,001 total
evaluations and at most 1,000 accepted updates. Start from the same original
saved-950 parent, rebuilding curvature history by repeating the earlier
trajectory. This repeats work and is not an optimizer-state resume. Verify the
first 201 trial metrics and 182 accepted iterates against the hashed reference
before interpreting subsequent progress; report any mismatch rather than
silently treating it as a continuation. Use the command above with
`--max-evaluations 1001 --max-iterations 1000` and output
`runs/capacity-lbfgs-eval1001`. No model or objective settings change.

Verify the repeated prefix with:

```sh
python3 scripts/audit_lbfgs_prefix.py \
  --reference runs/capacity-lbfgs-eval201 \
  --candidate runs/capacity-lbfgs-eval1001 \
  --declaration configs/capacity-lbfgs-long.json \
  --output runs/lbfgs-prefix-complete.json
```

The verifier checks the declaration's reference hashes and requires identical
input, objective, backend-source and optimizer settings apart from the declared
budgets. It compares all trial metrics and accepted-iterate scalar records,
including evaluation/iteration indices, within absolute tolerance 1e-10. Only
elapsed time is excluded. A live partial final line is ignored; missing complete
rows reject full verification. `--allow-partial` emits an explicitly incomplete
progress receipt and must not be treated as verification of the full prefix.
This check does not establish bitwise equality of parameters or curvature state.

The [complete prefix receipt](capacity-lbfgs-prefix.json) now verifies all 201
reference trial evaluations and all 182 accepted records (including initialization)
against the longer run launched at clean source `1b984d3`. Input, configuration,
backend-source and optimizer checks pass; scalar comparisons exclude only elapsed
time and use absolute tolerance 1e-10. At the receipt snapshot, the candidate had
203 evaluations and 184 accepted records. This establishes reproduction of the
recorded scalar trajectory before the extension, not bitwise optimizer-state
equality. The longer run was still live; its terminal capacity outcome remains
pending.

## Prepared-activation screen

While the longer fit runs, a [frozen-checkpoint screen](capacity-prepared-activation-screen.json)
checks the independently audited 201-evaluation endpoint for a return to a broadly
dead activation state. Using `Replay`'s prepared voltage, threshold and slope,
compute `q = sigmoid((v - threshold) * slope)` and `dq/dv = q*(1-q)*slope`.
The descriptive saturation cutoff is `q < 1e-6` or `q > 1-1e-6`; it is not a
biological threshold.

Six of 302 neurons meet this cutoff, all observed in the training subset:
AUAL, AUAR, RMFL, RMFR, SMBDR and URADR. Among the 201 observed neurons, median
release is 0.0547 and median voltage derivative is 0.2639 in normalized units.
The six neurons contribute 3.63% of the summed weighted mean-trace error in the
existing residual diagnostic. That error includes the first-sample mismatch;
its denominator differs from the zero-start capacity-gap denominator.

This screen does not support treating the whole prepared network as switched
off. It does not inspect activation during stimulation, all fitted-parameter
gradients, or downstream effects mediated by hidden neurons, and therefore does
not establish that saturation is irrelevant to optimization. No fit settings
or capacity criteria were changed. The receipt pins the checkpoint, input,
residual-report and independent replay source hashes and records the formula.

## Frozen-checkpoint gradient diagnostic

The [201-evaluation gradient receipt](capacity-lbfgs-201-stationarity.json)
re-evaluates the accepted endpoint using the unchanged fitting source and inputs.
It reproduces MSE exactly and finds **7,083 active coordinates**, gradient L2
norm **0.00120705**, and maximum absolute active gradient **0.000766651**.
The latter exceeds the declared `gtol=1e-9`; this checkpoint does not satisfy
that gradient criterion. This reinforces the reported budget exhaustion,
not a claim that the model has reached its capacity limit.

`backends/jax/audit_capacity_stationarity.py` reports norms by native parameter
family, stimulus kernel and observation gains. Frozen coordinates are excluded
from these norms. Values are in the optimizer's raw coordinates: differences
between families are not condition numbers or physiological importance scores.
This is an evaluation of the existing AD gradient, not an independent finite-
difference gradient check, a refit, or an optimizer termination decision.

The auditor checks training-subset and input hashes, all backend source hashes
recorded by the original manifest, JAX version, and checkpoint MSE. Two focused
tests verify grouping/masking against known norms and reject malformed masks,
misaligned names and nonfinite gradients. The real-data frozen evaluation also
passes. Apply the same diagnostic to the longer run's terminal accepted
checkpoint, after it finishes:

```sh
.venv-jax/bin/python backends/jax/audit_capacity_stationarity.py \
  --checkpoint runs/capacity-lbfgs-eval1001/last-accepted.json \
  --manifest runs/capacity-lbfgs-eval1001/manifest.json \
  --training runs/overfit-seed1-training.json --graph runs/c302-audit.json \
  --output runs/capacity-lbfgs-1001-stationarity.json
```

Even meeting a local gradient tolerance would not establish global optimality,
biological adequacy, or the unchanged 90% capacity gate. The longer run remains
pending until its terminal receipt and independent endpoint checks exist.
