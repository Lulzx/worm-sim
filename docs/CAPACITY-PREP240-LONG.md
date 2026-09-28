# Longer fit with audited preparation

The [201-evaluation fit](CAPACITY-PREP240.md) reaches 78.82% training-response
capture, misses the 90% gate and remains nonstationary. Unlike its earlier parent,
its endpoint gradients agree across 240/480/960-second preparation. The next
[declared run](../configs/capacity-prep240-long.json) increases the budget to
1,001 actual objective/gradient evaluations and at most 1,000 accepted updates,
with no other optimizer, model, data or numerical changes.

This starts from the same original warm parent as the 201-evaluation run,
repeating the prefix to rebuild L-BFGS history. It does not resume optimizer
state from the final checkpoint, and it is not an independent initialization.
Repeating the prefix costs evaluations; the total budget includes them.

```sh
.venv-jax/bin/python backends/jax/overfit_lbfgs.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --warm-start runs/capacity-scaling-curvature-v1-eval201/last-accepted.json \
  --dt .005 --maxcor 100 --coordinate-scaling curvature-v1 \
  --preparation-seconds 240 --max-evaluations 1001 --max-iterations 1000 \
  --output runs/capacity-prep240-eval1001

python3 scripts/audit_lbfgs_prefix.py \
  --reference runs/capacity-prep240-eval201 \
  --candidate runs/capacity-prep240-eval1001 \
  --declaration configs/capacity-prep240-long.json \
  --output runs/capacity-prep240-long-prefix.json
```

Verify all 201 reference trial records and 190 accepted records, including
initialization, within the existing 1e-10 scalar tolerance. Inputs, parameter
lineage, fitting source hashes and configuration must match except declared
budgets; elapsed time is excluded. An interim partial receipt is explicitly
incomplete. Do not interpret the extra evaluations until the complete prefix
check passes.

Use the final accepted iterate for the unchanged 90% gate, and report best
accepted loss separately. Preserve all trials and nonfinite failure artifacts.
At termination repeat independent NumPy replay, step/preparation controls,
240/480/960-second gradient comparison, raw/scaled stationarity and frozen-gain
calibration. The longer budget is not a convergence guarantee. No budget increase,
new dynamics or held-out model selection is authorized by interim results.
The full specification and scientific acceptance remain incomplete.


## Repeated-prefix verification

The [complete prefix receipt](capacity-prep240-long-prefix.json) verifies all
201 reference objective/gradient trial records and all 190 accepted-state
records, including initialization, within the declared 1e-10 scalar tolerance.
The audit checked the declared input, parameter-lineage, fitting-source and
configuration agreement, allowing only the declared budget changes. Elapsed
time was excluded. The candidate snapshot contained 203 evaluation records and
191 accepted-state records; its file hashes identify that snapshot, not the
still-growing final history.

This passes the prerequisite for interpreting the additional optimization.
It does not establish bitwise parameter/optimizer-state equality, convergence,
the 90% capacity gate or held-out performance. The endpoint checks below
are now complete.


## Completed result

The [full receipt](capacity-prep240-long-results.json) preserves the manifest,
all 864 finite evaluations, all 664 accepted-state records including initialization,
endpoint audits and artifact hashes. The run used fitting source `df84327`.
It stopped after **663 accepted updates**, before its 1,001-evaluation budget,
on SciPy's relative-loss-change criterion (`ftol=1e-12`).

| Measure | Endpoint |
| --- | --- |
| Training MSE | 0.04358849292541925 |
| Training-response capture | 79.3136% |
| Unchanged capacity gate | 90%; failed |
| Best accepted checkpoint | Same as final accepted checkpoint |
| Raw gradient infinity norm | 9.53962e-4 |
| Optimizer-coordinate gradient infinity norm | 1.54095e-4 |
| Declared gradient tolerance | 1e-9; not met |
| Learned positive gains | 0.02826–931.47 |

This improves capture by 0.495 percentage points over the matched 201-evaluation
run. SciPy reports optimizer success, but its loss-change stopping condition
must not be confused with gradient convergence or the capacity gate.

Independent NumPy replay differs from JAX in MSE by 6.94e-18. Halving the time
step changes MSE by +3.91e-8 (maximum prediction difference 0.00437). Extending
preparation from 240 to 480 or 960 seconds changes MSE by about 3.47e-14 and
predictions by at most 3.40e-9. The 240-to-480-second gradient relative L2
change is 9.12e-7, with cosine essentially one; the 480-to-960 change is 3.42e-14.
These checks support endpoint numerical stability, not biological validity.

All 3,638 chemical coefficients are nonzero. With stimulation removed,
prediction energy is 2.83e-20 versus 0.00786 with stimulation. Frozen positive
per-neuron gain recalibration closes only 2.79% of the remaining mean-response
bound gap. Neither a dead chemical start nor output scaling alone explains the
remaining error at this endpoint.

**Decision:** the small-target test remains underfit and nonstationary by the
declared gradient criterion. The next bounded diagnostic should investigate
why the line search stalls while gradients remain nonzero, including local
directional finite differences and loss profiles at this exact checkpoint.
This result does not justify declaring a model-capacity limit, adding dynamics,
or selecting models on the previously inspected held-out targets. No further
fit or budget extension is included in this report. A fresh confirmatory
cohort remains unsecured; see the [cohort audit](FRESH-HOLDOUT-AUDIT.md).


## Local descent and loss-profile audit

The [19-evaluation receipt](capacity-prep240-long-directions.json) tests the
three families with largest active-gradient L2 at the frozen endpoint. Each
direction is that family's normalized negative gradient in raw coordinates;
all other parameters stay fixed. Symmetric steps are 1e-5, 1e-6 and 1e-7.
The diagnostic checks fitting-source and input hashes and reproduces the
checkpoint MSE. It ran from `5527364` without backend changes.

```sh
.venv-jax/bin/python backends/jax/diagnose_capacity_curvature.py \
  --checkpoint runs/capacity-prep240-eval1001/last-accepted.json \
  --manifest runs/capacity-prep240-eval1001/manifest.json \
  --training runs/overfit-seed1-training.json --graph runs/c302-audit.json \
  --steps 0.00001 0.000001 0.0000001 \
  --output runs/capacity-prep240-long-directions.json
```

| Family | Relative slope disagreement at h=1e-7 | Gradient secant curvature |
| --- | ---: | ---: |
| threshold | 2.45e-07 | 612.836333 |
| rest | 1.74e-07 | 382.563942 |
| chemical_strength | 6.95e-07 | 12.458264 |

All three negative-gradient steps reduce loss at h=1e-6 and h=1e-7. The central
loss derivatives agree with autodiff to relative error below 1e-6 at the finest
step. These directional checks demonstrate remaining local descent; they do
not verify every coordinate or prove that the 90% capacity gate is attainable.

The threshold direction has a sharp excursion: at the **negative** 1e-5 probe,
MSE is 2.016545618, while its positive probe is 0.043588511. At 1e-6 both sides
remain near the endpoint and yield curvature about 612.84. This resembles the
large rejected losses in the training history, but the original line-search
parameter vectors were not saved, so the probes do not reproduce those exact
trials or establish their cause.

**Next decision:** check the high-loss threshold probe itself with independent
replay, a halved integration step and longer preparation. Endpoint stability
does not establish stability of nearby trial parameters. Distinguish a numerical
failure from sensitivity to preparation or a different dynamical response before
changing optimizer safeguards or launching another fit. No probe is promoted
to a fitted checkpoint, and no held-out scoring or new dynamics are introduced.
