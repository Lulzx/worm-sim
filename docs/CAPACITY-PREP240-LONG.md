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


## High-loss perturbation audit

The [excursion receipt](capacity-prep240-long-excursion.json) reconstructs the
negative 1e-5 threshold probe from the frozen endpoint gradient. Its JAX MSE
exactly matches the earlier directional receipt. The generated artifact is
explicitly an **unaccepted diagnostic perturbation**, not a fitted checkpoint.

```sh
.venv-jax/bin/python scripts/audit_capacity_excursion.py \
  --checkpoint runs/capacity-prep240-eval1001/last-accepted.json \
  --manifest runs/capacity-prep240-eval1001/manifest.json \
  --training runs/overfit-seed1-training.json --graph runs/c302-audit.json \
  --directions runs/capacity-prep240-long-directions.json \
  --output runs/capacity-prep240-long-excursion.json
```

| Probe replay | Training MSE |
| --- | ---: |
| JAX, dt 0.005, preparation 240 s | 2.016545618060591 |
| Independent NumPy, same settings | 2.016545618060596 |
| NumPy, dt 0.0025 | 2.016271150501991 |
| NumPy, preparation 480 s | 2.016545617649954 |
| NumPy, preparation 960 s | 2.016545617649954 |
| NumPy, no stimulation | 0.051455059101597 |

Halving the step changes predictions by at most 0.00396. Longer preparation
changes them by at most 7.05e-9, and the maximum prepared-state derivative falls
from 5.17e-12 to 6.76e-14. The no-stimulation prediction energy is 2.74e-21.
The high loss therefore survives these numerical/preparation controls.

A separate [prepared-state comparison](capacity-prep240-long-excursion-state.json)
finds a maximum voltage difference of **1.336 normalized units** between the
endpoint and probe, despite their threshold displacement having length only
1e-5. RMHL changes from -1.03416 to 0.301913. Calcium and synaptic-release
components change by up to 0.9681 and 0.4841. This comparison constructs the
independent `Replay` for each saved base model with the same graph and compares
its `.state` after preparation; input and replay-source hashes are recorded.

**Interpretation:** the large response change is accompanied by a large change
in the prepared resting state. The evidence is consistent with switching between
attraction basins during preparation, rather than merely a large local curvature
within the endpoint's resting state. This is not a formal bifurcation analysis:
step refinement is limited, and the original rejected optimizer vectors were
not retained. The next diagnostic is a frozen-parameter cross-initialization
of the two resting states to test whether both persist at the same parameters.
Do that before choosing a remedy for initialization or line-search behavior;
no new fit, model feature or held-out selection has been performed.


## Cross-initialization result

The [eight-case receipt](capacity-prep240-long-resting-states.json) fixes each
parameter set in turn, initializes from each previously prepared state, and
relaxes without input for 480 seconds at dt 0.005 and 0.0025. It records all
prepared states, training-response losses, input hashes and replay-source hash.

```sh
.venv-jax/bin/python scripts/audit_capacity_resting_states.py \
  --checkpoint runs/capacity-prep240-eval1001/last-accepted.json \
  --probe runs/capacity-prep240-long-excursion.probe.json \
  --excursion runs/capacity-prep240-long-excursion.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --output runs/capacity-prep240-long-resting-states.json
```

| Parameters | Initial prepared state | MSE, dt 0.005 | MSE, dt 0.0025 |
| --- | --- | ---: | ---: |
| Endpoint | Endpoint | 0.043588493 | 0.043588532 |
| Endpoint | High-loss probe | 2.018213887 | 2.017939861 |
| High-loss probe | Endpoint | 0.043588536 | 0.043588589 |
| High-loss probe | High-loss probe | 2.016545618 | 2.016271150 |

All eight final derivative infinity norms are below 2e-13. For each fixed
parameter set and step size, the final states from the two seeds remain more
than 1.33 units apart in infinity norm. This is numerical evidence for two
coexisting stationary states under the same parameters. It is not a formal
stability or bifurcation proof; no eigenvalue or neighborhood-stability test
was performed. It also does not establish which state is biologically correct.

**Decision:** investigate a deterministic prepared-state initialization for the
next bounded fitting experiment. Freeze the endpoint's low-loss resting state
as the common initial vector for every objective evaluation, keep unforced
preparation, and record the vector and its provenance. Before fitting, check
baseline loss, gradients and the previously troublesome threshold perturbation
under that exact objective. The baseline result above suggests this will retain
the endpoint response, but JAX gradients under the new seed remain unverified.

This is an explicit change to the initialization contract, not a transparent
optimizer restart. Never reseed from the previous accepted/trial state inside
an objective call: that would make results depend on evaluation order. Preserve
the original-seed result and failed capacity gate. This diagnostic does not
select among held-out responses, change dynamics, or launch another fit.


## Fixed-seed objective preflight

The [ten-evaluation audit](capacity-fixed-seed-audit.json) freezes the endpoint's
NumPy-prepared state as a constant initial vector and keeps 240 seconds of
fully differentiated unforced preparation. Parameters and data are unchanged.

- Baseline MSE: 0.04358849292545398 (change 3.47e-14).
- Active-gradient relative L2 change: 9.12e-7; cosine essentially one.
- Central finite differences at h=1e-6 agree with negative-gradient slopes in
  threshold, rest and chemical-strength families (relative errors below 2.3e-5).
- The old high-loss threshold perturbation now gives MSE 0.043588535892291544.
- Repeating the baseline after all perturbations changes neither loss nor gradient.

The runner now accepts `--fixed-seed-audit` explicitly. It checks the receipt's
parent checkpoint, graph and training hashes and audit-script hash, requires a
finite state of unchanged dimensions, and records the receipt hash in the run
manifest. The seed is fixed for every evaluation and saved in every checkpoint.
Subsequent continuations inherit that seed and retain the original rest-parameter
initialization used to reconstruct frozen coordinates. Other warm-start lineage
and parameter checks remain enforced. This changes the initialization contract;
it does not establish convergence, global smoothness or scientific validity.

```sh
.venv-jax/bin/python scripts/audit_capacity_fixed_seed.py \
  --checkpoint runs/capacity-prep240-eval1001/last-accepted.json \
  --manifest runs/capacity-prep240-eval1001/manifest.json \
  --training runs/overfit-seed1-training.json --graph runs/c302-audit.json \
  --output runs/capacity-prep240-fixed-seed.json
```

The audit ran against the historical fitting-source hashes before the runner
change. Reproducing it later requires that historical backend snapshot. The
receipt preserves exact hashes; it must not be regenerated with mismatched
sources. The new runner uses the recorded fixed vector, without rerunning or
mutating the historical audit.

Five overfit tests and seven quasi-Newton tests pass, including explicit seed
permission, dimension/finiteness rejection and parameter preservation. Two
[one-evaluation smoke receipts](capacity-fixed-seed-smoke.json) verify the explicit
seed and its inheritance by a continuation. Both reproduce MSE
0.04358849292545398 with zero accepted updates. These ran from the implementation
worktree and record its dirty state and exact backend hashes. No capacity
improvement is claimed from these smoke tests.
