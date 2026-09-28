# Preparation duration changes the optimization gradient

At the scaled comparison's final checkpoint, 120 seconds of unforced preparation
is insufficient to reproduce the longer-preparation gradient. This remains true
although its loss differs only slightly. Before extending that fit, use a longer
preparation and validate gradients as well as predictions.

The new `scripts/audit_preparation_gradients.py` evaluates the same frozen
checkpoint at one, two and four times its declared preparation duration. It
checks all fitting-backend source hashes, input hashes, training subset identity
and absence of selection trials before evaluation. No parameter is updated.
Only active raw-coordinate gradients enter the vector comparison. Two unit tests
cover analytic vector comparisons, zero norms and invalid inputs; they are also
included in the JAX CI job.

```sh
.venv-jax/bin/python scripts/audit_preparation_gradients.py \
  --checkpoint runs/capacity-scaling-curvature-v1-eval201/last-accepted.json \
  --manifest runs/capacity-scaling-curvature-v1-eval201/manifest.json \
  --training runs/overfit-seed1-training.json --graph runs/c302-audit.json \
  --output runs/capacity-scaling-preparation-gradients.json
```

The [receipt](capacity-preparation-gradients.json) retains exact input/script
hashes, all three losses and family gradient summaries. The run completed with
three objective/gradient evaluations in the worktree based on `cec2c3e`.
All three losses agree with the independent NumPy preparation audit within 1e-12.

| Preparation | Training MSE | Gradient comparison |
| --- | ---: | --- |
| 120 s | 0.0436563376839 | Original fitted objective |
| 240 s | 0.0436580387638 | Cosine −0.85904 vs 120 s; relative L2 difference 1.22972 |
| 480 s | 0.0436580387618 | Cosine 0.999999999982 vs 240 s; relative L2 difference 1.35e-5 |

The 240-to-480-second gradient difference has L2 norm 1.03e-8 and infinity norm
7.70e-9. This supports a stable longer-preparation gradient at this checkpoint,
not a guarantee at future iterates. The comparison is not a new finite-difference
check: the 120-second AD gradient can be correct for its finite-preparation
objective while differing materially from the longer-preparation objective.
Nor does the raw gradient comparison describe the actual L-BFGS search direction.

The prior 78.63% result remains valid for its declared 120-second objective,
and the scaling comparison remains valid under its common preparation. It does
not establish an equilibrated-model fit. The next fit should explicitly change
to 240-second preparation, initialize from the identical saved parameters,
record the resulting baseline loss shift, and reset optimizer history. Preparation
changes must be explicit in lineage and compatibility checks. Keep the 90% gate,
training-only scope and feature freeze, and recheck 480 seconds at its endpoint.
