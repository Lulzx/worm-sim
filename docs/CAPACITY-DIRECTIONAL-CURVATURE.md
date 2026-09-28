# Frozen-endpoint directional curvature

The larger-history fit remains nonstationary after its fixed evaluation budget.
Before changing dynamics or extending training, this diagnostic probes local
sensitivity at its last accepted checkpoint. It uses the same training-only
objective, exact input hashes and fitting backend sources. No checkpoint is
updated and no held-out observations are loaded.

`backends/jax/diagnose_capacity_curvature.py` selects the three parameter families
with largest active-gradient L2 norms, with alphabetical tie breaking. Within
each family it normalizes the negative gradient to a unit Euclidean direction,
leaving every other coordinate fixed. It evaluates symmetric perturbations of
0.001 and 0.0005 raw-coordinate units. The budget is 13 objective/gradient calls,
including the unperturbed checkpoint. Both the central loss difference and the
projected gradient difference are recorded. Analytic quadratic tests check these
formulas; invalid steps, directions and nonfinite probes are rejected.

```sh
.venv-jax/bin/python backends/jax/diagnose_capacity_curvature.py \
  --checkpoint runs/capacity-curvature100-eval201/last-accepted.json \
  --manifest runs/capacity-curvature100-eval201/manifest.json \
  --training runs/overfit-seed1-training.json --graph runs/c302-audit.json \
  --output runs/capacity-curvature100-directions.json
```

The [receipt](capacity-curvature-directions.json) records a successful 13-call run
from the working tree based on `a5a46b8`, with the exact diagnostic script hash.
The original fitting backend files were hash-checked against the fit manifest.
The baseline MSE reproduces the saved endpoint. This is a local probe, not a
refit or an optimizer comparison.

| Family | Gradient directional slope | Central loss slope, h=0.0005 | Gradient secant curvature, h=0.001 → 0.0005 |
| --- | ---: | ---: | ---: |
| Threshold | −0.001120 | −0.013322 | 362.05 → 330.73 |
| Rest | −0.000887 | −0.006829 | 216.51 → 204.82 |
| Chemical strength | −0.000145 | −0.000174 | 5.407 → 5.396 |

All positive-direction perturbations increase loss at these two step sizes,
despite following a negative gradient at zero. Curvature is positive and large
enough that these steps are outside the immediate descent interval. However,
the threshold and rest central loss slopes still differ substantially from their
AD slopes: these step sizes are too large to establish derivative agreement.
Their changing curvature estimates also preclude treating them as converged
infinitesimal measurements. This is not evidence that AD is incorrect.

The next diagnostic is a finer step sweep at this same frozen checkpoint to
establish the local derivative limit before choosing any coordinate scaling.
Do not infer a Hessian condition number, global capacity limit, or biological
importance from three gradient-aligned directions. This result does not meet
the training-capacity gate or justify new dynamical features.
