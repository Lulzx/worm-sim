# Atlas learning-rate schedules

The Rust atlas fitter supports optional constant or cosine learning-rate
schedules. Missing configuration preserves the original constant-rate Adam
behavior. For cosine decay, add this object to the fit configuration:

```json
"learning_rate_schedule": {"kind": "cosine", "minimum_fraction": 0.1}
```

The minimum fraction is a declared choice in [0, 1], not an empirically selected
default. With N updates and one-indexed update k, the rate is

```text
base_rate * [minimum_fraction
             + (1 - minimum_fraction) * (1 + cos(pi * (k - 1)/(N - 1)))/2]
```

The first update uses the base rate and the last uses base rate times the
minimum fraction. A one-update fit uses the base rate. Epoch zero only evaluates
initial parameters and has no applied rate. A minimum fraction of zero produces
a final zero-rate update: gradients and Adam moments are still computed, but
parameters do not move. The total update count is part of the frozen schedule;
changing it defines a different optimization trajectory.

Each epoch report persists `applied_learning_rate`, with null at epoch zero.
The independent fit auditor recalculates these rates from the configuration.
It accepts earlier receipts without schedule metadata. Strict configuration
parsing rejects unknown schedule fields; invalid rates, fractions and update
indices are errors.

The existing optimizer remains full-batch Adam with beta1=0.9, beta2=0.999,
epsilon=1e-8 and global gradient-norm clipping at 10. The schedule changes only
the update rate. Prior penalties, parameter sharing, frozen calcium scales,
validation-MSE checkpoint selection and data partitions keep their declared
semantics. AdamW, group-specific rates and random restarts remain separate
implementation requirements; this change does not claim them.

Tests cover endpoints, monotonicity, single-update behavior, constant-rate
compatibility, strict serialization and invalid settings. A fit-level test checks
the actual parameter updates: cosine decay to zero preserves the previous
checkpoint's parameters, kernel and observation gain, while the matched constant
schedule continues updating. Neither a real-data cosine comparison nor a
convergence claim has been made. The ongoing 25-update gain/control comparison
uses its original constant rate and source `dcdd0ff`.
