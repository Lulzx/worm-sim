# Learning rates for parameter groups

The extended JAX fitter supports §6/§8's group learning rates through declared
multipliers of the existing global schedule. Optax still supplies Adam/AdamW,
moments, cosine scheduling, and global gradient clipping. This changes optimizer
configuration, not the model equations or validation-selection rule.

## Configuration

Add an optional `optimization` object to an
[extended configuration](EXTENDED-FITTING.md):

```json
{
  "optimization": {
    "learning_rate_multipliers": {
      "base_types": {"chemical_strength": 0.5, "chemical_sign": 0.25},
      "base_groups": {"chemical_strength/annotated:N000->annotated:N001": 0.1},
      "parameters": {"kernel": 0.5, "modulation.raw_tau": 0.0}
    }
  }
}
```

The exact group in this example belongs to the synthetic fixture, not a biological
assignment. Selectors are explicit:

- `base_types` matches the portion of each native tied-group name before its first
  `/`, such as `tau`, `chemical_strength`, `chemical_sign`, or `gap_strength`.
- `base_groups` matches an entire entry in `base_model.parameters.groups[].name`
  and overrides that group's type multiplier.
- `parameters` matches a non-base parameter array: `kernel`, `log_gain`,
  `classifier`, or a module array such as `modulation.raw_tau`,
  `modulation.sensitivity`, `dark_edges.raw_strength`, `rectification.asymmetry`,
  or `plasticity.raw`. Every element of that array receives the multiplier.

Unspecified multipliers are one. Unknown selectors, duplicate native group names,
negative/nonfinite multipliers, booleans, or overflowing effective rates fail.
A selector for an absent module or classifier fails instead of being ignored.
The setting cannot unfreeze native frozen coordinates or inactive plasticity
parameters. Exact base-group selectors preserve the existing parameter tying:
they change the shared coordinate's rate, not individual members of that group.

## Update semantics

At update k, coordinate i has effective learning rate
`global_schedule(k) * multiplier[i]`. The fitter first zeros frozen gradients and
clips the shared active-gradient norm to 10, then obtains the normal Optax update,
then scales that complete update by the declared multipliers. Scaling the completed
update, rather than the gradient entering Adam, preserves the intended relative
learning rates after moment normalization. For AdamW the same multiplier also
scales decoupled weight decay. Configurations with effective base learning rate
times weight decay above one are rejected, following the native fitter's guard.

A zero multiplier freezes the coordinate, excludes its gradient from clipping,
and prevents AdamW decay. Freezing this way preserves the numerical loss value:
its prior contributions remain constant while their gradients are stopped.
Originally excluded native frozen-coordinate priors retain their existing rule.
The global schedule and epoch count still advance together; there are no separate
per-group schedule counters.

Checkpoint envelopes retain the configuration. Reload reconstructs active masks,
checks fixed extension coordinates, and reports the resulting active-parameter
count in prediction metadata. Optimizer moments are not serialized; nonzero-epoch
resume remains unsupported. Changing rates during a run is not implemented.

## Evidence

Tests compare several vector-scaled AdamW updates, including a cosine endpoint,
against separate Optax optimizers with the corresponding rates. They check selector
precedence, invalid settings, zero gradients without changed objective values,
frozen coordinates under weight decay, and checkpoint reload. The synthetic
Rust/JAX fitting-and-replay test also uses non-unit rates and a frozen modulation
time constant in both adjoint modes. Default unit-rate behavior remains covered by
the existing optimizer and objective tests.

No real-data optimization sweep or improved benchmark result is claimed. Priors,
per-type biological calibration, optimizer resume, and scientific validation
remain separate requirements.
