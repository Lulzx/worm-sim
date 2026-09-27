# Reverse-mode differentiation options

Specification §6's continuous-adjoint alternative and an explicit recursive
checkpoint budget are available in the JAX adaptive solver configuration. Both
use Diffrax implementations. The existing checkpointed method remains the default;
the compatibility Euler path is unchanged.

## Configuration

For a bounded number of solver checkpoints:

```json
{
  "method": "tsit5",
  "rtol": 1e-8,
  "atol": 1e-10,
  "adjoint": "checkpoint",
  "checkpoints": 8
}
```

`checkpoints` is a positive integer or null. Null preserves Diffrax's automatic
choice based on `max_steps`. This controls checkpoints inside each fast solve,
not a total process-memory budget. Trial outputs, parameter gradients, and state
carried across input intervals or multirate windows require additional memory.

For the optional continuous method:

```json
{
  "method": "tsit5",
  "rtol": 1e-8,
  "atol": 1e-10,
  "adjoint": "continuous",
  "adjoint_rtol": 1e-9,
  "adjoint_atol": 1e-11,
  "adjoint_max_steps": 100000
}
```

These objects go in `configuration.solver` in the
[extended fitting path](EXTENDED-FITTING.md). The `Adaptive` Python class accepts
the same fields. Saved checkpoints preserve them, so training and replay retain
the forward solver settings, and subsequent gradient evaluations retain the
adjoint settings.

The continuous backward solve uses the same solver family and `dtmax` as the
forward solve. Omitted backward tolerances and step limits inherit their forward
values. Backward step exhaustion raises an error. Checkpoint budgets are rejected
in continuous mode, and backward-solver options are rejected in checkpoint mode;
settings are never silently ignored.

## What the two methods differentiate

[Diffrax's adjoint documentation](https://docs.kidger.site/diffrax/api/adjoints/)
distinguishes differentiation of the numerical solve from solving the continuous
sensitivity equations backward in time. Its recommended checkpointed method
uses recomputation to reduce retained solver state. The continuous method returns
approximate gradients and is not generally preferred. Neither option establishes
a memory or speed advantage for WormSim without workload measurements.

WormSim passes differentiable parameters and carried state through the solver's
`args` and `y0` paths. Preparation and every constant-current interval are separate
solves, so a backward solve does not cross an unmodeled current jump. With
multirate enabled, each fast window uses the selected adjoint; JAX differentiates
the outer state transfer and coarse concentration update. Thus that path combines
continuous fast-window sensitivities with differentiation of the discrete split.

Simulation time grids, event times, and solver settings are fixed configuration,
not learned variables. These reverse-mode paths do not provide JAX forward-mode
JVP support. Backward reconstruction can be ill-conditioned for strongly
dissipative or long trajectories. An implicit backward solve can also become
expensive as the augmented adjoint state grows with parameter count. The option
must be checked on the intended workload; a finite gradient is not proof of an
accurate gradient.

## Validation and open acceptance work

Tests compare decay objectives and gradients with an analytic solution using both
Tsit5 and Kvaerno5; compare continuous and checkpointed gradients through prepared
responses with discontinuous currents; and check modulation/plasticity gradients
with multirate updates against finite differences. Tight forward tolerances are
used for the analytic interior-output check because saved-value interpolation
error can exceed a solver's local error tolerance.

Invalid/conflicting settings fail, backward budget exhaustion propagates, and
both adjoint modes are exercised in the synthetic population-fit/checkpoint-reload/
Rust-scoring pipeline. These are small-system engineering checks. Long-duration
(~1,000 s) gradients, full-network error and memory measurements, and biological
validation remain open. No new real-data fit or performance target is claimed.
