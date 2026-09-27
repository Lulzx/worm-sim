# Adaptive and implicit integration

The JAX backend now exposes `Adaptive(method="kvaerno5")` and
`Adaptive(method="tsit5")` through `Level0(..., adaptive=settings)`. Diffrax supplies
Kværnø 5 implicit integration, Tsitouras 5 explicit integration, PID error control,
nonlinear/linear solves and recursive checkpointed reverse differentiation. No
new solver or hand-derived adjoint is implemented here. Float64 is retained.

```python
from level0 import Level0, parameters, response
from solvers import Adaptive
import jax.numpy as jnp

engine = Level0(model, graph, times,
                Adaptive(method="kvaerno5", rtol=1e-7, atol=1e-9,
                         dt0=1e-3, dtmax=None, max_steps=100_000))
prediction = response(engine, parameters(model), jnp.asarray(target_index))
```

All tolerances and step settings are explicit and validated. `max_steps` applies
per continuous-input interval; solve statistics sum accepted/rejected steps over
the whole preparation and response. Exhausting the budget raises an error rather
than returning a partial trajectory as successful. Adaptive mode avoids creating
the fixed Euler timeline. Calling `Level0` without settings retains the exact
Euler compatibility path used for the reproduced fit.

## Input discontinuities

The positive kernel is constant between observation boundaries, with a separate
unforced preparation interval. The adaptive implementation solves each such
interval with its input held fixed, carrying the full state and its gradients
through a JAX scan. Each interval resets the controller. This is important for
Kværnø 5: the installed tableau includes an internal stage at approximately 1.2303
step lengths. A stage can therefore evaluate beyond the interval's nominal end.
Merely clipping step endpoints at input jumps still allowed that stage to read
the next input and produced an incorrect kernel gradient in the first check.
Holding the interval's forcing constant resolves the discrepancy. No gradient is
stopped at interval boundaries or at the end of preparation.

Tests cover a 1 ms membrane with 1 s calcium kinetics and discontinuous input,
against the exact passive membrane solution; reverse gradients through preparation
and kernel values against finite differences; settings validation; and explicit
failure when the solver runs out of steps. The previous Euler forward/gradient,
objective, optimizer and audit tests continue to pass. These small checks do not
prove all stiff biological models, long-horizon gradients or GPU workloads.

## Frozen network comparison

```sh
.venv-jax/bin/python backends/jax/compare_solvers.py \
  --model runs/level0-atlas-classification-fit/selected.json \
  --graph runs/c302-audit.json --target AVAL \
  --output runs/adaptive-c302-comparison
```

This command compares explicit/implicit adaptive full-state trajectories at fixed
parameters, recording input/source hashes, settings, step counts and timings.
It does not fit data or select a model. Dense implicit solves may be costly on the
full graph; no speedup or memory-efficiency claim is implied by library support.
The compatibility fit runner retains Euler. The separate
[extended fitter](EXTENDED-FITTING.md) serializes adaptive settings and evaluates
training and validation dynamics with the same solver; Rust scores the generated
predictions instead of rerunning them with a different integrator. [Multirate modulation](MULTIRATE.md) is also available with explicit coarse-step
settings and convergence checks. Continuous adjoints, truncated training and
distributed execution remain separate §6 requirements.

Primary documentation: [Diffrax ODE solvers](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/),
[step controllers](https://docs.kidger.site/diffrax/api/stepsize_controller/),
[adjoints](https://docs.kidger.site/diffrax/api/adjoints/).

## Full-network numerical receipt

At clean source `3c70526`, both adaptive solvers completed the frozen selected
neutral joint-fit model on the 302-neuron graph, with AVAL stimulated, 60 seconds
of preparation and 40 observations through 19.5 seconds. The
[receipt](adaptive-c302-comparison.json) records model/graph/state-array hashes,
settings, device, source and step counts. No parameters were updated.

| Relative / absolute tolerance | Maximum Tsit5–Kvaerno5 state difference | Tsit5 seconds | Kvaerno5 seconds |
| --- | --- | --- | --- |
| 1e-7 / 1e-9 | 3.26e-7 | 1.250 | 5.940 |
| 1e-9 / 1e-11 | 1.13e-9 | 0.613 | 8.923 |

Tightening tolerances changed Tsit5 states by at most 3.33e-7 and Kvaerno5 states
by at most 5.38e-8. The explicit/implicit discrepancy decreased by roughly two
orders of magnitude. This is convergence evidence for one frozen full-network
trajectory, not a global error bound or a test of all stimulation targets.
Timings are CPU wall time including compilation/cache effects; the faster second
Tsit5 invocation is not evidence that stricter tolerances improve performance.
The implicit solver was slower here despite fewer steps at the looser tolerance.
No claim is made that this fitted Level 0 configuration requires stiff integration.
