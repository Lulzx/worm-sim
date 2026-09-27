# Multirate neuromodulation

Specification §6's optional multirate path holds neuromodulator concentrations
constant during each fast integration window and updates them at the window end.
It is available through the JAX Level 0 API and the extended fitting/checkpoint
path. The default remains the existing fully coupled solver.

## Numerical contract

1. Subdivide every preparation/stimulus/output interval into windows no longer
   than `slow_dt`. Include all original boundaries exactly; no window crosses a
   current-kernel change. A short final window is allowed.
2. Hold the concentration vector at its left-boundary value. Use Diffrax Tsit5 or
   Kvaerno5 to evolve membrane, calcium, gates, and optional plasticity. Integrate
   the per-channel release drive alongside those fast states.
3. At the right boundary, take the average release drive over that window and
   update each concentration using its exact constant-drive relaxation:

```text
mean_drive = integral(sum(alpha * release[pre]), over window) / h
c_next = exp(-h / tau) * c + (1 - exp(-h / tau)) * (mean_drive + bath)
```

The implementation evaluates the second coefficient with `-expm1(-h/tau)`.
Bath is the existing constant concentration-drive term, not a timed drug input.
The new concentration affects the next fast window. Saved concentrations are the
post-update values at the requested output boundary; the fast states remain
continuous. Preparation follows the same rule and carries every state into the
response, with gradients through both the integral and concentration updates.

This splitting is **first order in the coarse step**. Using a high-order fast
solver does not remove concentration-lag error, nor does tightening its tolerance
control splitting error. The exponential update is exact for constant drive;
replacing a time-varying drive with its unweighted window mean is an approximation.
Refine `slow_dt` and compare with the fully coupled solution before interpreting
results. Tiny fast-solver error alone is not evidence of convergence.

## Configuration

Add this optional field alongside `solver` and `extensions` in the
[extended configuration](EXTENDED-FITTING.md):

```json
"multirate": {"slow_dt": 0.1, "max_windows": 100000}
```

It requires both a modulation module and a non-null adaptive fast solver.
The low-level equivalent is `Level0(..., adaptive=Adaptive(...), modulation=mod,
multirate=Multirate(slow_dt=0.1))`. A missing or null `multirate` retains fully
coupled dynamics. The checkpoint preserves the coarse-step settings, so replay
and validation use the same method as fitting. The
[synthetic fitting example](../backends/jax/examples/extensions-synthetic.json)
exercises this option with all four coupling modules.

`max_windows` caps the whole run, including preparation. Diffrax's `max_steps`
separately caps fast steps per window. Invalid settings, unavailable required
modules, and an exceeded window budget fail explicitly. Returned solver statistics
sum fast accepted/rejected steps and report `slow_windows`.

## Validation and resource limits

Tests check the analytic constant-bath concentration trajectory, unchanged fast
trajectories when receptor feedback is zero, exact boundary inclusion, coarse-step
convergence toward fully coupled integration, and finite-difference gradients of
slow parameters, plasticity, and stimulus parameters through preparation. Both
Tsit5 and Kvaerno5 are exercised. Checkpoint round-trip and the cross-language
fit/reload/Rust-scoring smoke test include multirate settings.

The scan retains only requested output states, the current state, and accumulated
step counts as explicit forward outputs. It does not materialize a full state
matrix at every coarse boundary. Its grid still uses storage proportional to the
window count, and reverse AD/solver checkpoints require additional memory. This
is not a measured peak-memory or speed improvement. Restarting the fast solver
at each boundary can increase cost, especially when the coarse step is too small.

Diffrax provides the [fast solvers and step control](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/)
and [checkpointed reverse differentiation](https://docs.kidger.site/diffrax/api/adjoints/).
The held-concentration split is an explicit WormSim numerical choice. Full-network
error/cost tradeoffs, biological validation, and specification performance targets
remain open.
