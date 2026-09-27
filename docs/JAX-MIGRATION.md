# Hybrid backend migration

The user revised the original Rust-only fitting direction. Rust continues to own
imports, fixed splits, compact artifacts and common benchmark scoring. New model
fitting uses established JAX ecosystem components. Existing Rust numerical code
remains available for independent comparison, without requiring each new model
feature to have a manually derived gradient.

The first migration check is complete at source
`06633467f3e40044c5ba46363932a2964461e699` with a clean worktree.
The frozen five-update joint trace/BCE model from `fe2972f` was replayed through
JAX/Equinox dynamics and Diffrax Euler integration, then scored by the unchanged
Rust evaluator. [Machine-readable receipt](jax-level0-replay.json) records source,
input/output hashes, package versions, device, timings and scoring provenance.

| Check | Result |
| --- | --- |
| Held-out targets / trials | 15 / 166 |
| Maximum absolute fluorescence difference | 1.6653e-16 |
| Rust-scored test pair AUROC | 0.6810058792712033, identical to reference |
| Rust-scored test MSE | 0.04989684961428055 |
| Macro trace correlation | 0.0021261078038790847 |
| Synthetic reverse gradients | Finite differences pass across membrane, calcium, chemistry, gap, gate, kernel and observation gain coordinates |
| Batched target check | JAX vmap agrees with independent NumPy replay |
| Optimizer check | Optax Adam lowers a composed synthetic loss |

Macro correlation differs from the original 0.002125397953460259 by approximately
7.1e-7 despite machine-precision fluorescence agreement: normalized correlations
of almost-flat responses are numerically sensitive. This is not a biological
improvement. Pair ranking and pooled MSE reproduce the reference result.

This run used float64 on JAX CPU on the local Apple Silicon machine. The first
target took 0.545 s including compilation; subsequent targets took 0.080–0.082 s,
including preparation. These are forward replay timings with other work running,
not batch-gradient benchmarks, GPU measurements or an optimizer speed comparison.

The backend uses Diffrax's checkpointed reverse AD and Optax; no custom adjoint
was added. Reproduction currently covers the saved model's forward result and
small synthetic derivative checks, **not the full training trajectory**. Next:
implement the train-only objective and frozen masks, checkpoint and selection
contracts, then compare the fitting trajectory before changing model structure.
Further spec implementation should reuse library solvers, inference and neuron
components where their equations and contracts fit the specification.

Reproduction commands and limitations are in the [backend README](../backends/jax/README.md).


## Training objective and first update

The native exporter now supplies validated training-only sufficient statistics
and pair labels. JAX computes trace MSE, pair BCE, optional stabilized correlation,
raw-coordinate/kernel/gain priors and molecular sign penalties using automatic
differentiation. Frozen coordinates are excluded from updates.

The [first-update receipt](jax-first-update.json) at clean source `dc8ef3d`
checks all 161 training targets at the frozen neutral initialization. MSE
0.07536645831538034 and BCE 0.19570439936652007 match the Rust report. The prior
value differs by 5.14e-16. An Optax Adam update with the reference clipping and
moments matches epoch-one tied parameters within 1.9078e-11; kernel error is
8.88e-16 and classifier error is zero. This is one-step parity, not yet full
training reproduction. The CPU pass including compilation took 139.54 seconds;
other jobs and cache conditions differ from earlier timings, so this is not a
controlled speedup measurement.
