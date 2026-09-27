# Trainable global atlas observation gain

The [training diagnostic](ATLAS-TRAINING-DIAGNOSTIC.md) found that a training-only
gain of 9.8391 improved validation MSE for the five-update neutral-prior model.
The Rust atlas fitter now optionally learns one positive global observation gain
jointly with dynamics, input kernel and pair classifier. Fluorescence is
`exp(observation_log_gain) * calcium_scale[i] * (calcium[i] - baseline[i])`.
Calcium scale remains fixed at one and the relative readout has zero offset.

Configuration `observation_gain` supplies `initial_gain` and `prior_strength`.
The optional penalty is strength times squared displacement of log gain from
its initial value. It is applied once, not multiplied by the neuron count. The
shared log-gain derivative sums all per-neuron readout derivatives from the full
prepared-response adjoint. Both trace MSE and pair BCE contribute. Training
weights and priors retain their previous normalization. The gain adds exactly
one trainable parameter and is saved in every checkpoint.

Absent gain configuration and state preserve the original unit readout, including
old serialized models. Inconsistent configuration/state, nonfinite values, and
gain overflow/underflow are rejected. The independent NumPy replay applies the
saved gain and checks its initialization and parameter count in the fit audit.
Tests cover finite differences, the first Adam update against an independent
forward loss, fluorescence and pair-label leakage, threefold prediction scaling,
legacy behavior and malformed gain states.

This global nuisance parameter is a staged fit improvement, not completion of
the specification's per-neuron/per-animal observation model. It is not a measured
optical calibration. It can trade off against input amplitude and neural response
parameters, so neither its fitted magnitude nor the kernel magnitude uniquely
identifies a biological quantity.

## Frozen next comparison

- [Longer unit-gain control](../configs/level0-atlas-longer-fit.json): 25 full-batch
  updates, otherwise identical to the neutral-prior five-update joint model.
- [Learned-gain fit](../configs/level0-atlas-gain-fit.json): the same 25-update
  protocol, with initial gain 10 and zero gain penalty. The initial value rounds
  the preceding training-only diagnostic (9.8391); it is not estimated from test
  responses. The decision to try it was also informed by validation improvement.

Both start neural/kernel/classifier parameters from the original initialization;
neither resumes an optimizer or a selected checkpoint. They use neutral sign
priors, dt 0.01 s, 60 s preparation, learning rate 0.01, and the same training
trials and pair labels. Validation trace MSE selects among epochs 0 through 25,
with earlier exact ties retained. Compare each against LDS and the earlier joint
fit through the common scorer and paired cluster intervals; also compare the
two longer runs. This contrast combines gain initialization with learning the
gain and cannot separate those two effects. No threshold is selected by test.

Twenty-five updates are a declared longer run, not a convergence guarantee. These
remain exploratory comparisons on the previously inspected cohort. Evaluate
training error reduction and numerical sensitivity as well as held-out scores.
Independent replay must pass before interpreting a result. Results are pending.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example fit_level0_atlas
target/release/examples/fit_level0_atlas data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json configs/level0-atlas-longer-fit.json runs/level0-atlas-longer-fit runs/randi-pairs.json
target/release/examples/fit_level0_atlas data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json configs/level0-atlas-gain-fit.json runs/level0-atlas-gain-fit runs/randi-pairs.json
```
