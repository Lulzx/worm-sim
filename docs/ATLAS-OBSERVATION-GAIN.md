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
Independent replay must pass before interpreting a result. Completed results
appear below.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example fit_level0_atlas
target/release/examples/fit_level0_atlas data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json configs/level0-atlas-longer-fit.json runs/level0-atlas-longer-fit runs/randi-pairs.json
target/release/examples/fit_level0_atlas data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json configs/level0-atlas-gain-fit.json runs/level0-atlas-gain-fit runs/randi-pairs.json
```

## Completed comparison: gain helps, but no LDS superiority

Both runs completed all 25 updates from source
`dcdd0fffc559f393c8c5e6838f88ba2b755d1d06`. The unit-gain control selects epoch 8
by validation MSE; subsequent updates worsen that criterion. The gain model
selects epoch 25, so optimization convergence is not established. Its learned
gain is **11.20956**, from initial gain 10. The control has 6,783 trainable
parameters and the gain model 6,784. The control's parameters exactly reproduce
the earlier joint fit through its first five updates.

| Metric | Unit-gain control | Learned gain | LDS |
| --- | ---: | ---: | ---: |
| Selected validation MSE | 0.04232169 | 0.04157182 | 0.04008532 |
| Test MSE | 0.04989414 | 0.04904275 | 0.04734484 |
| Test macro trace correlation | 0.0039633 | 0.0119434 | 0.0452857 |
| Test pair AUROC | 0.684178 | 0.698658 | 0.686187 |

Paired stimulated-target bootstrap differences (first minus second) are:

| Comparison | MSE difference [95% interval] | AUROC difference [95% interval] |
| --- | ---: | ---: |
| Gain minus unit gain | -0.0008514 [-0.0014839, -0.0001942] | +0.014480 [+0.002455, +0.023281] |
| Gain minus LDS | +0.0016979 [+0.0005889, +0.0025406] | +0.012472 [-0.019166, +0.046809] |
| Unit gain minus LDS | +0.0025493 [+0.0011342, +0.0037073] | -0.002008 [-0.034036, +0.035650] |

Gain improves both metrics over the matched unit-gain control in these conditional
intervals, but remains worse than LDS in MSE. Its AUROC interval against LDS
includes zero; neither superiority nor equivalence is established. The contrast
combines amplitude initialization and learning the gain, and does not isolate
their individual contributions. All 12,588 test trace correlations are defined.
The comparison uses 15 trace targets, 13 eligible pair-label targets, 2,000 paired
draws and seed 42. Separate recording and target resampling does not jointly
account for crossed dependence; recording IDs are not verified animals. These
are exploratory results on the repeatedly inspected cohort, conditional on the
frozen fits and without training-uncertainty coverage.

The gain model's test classifier BCE is 0.186414 versus 0.188609 for the constant
training-prevalence prediction. Its Brier score is **worse**, 0.0445496 versus
0.0444772. These are point comparisons without calibration intervals; there is no
blanket calibration-improvement claim. Common AUROC uses raw absolute response
area, not fitted classification probabilities.

Independent NumPy dynamics replay checks all 30 validation/test target grids per
model. Maximum fluorescence errors are 1.67e-16 (control) and 2.49e-15 (gain).
The auditors also verify selection, saved scores, pair areas, AUROC, parameter
counts and classifier outputs. They do not independently replay optimization or
bootstrap draws. Extending preparation to 120/240 s changes validation AUROC by
at most 0.000162 (control) and 0.000075 (gain); including the half-step check raises
the maximum changes to 0.000174 and 0.000249. Halving dt changes validation MSE
by 2.27e-8 and 5.64e-7, with maximum fluorescence changes 5.46e-5 and 6.01e-4.
Near-zero trace correlations and their defined counts remain numerically sensitive.

### Training diagnosis

The selected control is byte-identical to the already audited epoch-8 checkpoint:
training MSE 0.07535556 and **0.918%** of training mean-trace energy captured.
The selected gain model reaches training MSE **0.07474287** and captures **5.831%**.
Both remain far above the empirical shared-response training bound 0.06300051.
The gain model's pair-mean shape loss at diagnostic epsilon 0.01 is **1.00266885**,
corresponding to slightly negative stabilized correlation; this loss was not used
in fitting. Amplitude alone leaves substantial underfitting. The shape metric and
MSE weight traces differently, so their conclusions need not coincide.

The neutral-initialization mechanism and subsequent three-seed pilot are recorded
in [the initialization diagnostic](ATLAS-INITIALIZATION-DIAGNOSTIC.md) and
[sign restarts](ATLAS-SIGN-RESTARTS.md). Neither these completed controls nor that
ongoing pilot meets the full specification's biological acceptance requirements.

Receipts:

- Control: [independent fit audit](level0-atlas-longer-fit-audit.json),
  [uncertainty](level0-atlas-longer-fit-uncertainty.json),
  [numerical sensitivity](level0-atlas-longer-preparation-sensitivity.json),
  [training audit](longer-epoch8-training-audit.json).
- Gain: [independent fit audit](level0-atlas-gain-fit-audit.json),
  [uncertainty](level0-atlas-gain-fit-uncertainty.json),
  [numerical sensitivity](level0-atlas-gain-preparation-sensitivity.json),
  [native training diagnostic](gain-selected-training.json),
  [independent training audit](gain-selected-training-audit.json).
- Comparisons: [gain versus control](atlas-gain-versus-level0-atlas-longer-fit.json),
  [gain versus LDS](atlas-gain-versus-connectome-lds-first-fit.json),
  [control versus LDS](atlas-longer-versus-lds.json).
