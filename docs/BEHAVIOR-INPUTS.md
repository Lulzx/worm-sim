# Shared behavior inputs for Task 2

This protocol supplies identical behavior information to fitted model families:
observed exported channels during the ten-second history, then a common
training-fitted autoregression through the free forecast. Actual future behavior
is excluded even during neural-model training. The source remains
[retrospectively processed](PREPROCESSING-AUDIT.md); a history cutoff does not
remove upstream normalization/filtering dependencies.

## Protocol

The fixed channel list is `angular_velocity`, `head_angle`, `pumping`, `velocity`,
in lexical order. Each channel is standardized by its mean and population standard
deviation across training windows only. Missing values are excluded; constant
channels use scale one. A separate affine AR(1) is fitted to adjacent jointly
observed training pairs, never across window boundaries. Its slope is constrained
to [−0.995, 0.995], then the intercept is recomputed from the paired means. These
choices are fixed before the behavior-assisted fit, not selected on test animals.

At a window's first frame, an absent channel starts at its training mean. Each
subsequent frame propagates the AR estimate. At/before the origin, an available
measurement replaces that estimate. After the origin no measurement is read.
A fully absent channel therefore has defined inputs without silently removing
its window. Missing observations have mask zero; observed behavior has mask one.
There is no calibrated behavior-confidence field in the source schema.

The eight-dimensional vector is `[four standardized values, four observed masks]`.
Neural transitions from frame t to t+1 consume `u[t]`. All masks after the origin
are zero. This is a point-estimate exogenous trajectory: uncertainty in future
behavior is not marginalized, and errors from that approximation must be retained.
It does not claim that velocity/head angle/pumping are independent sensory causes.
Behavior may be downstream of the neural activity being modeled.

Four fitted scalars per channel (mean, scale, slope, intercept) add **16 scalars**
to each model's standalone accounting. The common artifact records training trial
IDs, sample interval, dataset/split/graph hashes and source revision. Its hash is
included in neural prediction metadata, allowing identical common inputs to be
verified across model families. Fitting the same protocol on the same training
cohort is deterministic. No model may fit behavior dynamics to held-out animals.

## Integration status

- **GRU implemented:** eight covariates join neural values/masks in the recurrent
  cell. Their weights are learned by full-window BPTT. Behavior AR coefficients
  stay fixed; gradients do not pass into observed behavior or the shared AR fit.
  Legacy zero-covariate artifacts retain their numerical predictions.
- **LDS implemented:** the same vectors enter a learned transition input matrix,
  including controlled Kalman inference and EM sufficient statistics. See
  [LDS details](LATENT-LDS.md#behavior-input-extension).
- **Level 0 implemented:** the same vectors enter signed membrane-current weights
  shared according to the neuron rest-parameter groups. Currents are held constant
  between behavior sample times during both history inference and free prediction.
  Conditional reverse-mode gradients include input weights and the input-dependent
  membrane time-constant term. They hold the inferred origin state fixed.

The first equal-input comparison is complete; see [held-out results](#held-out-comparison).
It does not establish long-horizon forecasting skill for Level 0.

With six hidden units and 149 training identities, the driven GRU has 6,677 neural
weights/biases, 298 neural normalization statistics and 16 behavior scalars:
**6,991 total**, versus 6,831 in the unconditioned GRU. The selection report keeps
behavior scalars separate to avoid double-counting them as neural calibration.
The extra capacity and changed inputs must both be disclosed in comparisons.

## Checks and reproduction

Tests compare behavior coefficients with independent normal equations, ensure
no cross-window pair is used, check masks and AR rollout, and demonstrate that
replacing post-origin behavior—even with invalid numeric values in the low-level
input test—cannot affect generated covariates. Full refits with changed test neural
and behavior futures leave coefficients, selection and predictions unchanged.
All driven GRU parameter gradients are checked by finite differences.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example audit_behavior_inputs --bin wormsim
target/release/examples/audit_behavior_inputs data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  runs/behavior-input-audit.json
```

The [declared GRU configuration](../configs/gru-behavior-fit.json) retains the
same hidden size, 30 epochs, optimizer settings and seed as its unconditioned
experiment. Candidate selection remains mean validation R² at 1/10/30 seconds.
The protocol audit does not evaluate neural forecasting quality.

## Real-data protocol audit

The [receipt](behavior-input-audit.json), from source
`22da047`, covers all **504 windows** (360 train, 72 validation, 72 test).
Every generated input vector is finite, every post-origin observation mask is zero,
and replacing all exported future behavior leaves all eight covariates exactly
unchanged. The receipt includes the 16 fitted behavior scalars and all source/split
identifiers. This is evidence about information access, not forecast skill.

The same committed executable loaded the archived no-behavior GRU artifact and
reproduced all 72 saved test trial prediction arrays exactly. Source/model metadata
changes as expected; neural numerical outputs are unchanged. The real-data Level 0 fit and comparison are recorded below.

## Baseline fits: validation only

The [selection receipt](behavior-baseline-selection.json) binds both baseline fits
to committed source `ad1b06d41911e50547b420804bf9d213a4536746`, records all candidates,
and verifies that the embedded common behavior artifacts are **exactly equal**.
No behavior-assisted test predictions were generated during that selection-only
experiment. Subsequent test scoring is recorded below.

| Model selected on validation | 1 s R² | 10 s R² | 30 s R² | Total scalars |
| --- | ---: | ---: | ---: | ---: |
| LDS rank 32, EM update 2 | 0.49642 | 0.09819 | 0.04878 | 7,567 |
| GRU six hidden units, epoch 11 | 0.16135 | 0.04424 | 0.05657 | 6,991 |

These scores select the artifacts; they are not independent test estimates or
proof of a benefit from behavior. The GRU's extra covariate weights also change
initialization and capacity. A descriptive comparison against its no-behavior run
cannot isolate those effects. LDS adds 256 learned input weights and the same 16
behavior scalars. Both fitted models still lack uncertainty propagation through
the estimated future behavior trajectory.

The summed LDS preparation/candidate timers are 80.77 seconds; GRU candidate
timers sum to 34.54 seconds on the M4 Pro CPU. Both exclude file loading, candidate
writes and separate final scoring; GRU excludes preparation, while LDS includes
preparation but excludes per-rank PCA initialization. These differing boundaries
must not be treated as an apples-to-apples performance comparison.

The receipt also verifies exact numerical compatibility on all 72 archived
no-behavior LDS test predictions. The subsequent Level 0 fit and three-model test comparison are recorded below.

```sh
target/release/wormsim lds-fit data/c302-herm.wsc   runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json   configs/lds-behavior-fit.json runs/lds-behavior-fit.json
target/release/wormsim gru-fit data/c302-herm.wsc   runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json   configs/gru-behavior-fit.json runs/gru-behavior-fit.json
python3 scripts/record_behavior_baseline_fits.py
```

## Level 0 input extension

The declared [configuration](../configs/level0-behavior-fit.json) retains the
unconditioned filter experiment's two epochs, learning rate, initialization,
block-EKF settings and validation criterion. Each input row starts at zero and
receives a zero-centered mean-square prior with strength 0.01. The c302 graph's
203 neuron sharing groups produce 1,624 signed weights for eight features.
These learned currents are an empirical coupling, not a source-backed sensory
projection. Missing class/transmitter annotations and assumed L/R sharing remain
limitations.

The standalone count is 8,680 fitted parameters including the 16 common behavior
scalars, plus 298 neural calibration statistics: **8,978 scalars including
calibration**. The 906 inferred state variables per trial are reported separately.
Behavior coefficients remain fixed during neural fitting; gradients do not pass
through the behavior fit or the history-state inference algorithm.

Finite-difference tests cover every raw dynamics parameter, initial-state
component and sample-current component on a coupled synthetic graph. An independent
event-driven solver agrees with the driven forecast. Both shooting and block-EKF
inference ignore post-origin currents. Tied-current reduction is separately checked
against finite differences. A full synthetic refit with altered test neural and
behavior futures preserves every candidate's weights, selection and predictions.
Legacy artifacts without input fields remain readable and numerically compatible.

## Held-out comparison

The [comparison receipt](behavior-comparison.json) binds all three selected models,
their [LDS](lds-behavior-receipt.json), [GRU](gru-behavior-receipt.json) and
[Level 0](level0-behavior-receipt.json) scoring receipts, and the common behavior
parameters. Their behavior coefficients, calibration, channel order, sample
interval and dataset/training lineage are exactly equal. Training source revisions
differ (`ad1b06d` for the baselines, `7282bdd` for Level 0); those metadata fields
are preserved rather than claiming equal full-artifact hashes. All three call the
same Rust input generator, whose source hash is recorded.

**Retrospective benchmark only.** Scores are confidence-weighted macro-neuron R²
on the same 72 windows from three test animals. Brackets contain percentile 95%
intervals from 2,000 whole-animal bootstrap draws with seed 42. Every draw has a
defined score; three animals nevertheless give little uncertainty resolution.

| Selected behavior-assisted model | 1 s R² [95% interval] | 10 s R² [95% interval] | 30 s R² [95% interval] |
| --- | --- | --- | --- |
| LDS rank 32, update 2 | 0.532 [0.503, 0.550] | 0.057 [−0.201, 0.124] | 0.014 [−0.026, 0.045] |
| GRU six hidden units, epoch 11 | 0.123 [0.070, 0.162] | 0.031 [−0.030, 0.141] | −0.019 [−0.098, 0.060] |
| Level 0, epoch 2 | 0.749 [0.695, 0.761] | −0.068 [−0.142, −0.020] | −0.097 [−0.144, −0.080] |

Level 0 **fails the positive long-horizon forecast hurdle**: both intervals are
below zero. The LDS long-horizon point estimates are positive but both intervals
include zero. These marginal intervals do not measure significance of pairwise
model differences. Test animals have been inspected during earlier experiments,
so this is an exploratory result, not a fresh confirmatory evaluation.

For orientation, the previously scored no-behavior models gave 1/10/30-second R²:
LDS 0.530/0.067/0.016, GRU 0.113/0.023/0.013, and filter-based Level 0
0.722/−0.008/−0.022. The per-neuron AR control gave 0.764/0.071/−0.029.
Behavior assistance therefore provides no consistent descriptive long-horizon
improvement. This comparison does not isolate the effect of behavior from extra
capacity, changed initialization, or a differently selected epoch.

### Level 0 selection and numerical check

Both epochs used all 360 training windows. Validation selected epoch 2 by the
predeclared mean of the three horizons, including the epoch-zero candidate:

| Epoch | 1 s validation R² | 10 s | 30 s | Mean criterion |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.646283 | −0.001361 | −0.011423 | 0.211166 |
| 1 | 0.662788 | 0.013260 | −0.045250 | 0.210266 |
| 2 | 0.669626 | 0.026144 | −0.049639 | 0.215377 |

This is the first declared Level 0 experiment here to select trained weights,
but the mean validation improvement is only 0.00421, and its 30-second validation
score worsens. Origin reconstruction remains high (0.94497); good reconstruction
does not imply useful free prediction. The M4 Pro CPU candidate timers sum to
176.58 seconds, including validation and excluding loading, initialization,
checkpoint writes and separate scoring. No backend change was needed.

With the selected weights fixed, halving the timestep from 0.01 to 0.005 seconds
changes validation R² by +0.000049/+0.000103/+0.0000007. This is a numerical
sensitivity check, not another trained candidate or a test-based timestep choice.
The current executable also reproduces all 72 archived no-behavior Level 0 test
prediction arrays exactly.

A useful fitted biological model remains outstanding. Source-backed class and
transmitter priors, conditional-state training limitations, mismatch between
fluorescence and latent dynamics, and the prospective-processing requirement
remain unresolved. Additional GPU/backend optimization cannot address these
scientific failures.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --features hdf5 --bin wormsim
target/release/wormsim level0-fit data/c302-herm.wsc runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json configs/level0-behavior-fit.json runs/level0-behavior-fit.json
python3 scripts/score_level0_fit.py --model runs/level0-behavior-fit.json --prefix runs/level0-behavior --receipt runs/level0-behavior-receipt.json
python3 scripts/score_latent_lds.py --model runs/lds-behavior-fit.json --prefix runs/lds-behavior --receipt runs/lds-behavior-receipt.json
python3 scripts/score_gru.py --model runs/gru-behavior-fit.json --prefix runs/gru-behavior --receipt runs/gru-behavior-receipt.json
target/release/wormsim level0-predict data/c302-herm.wsc runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json runs/level0-filter-fit.json test runs/level0-filter-compat-predictions.json
python3 scripts/record_behavior_comparison.py
```
