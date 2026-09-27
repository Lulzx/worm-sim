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
- **Level 0 pending:** the same vectors must enter tied current-input weights
  during history inference and free prediction, with matching parameter gradients.

The equal-input comparison is incomplete until Level 0 is integrated and all three models are fitted
and scored. A GRU-only result must not be presented as that comparison.

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
changes as expected; neural numerical outputs are unchanged. Level 0
behavior integration and the equal-input model comparison remain pending.

## Baseline fits: validation only

The [selection receipt](behavior-baseline-selection.json) binds both baseline fits
to committed source `ad1b06d41911e50547b420804bf9d213a4536746`, records all candidates,
and verifies that the embedded common behavior artifacts are **exactly equal**.
No behavior-assisted test predictions were generated for this experiment.

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
no-behavior LDS test predictions. Level 0 input weights/inference/gradients and
the three-model test comparison remain outstanding.

```sh
target/release/wormsim lds-fit data/c302-herm.wsc   runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json   configs/lds-behavior-fit.json runs/lds-behavior-fit.json
target/release/wormsim gru-fit data/c302-herm.wsc   runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json   configs/gru-behavior-fit.json runs/gru-behavior-fit.json
python3 scripts/record_behavior_baseline_fits.py
```
