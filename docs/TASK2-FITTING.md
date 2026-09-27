# Task 2: controls before biological fitting

**Preprocessing qualification:** the source traces are whole-recording z-scores.
These results concern retrospective processed-signal prediction, not an end-to-end
causal forecast. See the [content-bound audit](PREPROCESSING-AUDIT.md).

Controls below established the bar for the subsequent Level 0 fits and
[shared-input comparison](BEHAVIOR-INPUTS.md). Backend optimization remains
deferred. Task 1 now has [processed individual trials and a held-out-neuron
split](RANDI-ATLAS.md), but its fitted model comparison remains untested.

## Controls and metric conventions

`bench-control GRAPH DATA SPLIT PARTITION CONTROL OUTPUT` accepts:

- `history-mean`: mean of finite observations at or before the forecast origin.
- `half-blend`: half that mean, half the last observed value.
- `training-mean`: confidence-weighted per-neuron mean using training animals only.
- `ar`: separate affine AR(1) per neuron, fitted to adjacent, jointly observed
  training samples. Weighted least squares slope is constrained to [-1,1]; the
  intercept is recomputed from the paired means. No validation tuning. Its three
  scalars per neuron include the training mean used before any history observation.

`bench-persist` remains the zero-fit last-observation control. Prefix observations
are allowed at the origin, never after it. Missing AR inputs are predicted forward;
observed prefix samples reset the scalar state. Training-mean and AR controls use
persistence for identities absent from training, and name/count those fallbacks
in the prediction metadata. On this split they are AVBL and URAVR; these cases are
not silently dropped. Thus “training mean” is a mixed control for these identities,
not an oracle constant predictor for every held-out neuron.

The primary horizon score remains the arithmetic mean of per-neuron R², pooling
windows across animals separately for each neuron, using identity confidence as
sample weight. Each denominator uses **held-out target variance at that horizon**.
A training mean therefore need not score exactly zero. Neither pooled R² across
all neuron identities nor averaging animal-level R² is the same metric.

Every animal-split `bench-score` report now includes 2,000 whole-animal bootstrap
replicates (seed 42), percentile 95% intervals, per-animal scores, and a unit-weight
sensitivity score. Each draw retains every window for a sampled animal and
recomputes the target mean and R² by neuron, then averages defined neuron scores.
Repeated animal blocks are merged with weighted parallel moments, not treated as
independent windows. Undefined draws are counted. Model reports use the same
animal draws, but intervals are marginal, **not paired-difference significance tests**.
With three test animals these intervals give weak population uncertainty estimates.
Neuron coverage changes when animals are omitted; this is inherent in the stated
resampled macro metric, not evidence of a fixed-neuron population effect.

## First linear experiment

`linear-fit GRAPH DATA SPLIT MODEL` fits a dense affine VAR(1) in standardized
fluorescence coordinates. This is an observed-coordinate linear state-space model,
not a latent LDS learned with Kalman smoothing/EM. Training means/scales and all
transition coefficients use only training animals. Missing inputs are standardized
zero, missing targets excluded. At inference the model runs through the 10-second
prefix, clamping available observations, then freely predicts 30 seconds. It does
not initialize all 302 Level 0 neurons and does not substitute for that work.

The fixed ridge grid is [0.0001, 0.001, 0.01, 0.1, 1]. Selection maximizes the
mean validation macro R² across the three horizons, with first-candidate tie
breaking. Ridge penalizes the intercept too. The initial experiment has no spectral
stability constraint. All five candidates failed at long validation horizons;
selected ridge 0.001 also failed badly on test. This failure is retained as a
negative experiment, not a useful competitive LDS baseline or a statement that
nonlinear models cannot work. A stable latent LDS is now fitted and scored separately in [LATENT-LDS.md](LATENT-LDS.md).

## Reproduction

Build from a clean committed revision, so the executable records its actual source:

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release
python3 scripts/run_task2_controls.py --receipt runs/wormwideweb-controls-receipt.json
```

The data and animal split are those in [WORMWIDEWEB.md](WORMWIDEWEB.md).
Numerical/causality tests cover recovery of a known linear system, independent
normal equations with missing data, held-out-future invariance, merged moments
versus explicit duplicated observations, and exact two-animal bootstrap support.

History-only full-network initial-state inference is now implemented and audited
in [INITIAL-STATE.md](INITIAL-STATE.md). Explicit sharing and population fitting
are implemented in [LEVEL0-FIT.md](LEVEL0-FIT.md), followed by the
[filtering comparison](LEVEL0-FILTER.md). Biological class/sign annotations remain
missing. The test results already inspected are exploratory evidence; further
model development must not be described as fresh confirmation on these animals.

## Exploratory held-out results

Committed-source [receipt](wormwideweb-controls-receipt.json), same fixed test
animals and exact horizons:

| Model | 1 s R² | 10 s R² | 30 s R² |
| --- | ---: | ---: | ---: |
| Persistence | 0.706 | −0.466 | −0.651 |
| History mean | 0.474 | −0.120 | −0.289 |
| Half blend | 0.723 | −0.128 | −0.331 |
| Training mean, with disclosed fallbacks | 0.001 | −0.034 | −0.039 |
| Per-neuron AR(1) | 0.764 | 0.071 | −0.029 |
| Unconstrained dense linear experiment | 0.696 | −4.191 | −341.921 |

AR(1) marginal 95% animal-bootstrap intervals are [0.693, 0.795] at 1 s,
[−0.027, 0.113] at 10 s, and [−0.055, 0.001] at 30 s. This is not evidence of
reliable positive long-horizon R² across animals. No model here establishes the
project's biological claim. The full receipt includes intervals for every control.

An independent Python calculation using `math.fsum`, unit weights, and the same
126 per-neuron groups reproduces Rust persistence R² as
0.7025157664703914 / −0.4863865227484400 / −0.6711311897799123. Thus the user's
approximate 0.70 / −0.35 / −0.63 table is not reproduced by removing confidence
weights alone. A different preprocessing revision or aggregation convention may
explain it; its exact cause remains unverified. The scorer convention and the
content hashes are retained rather than changed to match the supplied table.

The first two-epoch Level 0 population fit is recorded in [LEVEL0-FIT.md](LEVEL0-FIT.md).
Both trained epochs failed to improve validation, and selection retained the initial
candidate. Validation history reconstruction decays almost completely by the
forecast origin; inference quality, not backend throughput, is the next issue.

A rank-32 stable latent LDS now scores 0.530 / 0.067 / 0.016 on test animals. Both
long-horizon animal-bootstrap intervals include zero. See [LATENT-LDS.md](LATENT-LDS.md)
for all candidate receipts, parameter counts and coverage assumptions.

Filtering now repairs the lost prefix state: origin reconstruction R² rises to
0.947, but the two trained candidates still lose to epoch zero. The selected
filter model scores 0.722 / −0.008 / −0.022 on test, with negative long-horizon
intervals. See [LEVEL0-FILTER.md](LEVEL0-FILTER.md) for the controlled comparison,
selection receipts and timestep audit.

The [connectome-free GRU](GRU.md) is now fitted and scored with 6,831 total scalars,
including normalization. Validation selects epoch 7 of 30; test R² is
0.113 / 0.023 / 0.013. Both long-horizon animal-bootstrap intervals include zero.
The stable LDS has higher point scores at all three horizons. All baseline
receipts remain exploratory, and equal behavior inputs/preprocessing audit are next.
