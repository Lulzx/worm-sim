# Stable latent LDS baseline

**Preprocessing qualification:** the source traces are whole-recording z-scores.
These results concern retrospective processed-signal prediction, not an end-to-end
causal forecast. See the [content-bound audit](PREPROCESSING-AUDIT.md).

This replaces neither the retained failed dense VAR experiment nor the Level 0
model. It supplies the missing latent linear dynamical baseline for Task 2:

`x[t+1] = A x[t] + process_noise`

`standardized_fluorescence[t] = C x[t] + observation_noise`

The latent dimension is independent of the number of recorded neurons. Full
process and initial covariances are fitted, with diagonal observation covariance.
Prediction starts with zero latent mean and the fitted initial covariance,
assimilates each available sample in the ten-second history using Kalman updates,
and then forecasts freely. Missing observations are skipped, not filled with zero
as targets. Future observations never enter the prediction filter.

The Kalman/RTS and moment-estimation basis is described by
[Ghahramani and Hinton, 1996](https://mlg.eng.cam.ac.uk/zoubin/papers/tr-96-2.pdf).
Our constrained variant and data conventions are specified below; this is not a
claim that the paper endorses these worm-specific choices.

## Training and selection

All 360 training windows from 15 animals are used. Per-neuron means and scales
are fitted to confidence-weighted training observations. A deterministic PCA
initialization uses a covariance of mean-imputed standardized training samples,
weighted by square-root confidence. That imputation is confined to initialization;
Kalman inference and EM statistics use masks for missing observations.

Each training sequence undergoes forward filtering and backward RTS smoothing.
Moment updates fit A, C, full Q and initial covariance, plus diagonal R. C and R
updates include only observed targets. Confidence enters as effective measurement
variance `R[i] / confidence`; this is a weighting convention, not a calibrated
probabilistic model of neuron identity errors. Accordingly, R's M-step divides
weighted expected squared residuals by the **number** of positive-confidence
observations, not by the sum of their weights.

A is rescaled when its Euclidean operator norm exceeds 0.995 per 0.5-second step.
The norm bound uses a Jacobi diagonalization of AᵀA followed by a Gershgorin bound
on the residual matrix, with a small numerical margin. This is stronger than
requiring eigenvalues inside the unit circle and rules out transient norm growth.
It may restrict useful nonnormal dynamics in the current latent coordinates.

The declared first grid is ranks **4, 8, 16, 32**, each with its initialization and
eight updates. Rank 32 was added before any real-data fit or candidate scoring to
provide a capacity close to Level 0: 7,295 LDS scalars versus the declared 7,040
Level 0 capacity. Lower ranks have fewer parameters. Selection maximizes the mean
validation macro-neuron R² at 1/10/30 seconds, with earlier candidates winning ties.
Test animals are not used for candidate selection.

Normalized moment solves use ridge 0.0001. Q and initial covariance receive 1e-6
identity jitter; diagonal R is floored at 1e-4. Stability projection, ridge and
variance floors mean the update is a **constrained EM-style procedure**, with no
guarantee of likelihood monotonicity. Candidate reports retain the training NLL
measured before that update and the resulting validation scores. A numerical
failure stops the run rather than dropping a training sequence.

## Coverage and counts

The model covers the 149 identities observed in training. AVBL and URAVR are
present in test but absent from training; those output traces use last-observation
persistence, explicitly named/countable in prediction metadata, as in the prior
dense baseline. They are not silently excluded. This differs from Level 0's
assumed readout for training-unseen neurons and must be considered when comparing
long-horizon aggregate scores.

The parameter count is `k² + N*k + k*(k+1) + 3*N`: transition, loading, the two
symmetric covariances, diagonal noise, and training means/scales. There is no
connectome, behavior input, latent bias, learned initial mean or per-test-animal
parameter fit. Inference adapts the k-dimensional posterior state to the observed
history, not to future targets.

## Reproduction and checks

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release
./target/release/wormsim lds-fit data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  configs/lds-first-fit.json runs/latent-lds.json
./target/release/wormsim lds-predict data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  runs/latent-lds.json test runs/latent-lds-predictions.json
./target/release/wormsim bench-score data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  runs/latent-lds-predictions.json test runs/latent-lds-report.json
```

Tests include a hand-derived two-time Gaussian posterior, a multivariate NumPy
joint-conditioning oracle independent of Kalman/RTS recursions, masked fitting,
stability checks, and a complete refit after replacing test futures. The last
check requires identical fitted matrices, selected rank/iteration and predictions.
A stable fitted baseline is not evidence that the biological model beats it, or
that the long-horizon success criterion is achieved.

## First real-data result

The [committed-source receipt](latent-lds-receipt.json) records all 36 candidates.
Validation selects rank **32**, update **3**, with **7,295** parameter scalars.
Its transition norm bound is 0.9950000000000001 (floating-point rounding at the
configured cap). Training likelihood continues improving at later updates while
validation forecast scores decline; the selected candidate is retained unchanged.

| Partition | 1 s R² | 10 s R² | 30 s R² |
| --- | ---: | ---: | ---: |
| Validation | 0.496 | 0.106 | 0.043 |
| Test | 0.530 | 0.067 | 0.016 |
| Test animal-bootstrap 95% interval | [0.489, 0.548] | [−0.205, 0.150] | [−0.028, 0.066] |

Both long-horizon intervals include zero. The positive test point estimates do
not establish population-level long-horizon forecasting. AR(1)'s test point
scores are 0.764 / 0.071 / −0.029: it remains stronger at one second. The apparent
30-second difference is not a paired-bootstrap significance result. Neither this
baseline nor the failed biological fit completes Task 2's success criterion.

Validation prefix reconstruction is 0.603 at the first frame and **0.596 at the
forecast origin**. The previous Level 0 shooting-inference experiment fell from
0.645 to 0.0005. This supports testing a filtering-based Level 0 state estimate,
but does not isolate the inference algorithm as the cause: the two models and
fitted parameter sets differ.

The summed timed preparation/candidate phases took 97.65 seconds on this M4 Pro
CPU. That excludes file loading, per-rank PCA initialization, artifact writing and
final held-out scoring; it is not an end-to-end throughput claim.

Reproduce the compact scored receipt and prefix diagnostic after fitting with:

```sh
python3 scripts/score_latent_lds.py --receipt runs/latent-lds-receipt.json
```

Only selected-model test predictions are evaluated. Source preprocessing
causality/units, biological class/sign annotations, equal behavior inputs, and successful Level 0
forecasting remain open work.

The matched-budget [GRU comparator](GRU.md) is now fitted and scored. Its long-horizon
point estimates are also positive, with intervals spanning zero.

## Behavior input extension

The optional [shared behavior protocol](BEHAVIOR-INPUTS.md) drives
`x[t+1] = A x[t] + B u[t] + noise`, retaining `y[t] = C x[t] + noise`.
The same input vectors enter both Kalman filtering and free forecast transitions.
The smoother includes their effect in the predicted means; covariance propagation
and RTS gains remain conditional on the fixed input trajectory. Actual future
behavior is never supplied, including during training. Input-forecast uncertainty
is not marginalized.

The M-step forms moments of `[x[t], u[t]]`, solves the ridge-regularized joint
regression, and projects A to the same operator-norm bound. With projected A fixed,
it refits B conditionally from the input and state–input moments. The process
covariance is the full expected transition residual covariance using that final
A and B. Observation/noise and initial-covariance updates retain their original
rules. This is an EM-style constrained update, not a guarantee of monotonic
likelihood under projection/ridge/floors.

An independent [NumPy joint-Gaussian oracle](../scripts/generate_lds_control_fixture.py)
conditions the entire trajectory directly, without a Kalman recursion. Tests
match its smoothed means, marginal/lag covariances, likelihood and every parameter
of one M-step. The fixture activates the stability projection, testing the
conditional B refit and final process covariance. A full refit with altered test
neural/behavior futures preserves parameters, selection and predictions. Legacy
artifacts default to zero input dimensions and preserve predictions.

The [fixed driven configuration](../configs/lds-behavior-fit.json) retains ranks
4/8/16/32, eight updates, ridge 1e-4 and transition cap 0.995. B starts at zero;
only training-animal EM updates estimate it. Four behavior channels supply eight
values/masks, adding `8 * rank` learned B entries plus 16 shared behavior scalars.
A rank-32 driven candidate therefore contains 7,567 total scalars. The report
states these extra counts separately; the unconditioned receipts above remain
unchanged. The behavior-assisted comparison will be scored after Level 0 consumes
the same protocol.

The first controlled LDS run selects rank 32, update 2 on validation. Its
validation-only results and exact shared-input comparison with GRU are recorded
in [BEHAVIOR-INPUTS.md](BEHAVIOR-INPUTS.md#baseline-fits-validation-only). No new
behavior-assisted test claim is made before the Level 0 integration.
