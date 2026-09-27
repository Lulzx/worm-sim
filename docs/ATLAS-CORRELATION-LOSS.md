# Optional correlation loss for atlas pair-mean responses

The Rust atlas fitter accepts an optional `correlation` object with positive
`weight` and `epsilon`. It adds a shape-sensitive objective to the existing
confidence-weighted original-trial MSE, optional pair-detection BCE and priors.
It introduces no trainable parameters. Existing configurations leave it disabled.
The longer unit-gain and learned-gain experiments remain unchanged and do not
use this loss.

For each training stimulation target, the existing sufficient-statistic builder
forms a confidence-weighted mean trace for each observed responding neuron.
Each eligible target/responding-neuron pair contributes once, including observed
self-responses. A pair is eligible when its target mean has at least two samples
and is nonconstant. Eligibility never depends on the model's prediction. A flat
prediction still has a defined objective and generally a nonzero gradient.

Over observed samples, let p and y be the prediction and training mean trace.
Using population moments, the stabilized correlation is

```text
r = covariance(p, y) / sqrt((variance(p) + epsilon²) * (variance(y) + epsilon²))
L_shape = mean_over_eligible_pairs(1 - r)
L_total = original_trial_MSE + correlation.weight * L_shape
          + classification.weight * pair_BCE + prior_penalties
```

The BCE term is present only when configured. Epsilon has fluorescence
standard-deviation units and is a declared stabilization assumption, not a
measured noise level. It prevents undefined gradients at flat predictions, but
also means this is not exactly scale-invariant Pearson correlation. Centering
makes it invariant to a constant fluorescence offset. Constant target means
and zero-confidence traces are excluded; a correlation-enabled fit fails if no
eligible training pairs remain. Epoch reports record both the eligible pair
count and the unweighted mean shape loss preceding each update.

This objective compares the model with **training pair means**. It is not mean
correlation over individual trials, and does not replace the separately reported
test trace correlations. Confidence weights determine the target means; eligible
pairs then receive equal weight, independent of their trial count. The standalone
loss masks missing samples, while the current atlas aggregation still requires
complete positive-confidence traces. No claim of missing-data support for the
whole population fitter follows from the loss's masking support.

The fluorescence cotangent combines MSE, BCE and correlation before one reverse
pass through neural dynamics, preparation, input current and observation gain.
Selection remains minimum validation **MSE**, including epoch zero and earlier
ties. Validation/test observations do not enter the shape targets or gradients.

Tests verify every fluorescence derivative with missing samples; exclusion of
constant/zero-confidence targets; finite gradients at flat predictions; offset
invariance; invalid shapes and variance floors; all neural-parameter derivatives
and the shared log-gain derivative through the prepared model; and unchanged
checkpoints after held-out fluorescence or pair-label mutations. Integration
tests independently calculate the recorded training shape loss. These establish
implementation correctness, not a biological improvement. No real-data
correlation-loss fit has been run yet; choose its weight and stabilization only
under an explicitly recorded training/validation protocol.

## Independent training audit

`diagnose_atlas_training` reports the training shape loss and eligible pair count
when the model config enables it. An optional final epsilon argument evaluates
the same diagnostic on an existing MSE fit without changing that model. The
receipt identifies whether the epsilon was configured for fitting. Predictions
are merged across trials to cover all neurons in each target's aggregate, with
duplicate-column equality checked.

```sh
target/release/examples/diagnose_atlas_training data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json runs/level0-atlas-classification-fit/selected.json runs/neutral-shape-diagnostic.json 0.01
python3 scripts/audit_atlas_training.py --model runs/level0-atlas-classification-fit/selected.json --native runs/neutral-shape-diagnostic.json --output runs/neutral-shape-audit.json
```

The independent auditor reconstructs confidence-weighted means directly from
original training trials, replays the neural model in NumPy, calculates centered
covariance independently and checks the native count and loss within 1e-10.
It does not call the Rust correlation implementation or refit the model. Analytic
Python checks cover matching, reversed, shifted and constant signals and confirm
the effect of the variance floor. These diagnostic values are distinct from
held-out performance and do not justify selecting epsilon using test outcomes.
