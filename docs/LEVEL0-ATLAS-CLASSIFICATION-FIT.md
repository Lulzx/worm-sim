# Joint atlas trace and detection fitting

The Rust atlas fitter now accepts a pair-evidence file and an optional
`classification` configuration. With both present, it minimizes original
confidence-weighted training-trial MSE plus `weight` times mean ordered-pair
Bernoulli cross entropy, plus the existing parameter/kernel priors. Each training
pair contributes once, independently of its number of stimulation trials. Labels
are published q < 0.05 detections; a negative means not detected, not an absent
connection. Validation/test pair labels never enter fitting or initialization.

The classifier uses area = sample_dt * sum(sqrt(response² + epsilon²) - epsilon),
feature = log1p(area / area_scale), and logit = bias + softplus(raw_slope) * feature.
Bias starts at the Jeffreys-smoothed training prevalence logit, and the effective
slope starts at one. Both are learned jointly with network parameters and the
positive shared current kernel. The positive slope cannot reverse a poor ranking.
`area_scale` is a declared reference unit, not an empirical detection threshold;
`epsilon` smooths the absolute value. Neither represents measurement noise.

A single discrete adjoint propagates both losses through the fluorescence
readout, shared current, response baseline and unforced preparation. No detached
predictions or artificial MSE targets are used. Trace sufficient statistics still
preserve the original trial MSE; irreducible trial variance is reported in that
MSE, though it has no gradient. Epoch receipts report preceding training MSE,
unweighted mean pair BCE and the prior penalty separately. The total training
objective is MSE + classification.weight * BCE + penalty.

## Frozen first-run protocol

`configs/level0-atlas-classification-fit.json` specifies five full-batch Adam
updates, dt 0.01 s, 60 s unforced preparation, classification weight 0.1,
area_scale 0.01 and epsilon 1e-8. Other settings match the preceding atlas fits.
These are an initial declared configuration, not a hyperparameter optimum.
Checkpoint selection remains minimum validation trace MSE, including epoch zero,
with earlier exact ties retained. Validation classification labels are not used
for selection. All candidates and the two classifier parameters are persisted.
The supplied full evidence hash is recorded for provenance, including held-out
labels that do not participate in gradients.

The common evaluator continues to rank the original absolute fluorescence area,
not classifier probability or the smoothed training surrogate. This preserves
comparability with the frozen LDS and earlier nonlinear results. Validate
preparation duration and integration step sensitivity of AUROC as well as MSE;
nearly zero traces previously made correlation and ranks numerically sensitive.
No fresh confirmatory holdout remains: the existing validation/test targets have
already been inspected in earlier exploratory comparisons. Bootstrap intervals
condition on the fitted model and do not cover training uncertainty.

Run after committing and source-stamping the executable:

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release --example fit_level0_atlas
target/release/examples/fit_level0_atlas data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json configs/level0-atlas-classification-fit.json runs/level0-atlas-classification-fit runs/randi-pairs.json
```

Verification includes finite differences for the classifier and its full neural
adjoint (all raw parameters, initial state, readout and held current), exact MSE
adjoint parity, original-trial MSE normalization, pair-count normalization,
held-out-label mutation through every optimization checkpoint, invalid lineage,
and unchanged legacy MSE-only fitting. These establish implementation properties;
biological improvement must be established by the actual run and common scorer.
