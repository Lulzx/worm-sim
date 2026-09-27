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
positive shared current kernel. Calcium scale and readout gain remain fixed at
one, with zero readout offset. The positive slope cannot reverse a poor ranking.
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


## First completed fit: no Task 1 superiority

Source `fe2972f1a7bbb105534b6a0eb9b7671e1f81ee26` completed the frozen
five-update protocol with **6,783 trainable parameters**, selecting epoch 5 by
validation MSE. The summed epoch timers were 747.16 seconds, including gradient
and validation work but excluding loading, aggregation, checkpoint serialization
and final evaluation. A release build overlapped part of the first update; these
timings are run diagnostics, not an isolated throughput benchmark.

The [independent audit](level0-atlas-classification-fit-audit.json),
[cluster intervals](level0-atlas-classification-fit-uncertainty.json), and
[paired LDS comparison](atlas-classification-comparison.json) retain the source,
data/split/evidence identities, checkpoint and prediction hashes, and settings.

| Test metric | Joint Level 0 | Constrained LDS |
| --- | ---: | ---: |
| Pooled trace MSE | 0.04989685 | 0.04734484 |
| Macro trace correlation | 0.0021254 | 0.0452857 |
| Pair AUROC | 0.681006 | 0.686187 |

For Level 0 minus LDS, the stimulated-target paired bootstrap AUROC difference
is **−0.005181**, 95% interval **[−0.037673, +0.032125]** across 13 eligible
pair-label targets. This does not establish superiority or equivalence.
The pooled MSE difference is **+0.0025520**, with target-cluster interval
**[+0.0011339, +0.0037112]** across 15 targets and recording-cluster interval
**[+0.0019738, +0.0032052]** across 67 recording IDs. Both favor the LDS.
All 12,588 test trace correlations are defined for both models. Separate target
and recording resampling does not account for crossed dependence jointly, and
recording IDs are not verified animal identities. All 2,000 draws were defined;
intervals condition on these frozen fits and the previously inspected cohort.

The independently calculated classifier probabilities give test mean BCE
**0.187410** and Brier score **0.0442944**, versus **0.188609** and **0.0444772**
for a constant probability equal to the smoothed training prevalence (0.049637).
These are small point improvements, without a calibration bootstrap or a claim
of statistical significance. They do not change the unchanged raw-area AUROC
comparison. The learned bias is −2.9909273 and raw slope 0.5915203. Validation
MSE is 0.0423225053; validation pair AUROC is 0.7630441.

## Numerical and independent checks

Independent NumPy Euler replay of all 30 selected validation/test target grids
matches saved fluorescence to **2.23e−16** maximum absolute error. This includes
parameter transforms, preparation, drive and baseline subtraction, but does not
independently repeat optimization. The audit also recalculates saved trace scores,
response areas, direct pairwise AUROC, classifier BCE/Brier, and training-count
classifier initialization. Rust gradient/isolation tests and the Python audit
arithmetic tests are separate checks.

The selected prepared state's unforced derivative L2 norm is **2.19e−13**.
Doubling preparation from 60 to 120 seconds changes validation predictions by at
most **1.24e−13**. Halving dt changes them by at most **5.32e−5** and changes
validation MSE by **2.29e−8**. The full
[duration/step sensitivity receipt](level0-atlas-classification-preparation-sensitivity.json)
checks 60, 120 and 240 seconds, plus dt 0.005 at 240 seconds. Validation AUROC
varies by at most **4.97e−5** over these settings. Correlation and defined-trace
counts remain sensitive near numerically constant predictions: defined validation
correlations range from 11,407 to 11,730. No diagnostic replaces frozen test
predictions or changes the predeclared selection rule.

This is an initial five-update fit, not evidence of optimization convergence.
The 30-second preparation in the preceding MSE-only experiment differs from this
protocol, so their results are not a controlled classification-loss ablation.
Calcium scale remains fixed, and the graph still has neutral signs and no
source-backed molecular annotations. The next model work is informed by the
[sign-prior source audit](SIGN-PRIOR-SOURCE-AUDIT.md); the current result does not
satisfy the specification's Task 1 baseline-superiority requirement.
