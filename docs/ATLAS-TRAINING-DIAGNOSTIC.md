# Shared-response training error and amplitude diagnostic

The neutral-prior joint atlas model substantially underfits the training traces.
This is separate from the difficulty of predicting held-out stimulated neurons.
The native [decomposition](atlas-neutral-training-decomposition.json) and
independent [NumPy replay and gain audit](atlas-neutral-training-decomposition-gain-audit.json)
use all 2,842 training trials across 161 stimulation targets, with the original
confidence weights. No test responses enter this diagnostic.

| Training quantity | MSE |
| --- | ---: |
| Zero response | 0.07547002 |
| Fitted neutral-prior joint model | 0.07535820 |
| Empirical mean response for each training target | 0.06300051 |
| Mean-response bound additionally constrained to start at zero | 0.06315307 |

Weighted variance decomposition gives zero-response MSE = within-target trial
variance + energy of the mean traces. The latter is 0.01246952. The fitted model
captures only **0.897%** of that available reduction in training MSE. These means
are an empirical bound for a shared deterministic target response, not a
deployable held-out-target baseline. Within-target variation is not necessarily
biological noise: omitted trial state, animal context and inputs may explain it.
The model's dynamics can further restrict which mean responses are attainable.

The independent audit replays all 161 training target grids and verifies the
native scores within 1e-10. It also fits one nonnegative global observation gain
by training-only least squares, keeping all neural dynamics and input parameters
frozen: gain = max(0, sum(w*y*prediction) / sum(w*prediction²)).

The resulting gain is **9.8391**. Training MSE improves to **0.07489045**;
validation MSE improves from **0.04232251 to 0.04164451**. Even with this gain,
only about 4.65% of the mean-trace energy is captured. Amplitude calibration helps
but does not account for most of the underfitting. This is a validation diagnostic,
not a replacement model or a new test-set result. It adds one fitted parameter,
does not refit the joint classifier, and cannot repair waveform shape.

The native diagnostic was built from `e36e101`; the independent gain audit uses
the script committed in `bf15fbf`. Both receipts retain model and data hashes.
The model itself comes from `fe2972f`. Reproduce with:

```sh
target/release/examples/diagnose_atlas_training data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json runs/level0-atlas-classification-fit/selected.json runs/training-decomposition-new.json
python3 scripts/audit_atlas_training.py --model runs/level0-atlas-classification-fit/selected.json --native runs/training-decomposition-new.json --check-global-gain --output runs/training-gain-audit-new.json
```

Next fitting work should address observation amplitude and optimization duration
under training/validation controls. Five Adam updates are not evidence of
convergence. Neither this diagnostic nor a failed linear baseline establishes a
ceiling on what a nonlinear model can predict.

See the [initialization diagnostic](ATLAS-INITIALIZATION-DIAGNOSTIC.md) for
independently checked epoch-8 amplitude/shape scores and the zero chemical
driving-force mechanism at the original neutral initialization.
