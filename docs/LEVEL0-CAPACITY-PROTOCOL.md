# Level 0 training capacity protocol

Feature development is paused. The immediate question is whether the existing
Level 0 dynamics can fit training responses, before another held-out comparison.
The original five-update result is not a convergence result. The later 25-update
controls and three sign restarts also remain substantially underfit.

## First experiment: two training targets, 300 updates

Use ADAL and ADAR, the first two targets in the authoritative sorted training
export, comprising 50 trials. This is a deliberately small, related target pair;
success would need replication on unrelated training targets. No target is chosen
by validation or test error. Use seed 1's saved epoch-zero random-sign model,
without selecting between seeds. Its signs are optimization initializations, not
biological polarity assignments.

- Set normalized resting voltages and the voltage preparation seed to −0.2.
  Retain the full parameter-dependent 60-second unforced preparation.
- Retain the 302-neuron connectome, original ties, 0.01-second Euler grid, shared
  stimulus kernel, and calcium dynamics. Disable all optional dynamical extensions.
- Fit positive per-neuron observation gains, initially 10, using log coordinates.
  These replace the global gain; freeze global gain and native calcium scales.
  Neurons absent from both target observations have zero gain data gradients.
- Optimize confidence-weighted trace MSE with Adam at 0.01 for 300 updates.
  Disable classification, correlation, weight decay, and all priors for this
  capacity test. These settings do not define a production benchmark fit.
- Record every iterate's MSE, gradient norm, gain range, and elapsed time. Save
  checkpoints every 50 updates and the final iterate. Abort on nonfinite values.
- Compute the within-target empirical mean-response bound, plus the stricter
  bound requiring response at time zero to be zero. Report
  `(zero-response MSE − model MSE) / (zero-response MSE − zero-start bound)`.
  The provisional capacity gate is at least 90% at the final iterate.

The bounds concern shared deterministic responses to a target. They are neither
biological noise estimates nor attainable guarantees for these dynamics.
Selection uses training MSE only. The best-iterate metric is recorded, but only
scheduled checkpoints are saved; it is not a claim that every best iterate is
recoverable. Diagnostic envelopes explicitly name their subset and are not
accepted as ordinary full-training benchmark checkpoints.

## Execution

After building the Rust examples and installing the pinned JAX environment:

```sh
target/release/examples/export_atlas_training \
  data/c302-herm.wsc runs/randi-data.json data/randi-neuron-split.json \
  runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  runs/overfit-seed1-training.json runs/randi-pairs.json

.venv-jax/bin/python backends/jax/overfit.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --targets ADAL ADAR --steps 300 --output runs/level0-capacity-adal-adar-300

.venv-jax/bin/python scripts/audit_capacity_fit.py \
  --checkpoint runs/level0-capacity-adal-adar-300/epoch-300.json \
  --manifest runs/level0-capacity-adal-adar-300/manifest.json \
  --training runs/overfit-seed1-training.json --graph runs/c302-audit.json \
  --output runs/capacity-final-audit.json
```

Output paths must be new. The fitter and independent NumPy auditor load only the
Rust training export, graph, and model artifacts. They do not load the full atlas
or held-out recordings. Input file hashes and exact subset trial IDs are retained.
The native export command validates the original split before emitting training
statistics. The independent audit checks weighted bounds, MSE, and chemical
coupling at the prepared state.

## First run result

The declared 300-update run completed in 505 seconds on CPU (including JAX
compilation; concurrent tests ran during part of it, so this is not a controlled
performance benchmark). [Receipt and training curve](level0-capacity-adal-adar-300.json)
retain input hashes, subset IDs, source hashes, and independent NumPy checks.
The implementation is `497f083`; the original manifest correctly records launch
from its dirty parent. The separate source receipt identifies the exact committed
backend files without rewriting that launch history.

| Quantity | MSE |
| --- | ---: |
| Zero response | 0.05145506 |
| Initialized model | 0.05092815 |
| 100 updates | 0.04856137 |
| 200 updates | 0.04548843 |
| 300 updates | 0.04484869 |
| Zero-start empirical mean bound | 0.04153675 |

Captured zero-start mean-response energy rises from **5.31% to 66.61%**. The final
iterate is also the best observed training iterate, but it **fails the declared
90% capacity gate**. This is substantial training improvement, not convergence,
a held-out comparison, or proof that the remaining gap is a capacity limit.
The intervention combines longer optimization, shifted rest, random signs and
per-neuron gains; this run does not isolate their individual contributions.

Independent NumPy replay matches both endpoint MSEs to the displayed floating-point
precision and confirms all 3,638 chemical coupling coefficients are nonzero.
Final half-step MSE is 0.044848704, with maximum prediction change 0.000716.
Doubling preparation to 120 seconds gives MSE 0.044845435, maximum prediction
change 0.01309, and reduces the unforced derivative maximum from 8.30e-5 to
4.55e-8. Aggregate improvement survives both checks, but 60 seconds is no longer
an equilibrated initial state for every fitted parameter set. Before a larger
fit, test longer preparation and require sensitivity checks on its checkpoints.

Final unregularized observation gains span 0.513–215.924. They are diagnostic
calibration parameters, not physiological estimates. Their growth and the
remaining shape error warrant investigation before full-cohort fitting. No new
validation or test response scores were produced. All 49 JAX tests passed,
including gain gradients, subset rejection, checkpoint reload, and the existing
Rust-scored integration tests.

## Decisions after this run

If training error is still falling at 300 steps, test a declared 1,000-update
budget from the same initialization; optimizer resume is not implemented. If it
plateaus far above the bound, inspect response shape, stimulus amplitude and
calcium time constants, then compare learning rates on these training targets.
A missed capacity gate does not distinguish optimization from model capacity by
itself. Do not add plasticity, dark edges, or more backends to explain the miss.

After a convincing small-target fit, repeat on unrelated training targets and
seeds, then run the full training cohort for 1,000 updates. Restore explicit
regularization and compare trace-only and joint classification objectives using
the existing validation partition, labeled exploratory. Use the same Rust
scorers and LDS comparison, with paired target-cluster uncertainty. Passing a
training gate does not establish superiority over LDS.

## Fresh confirmation

The existing validation and test targets have been inspected. They remain useful
for exploratory comparisons but cannot become fresh by reshuffling. No fresh
confirmatory holdout has yet been secured. The [metadata-only DANDI overlap audit](FRESH-HOLDOUT-AUDIT.md)
rejects the atlas NWB release as an independent recording cohort. Before new model selection, identify
additional uninspected recordings and freeze their identities and hashes using
metadata only, including recording/animal overlap checks against this atlas.
Predeclare preprocessing, eligibility, endpoints, and the final comparison before
opening response values. If no independent cohort is available, report the
limitation and make no confirmatory claim. Neuromodulation remains the first
scientific extension after the core fitting and comparison gates are met.

## Residual diagnosis and longer-run declaration

The [independent residual audit](level0-capacity-residual-diagnostic.json) fits
closed-form scales to the saved 300-update predictions, sharing each neuron's
scale across both targets. An optimal nonnegative rescaling reduces MSE from
0.04484869 to 0.04473680, closing only **3.38% of the remaining gap** to the
zero-start bound. Even allowing inadmissible negative observation scales gives
0.04448517. These are frozen-dynamics diagnostics, not new fitted checkpoints or
held-out scores. Waveform and target dependence need further optimization;
rescaling alone cannot remove most of the residual. The calculation is checked
against independent weighted `numpy.linalg.lstsq` solutions, including negative
optima and zero predictions.

At the same frozen parameters, extending preparation from 120 to 240 seconds
changes MSE by 1.14e-9 and predictions by at most 6.85e-6. This supports trying
120 seconds for the next fit, but is not a convergence guarantee for future
parameter values. Numerical checks must be repeated after training.

The next run is declared as **1,000 updates**, seed 1, ADAL/ADAR, Adam 0.01,
rest −0.2, initial per-neuron gain 10, and **120-second preparation**. It starts
from the original epoch-zero model; it does not resume the 300-update optimizer.
All other capacity-test settings and the 90% gate remain unchanged. Changing
preparation means this is not a pure duration ablation. No validation/test
selection or new dynamical features are introduced.

```sh
.venv-jax/bin/python backends/jax/overfit.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --targets ADAL ADAR --steps 1000 --preparation-seconds 120 \
  --output runs/level0-capacity-adal-adar-1000-prep120
```

The runner now records exact backend source hashes and its process ID. A live
process and advancing progress file establish that the run is active; a manifest
alone does not. Only a terminal `result.json`, followed by independent replay and
preparation/step checks, supports reporting the completed outcome.

### Interim zero-current control

The longer run remains in progress. Its saved **150-update** checkpoint was
checked with an additional control in `check_capacity_numerics.py`: set every
stimulus-current sample to zero after the identical unforced preparation, leaving
all fitted parameters and the readout fixed. This measures autonomous drift over
the response window, rather than only the derivative at its start.

The [interim receipt](capacity-prep120-interim-control.json) gives stimulated MSE
0.04624398 and zero-current MSE 0.05145682 (the zero-response baseline is
0.05145506). Zero-current prediction energy is 2.37e-10 versus 0.005303 with
stimulation, a ratio of 4.46e-8 on the observed, confidence-weighted traces.
Residual drift therefore does not explain the training improvement at this
checkpoint. Doubling preparation changes MSE by about 1.06e-7. These controls
must be repeated on the terminal checkpoint; they do not certify the ongoing
run's final parameters, optimization convergence, or generalization.
