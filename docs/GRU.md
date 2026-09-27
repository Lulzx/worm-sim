# Connectome-free masked GRU

This Task 2 baseline uses a dense recurrent network and the same fixed animals,
10-second observation prefix, 30-second free forecast and common scorer as the
LDS and Level 0 experiments. Anatomy supplies canonical names and lineage checks;
no edges, neuron classes or sign priors enter the network. Behavior channels are
not used in this experiment by any comparator.

The Rust implementation uses a reset-before GRU, following the gated recurrent
unit introduced by [Cho et al. (2014)](https://arxiv.org/abs/1406.1078):

```
r = sigmoid(Wr x + Ur h + br)
z = sigmoid(Wz x + Uz h + bz)
c = tanh(Wc x + Uc (r * h) + bc)
h_next = z * h + (1-z) * c
y_next = C h_next + b
```

Here `z` retains the previous state. The initial hidden state is fixed at zero.
Input `x` concatenates N standardized neuron values and N observation-confidence
masks. During history, present observations replace predicted values and their
masks equal identity confidence; missing values use the model prediction and mask
zero. Beyond the origin, every input is a model prediction with mask zero. There
is no teacher forcing on future targets, including during training. Prediction at
frame t uses observations only through t−1. Prefix outputs are one-step predictions,
not the posterior reconstructions reported for the LDS/history filter.

Weighted per-neuron means and standard deviations use training animals only.
Identities never observed in training use a disclosed persistence fallback on
prediction, matching the LDS/AR convention. This differs from Level 0's default
readout for those identities. Missing/zero-confidence training targets are masked.
The loss is confidence-weighted MSE in standardized coordinates, averaged within
each window; windows receive equal optimizer updates. It is not the Level 0 raw
fluorescence training objective. All models are selected and evaluated by the
same unstandardized fluorescence R² scorer.

The reverse pass differentiates through every prefix and forecast step, including
feedback of predicted fluorescence to subsequent inputs. History observations and
masks are constants; hidden states are not detached at the origin. Tests check all
weights against finite differences of an independently aggregated prediction loss,
a scalar recurrence, fitting loss reduction and invariance of an entire refit and
test prediction after replacing test futures.

## Fixed first experiment

[Configuration](../configs/gru-first-fit.json): six hidden units, 30 epochs,
learning rate 0.003, Adam moments 0.9/0.999 with epsilon 1e-8, gradient norm clip 1,
and decoupled weight decay 1e-4 on all weights/biases. Seed 42 initializes
SplitMix64 Xavier-uniform weight matrices and zero biases; windows are shuffled
by SHA-256(seed, epoch, trial ID). Optimizer state persists between epochs.

With 149 training-observed identities there are
`3h(2N+h+1) + N(h+1) = 6,533` trainable weights/biases and 298 training-derived
normalization statistics: **6,831 total scalars**. The six inferred hidden states
per trial are not fitted population scalars. For comparison Level 0 has 7,040
trainable scalars plus 298 calibration statistics, and the selected rank-32 LDS
has 7,295 scalars including normalization. Capacity counts do not establish
biological identifiability or matched expressivity.

Epochs 0 through 30 are evaluated on validation animals. Selection maximizes mean
macro-neuron R² at 1/10/30 seconds, with earlier epochs winning ties. Each candidate
is saved; these artifacts are not optimizer-resume checkpoints. Selected-model
test scoring occurs after selection and includes the same whole-animal bootstrap.
This is one fixed architecture and seed, not an exhaustive GRU hyperparameter
search. Test animals were inspected in earlier experiments, so claims remain
exploratory. Source preprocessing causality and biological forecasting acceptance
are still open.

```sh
WORMSIM_COMMIT="$(git rev-parse HEAD)" cargo build --locked --release
mkdir -p runs
./target/release/wormsim gru-fit data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  configs/gru-first-fit.json runs/gru-first-fit.json
./target/release/wormsim gru-predict data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  runs/gru-first-fit.json test runs/gru-first-test-predictions.json
./target/release/wormsim bench-score data/c302-herm.wsc \
  runs/wormwideweb-benchmark.json data/wormwideweb-animal-split.json \
  runs/gru-first-test-predictions.json test runs/gru-first-test-report.json
```

## First-run outcome

The [committed-source receipt](gru-first-receipt.json) records source
`000cb23f03b46a1da4941a196c5f62cf4e1e3dbd`, all 31 candidates and the selected
model hash. The run completed 10,800 training-window updates. Epoch **7** wins
validation with 0.12040 / 0.04892 / 0.04604 at 1/10/30 seconds. Training MSE falls
from 0.96649 in epoch 1 to 0.82389 in epoch 30, but the later epochs do not improve
the declared validation criterion.

| Selected epoch 7, test | R² | Marginal animal-bootstrap 95% interval |
| --- | ---: | --- |
| 1 s | 0.11342 | [0.03572, 0.13900] |
| 10 s | 0.02256 | [−0.01531, 0.08102] |
| 30 s | 0.01344 | [−0.02586, 0.05741] |

The two long-horizon intervals include zero. These are exploratory positive point
estimates, not evidence of robust positive forecasting across animals. The small
GRU has lower point scores than the stable LDS at every horizon and than AR(1) at
1 and 10 seconds. This experiment does not show a nonlinear-baseline advantage.
Nor does failure of this one architecture establish that other GRUs cannot do
better. No test-based architecture or epoch tuning was performed in this run.

The timing sum for training and validation candidates is **26.93 seconds** on the
M4 Pro CPU. It excludes loading, normalization/initialization, candidate writes and
separate final scoring. Speed is adequate for this model; no accelerator is needed.
The 48 fallback traces on test comprise AVBL and URAVR, the same two identities
absent from training in the LDS and AR comparisons.

```sh
python3 scripts/score_gru.py
```

The next comparison should give all fitted models equal access to behavior, with
an explicit distinction between behavior observed during history and future
behavior unavailable at forecast time. Upstream neural/behavior preprocessing
causality must be audited before interpreting that comparison as a causal forecast.
