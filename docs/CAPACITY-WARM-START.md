# Paired learning-rate test after the long capacity fit

The [1,000-update result](LEVEL0-LONG-RUN.md) improved training fit but missed the
90% capacity gate and showed transient loss spikes. The next experiment compares
a lower learning rate with a reset-optimizer control, keeping the dynamics,
readout, target subset and objective unchanged.

The [declaration](../configs/capacity-warm-learning-rates.json) fixes checkpoint
950 by file hash, 200 additional updates in each run, and rates **0.001** and
**0.01**. Both runs start from exactly the same fitted parameters and fresh Adam
moments. The control separates the learning-rate change from optimizer reset.
This is a warm start, not continuation of the original Adam state. No validation
or test observations enter either run.

The primary comparison is final training MSE after 200 additional updates. Report
both runs, plus best saved MSE and the complete loss/gain histories. The capacity
gate remains final capture of at least 90% of the same zero-start mean-response
energy. Numerical and drift checks must be repeated on final and selected saved
checkpoints before interpreting improvements. These are optimization diagnostics,
not evidence of held-out performance or a reason to add new dynamical modules.

## Implementation contract

`overfit.py --warm-start` accepts only a diagnostic checkpoint with identical
targets, data lineage, initial state, parameter layout, readout configuration and
objective. Only the requested update budget and learning rate may differ. Reload
rejects modified frozen coordinates. Before updating, the initial training MSE
must reproduce the parent's recorded MSE within 1e-10.

Each run records the parent file hash, parent epoch/source, and explicit optimizer
reset. Its epoch zero means **zero additional updates in this run**, not the
original untrained initialization. Saved diagnostic artifacts also carry that
parent reference. The runner always constructs a fresh Optax optimizer state.

Every strict improvement now replaces `best.json` atomically; earliest exact ties
remain selected. Periodic 50-update snapshots and the final snapshot are also
retained. The terminal receipt hashes the best artifact. This policy applies to
new runs; the missing parameter snapshots in the completed long run have not
been reconstructed or silently replaced.

## Run

```sh
.venv-jax/bin/python backends/jax/overfit.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --targets ADAL ADAR --steps 200 --learning-rate 0.001 \
  --preparation-seconds 120 \
  --warm-start runs/level0-capacity-adal-adar-1000-prep120/epoch-950.json \
  --output runs/capacity-warm-lr0001

.venv-jax/bin/python backends/jax/overfit.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --targets ADAL ADAR --steps 200 --learning-rate 0.01 \
  --preparation-seconds 120 \
  --warm-start runs/level0-capacity-adal-adar-1000-prep120/epoch-950.json \
  --output runs/capacity-warm-lr001-reset-control
```

A separate one-update integration smoke reproduced the parent at local epoch
zero (MSE 0.04386919577634917), then exercised the new best-checkpoint path. It is
not either declared 200-update run or an optimizer comparison result.
