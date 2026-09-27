# Paired learning-rate test after the long capacity fit

The [1,000-update result](LEVEL0-LONG-RUN.md) improved training fit but missed the
90% capacity gate and showed transient loss spikes. The declared experiment compared
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


## Outcome: both runs failed before 200 updates

Both runs launched from clean source `d44b04f`. The [complete receipt](capacity-warm-results.json)
retains both manifests and every finite training iterate. Neither produced a
normal terminal result; the predeclared final-at-200 comparison is unavailable.

| Learning rate | Last finite update | Best retained MSE | Captured response energy | Failure update |
| --- | ---: | ---: | ---: | ---: |
| 0.001 | 180 | 0.04385085 | 76.67% | 181 |
| 0.01 | 19 | 0.04386920 (initial parent) | 76.48% | 20 |

Both exited with a nonfinite objective/gradient error. The lower rate improved
slowly but did not pass the 90% gate. Its best retained checkpoint independently
replays in NumPy with identical MSE. Halving the step changes MSE by 1.81e-7;
doubling preparation changes it by 5.23e-11. Unstimulated prediction energy is
3.38e-13 versus 0.00761 with stimulation. These checks validate the retained
iterate, not the next failed update.

### Reproduced higher-rate failure

`backends/jax/replay_capacity_failure.py` verifies original backend and input
hashes, reconstructs fresh Adam moments, matches all 20 finite recorded MSEs,
and captures the finite parameters whose evaluation fails at update 20.
It is a diagnostic replay, not a completed or restarted comparison run.
The evaluation has a nonfinite objective and 7,076 nonfinite gradient coordinates.

The [independent stability audit](capacity-reset-stability-audit.json) finds that
Euler preparation fails at a 0.01-second step but succeeds at 0.005 and 0.0025.
Their training MSEs are 0.04478081 and 0.04478097. At the finer-step prepared
state, the analytically assembled Jacobian agrees with directional finite
differences to 1.89e-8. Its local Euler step limit is about 0.01103 seconds:
the original step is locally stable there. Thus equilibrium linearization does
**not** establish the cause of failure along the nonlinear preparation path.
Finer-step finite predictions also do not establish finite or accurate gradients.
The lower-rate failure has not been reproduced at its failing parameters.

### Decision

Keep dynamical features frozen. Before another longer fit, check the preparation
trajectory and differentiated objective at refined steps or with the existing
adaptive solver. The runner now automatically writes `failure.json` on a returned nonfinite
objective or gradient, retaining finite parameters and nulling nonfinite metrics.
If parameters themselves are nonfinite, it records their count and omits the
model. Exceptions raised inside the solver are not yet captured by this path. Declare any changed-solver fit separately; do not treat it as completion
of these failed runs. The capacity gate and fresh-holdout requirement remain
unchanged.


### Refined-step gradient check

The [gradient receipt](capacity-reset-gradient-audit.json) evaluates the captured
higher-rate failing parameters with the original trace objective at steps 0.005
and 0.0025 seconds. All 7,287 gradient coordinates are finite. Central differences
along the normalized gradient at perturbation 1e-5 agree with autodiff to relative
errors 2.26e-8 and 2.20e-8; the larger 1e-4 perturbation gives about 2.24e-6.
The two gradients have cosine 0.999999964 and relative norm difference 0.0003873.
MSE agrees with the independent NumPy refinement to floating-point precision.

This verifies one direction at one failing parameter state, not every coordinate
or a full fitting trajectory. It supports a separately declared refined-step fit;
it does not retroactively complete either failed run. The audit is reproducible
with `backends/jax/audit_capacity_gradients.py` and records input/backend hashes.
The original failure replay requires the original backend files identified by its
manifest; use that source version when reproducing it after runner changes.
