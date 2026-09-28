# Capacity continuation with 240-second preparation

The [gradient audit](CAPACITY-PREPARATION-GRADIENTS.md) found materially different
120- and 240-second gradients at the scaled endpoint, while 240 and 480 seconds
agreed closely. The [declared continuation](../configs/capacity-prep240.json)
therefore changes preparation explicitly to 240 seconds before further fitting.
This changes the finite-preparation objective. It is not a new dynamical feature
or an independent initialization, and a small loss shift is not treated as a
parameter-fit improvement.

The warm-start helper rejects preparation changes by default. The L-BFGS CLI
permits an explicit positive, nondecreasing duration override, retains all other
lineage/parameter-layout checks, and saves parent/run preparation durations and
an `initial-objective.json` loss-shift receipt. It reuses parameters only and
resets curvature history. Existing Adam warm-start behavior remains unchanged.

The fit has 201 actual objective/gradient evaluations, at most 200 accepted
updates, 100 curvature pairs, dt=0.005 and the same threshold/rest scaling as the
parent. The training subset, frozen coordinates, priors and observation model
are unchanged. Initial MSE must agree with the frozen 240-second audit within
1e-10. Report the last accepted endpoint and the existing 90% capacity gate.
No silent budget extension or held-out scoring is allowed.

```sh
.venv-jax/bin/python backends/jax/overfit_lbfgs.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --warm-start runs/capacity-scaling-curvature-v1-eval201/last-accepted.json \
  --dt .005 --maxcor 100 --coordinate-scaling curvature-v1 \
  --preparation-seconds 240 --max-evaluations 201 --max-iterations 200 \
  --output runs/capacity-prep240-eval201
```

After termination, independently replay the endpoint, check half-step and longer
preparation predictions, compare gradients at 240/480/960 seconds, and report
stationarity in both raw and optimizer coordinates. Gradient stability at the
parent is not a guarantee for future iterates. The original specification and
scientific acceptance gates remain open.

Five capacity/warm-start tests and seven optimizer tests pass. The
[one-evaluation smoke](capacity-prep240-smoke.json) preserves native and extension
parameters exactly and reproduces the previous 240-second loss exactly:
0.04365803876375997, a +1.70108e-6 shift from the parent's 120-second objective.
It performs no updates and is excluded from the declared fit. Its manifest
correctly records the pre-commit dirty worktree.
