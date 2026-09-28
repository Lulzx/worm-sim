# Fixed-seed fitting comparison

The [frozen-seed preflight](CAPACITY-PREP240-LONG.md#fixed-seed-objective-preflight)
retains the endpoint response and gradients while avoiding the audited loss
excursion. This [declared comparison](../configs/capacity-fixed-seed-comparison.json)
tests whether that initialization change improves a bounded fit.

Both arms start from the same `capacity-prep240-eval1001/last-accepted.json`
parameters with fresh L-BFGS history. Each gets at most 201 actual objective/
gradient evaluations and 200 accepted updates. Both use dt 0.005, 240-second
preparation, `curvature-v1` scales and 100 curvature pairs. The original arm
inherits the original initial vector. The fixed arm uses the audited prepared
vector, held constant for every evaluation. All data, masks and dynamics agree.

```sh
.venv-jax/bin/python backends/jax/overfit_lbfgs.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --warm-start runs/capacity-prep240-eval1001/last-accepted.json \
  --dt .005 --preparation-seconds 240 --maxcor 100 \
  --coordinate-scaling curvature-v1 --max-evaluations 201 --max-iterations 200 \
  --output runs/capacity-seed-original-eval201

.venv-jax/bin/python backends/jax/overfit_lbfgs.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --warm-start runs/capacity-prep240-eval1001/last-accepted.json \
  --fixed-seed-audit docs/capacity-fixed-seed-audit.json \
  --dt .005 --preparation-seconds 240 --maxcor 100 \
  --coordinate-scaling curvature-v1 --max-evaluations 201 --max-iterations 200 \
  --output runs/capacity-seed-fixed-eval201
```

The primary comparison is last accepted training MSE and termination status;
best accepted MSE is secondary. The 90% capacity gate is unchanged. Compare
input/source hashes and full histories, then independently replay endpoints,
check integration/preparation sensitivity and raw/scaled gradients. A lower
training loss does not establish the correct biological resting state or
held-out superiority. Existing held-out targets stay unused.

Both runs may execute concurrently; timings are not controlled benchmarks.
Neither arm receives additional evaluations after its declared termination.
Nonfinite failures are retained. This declaration contains no results; endpoint
receipts are required before interpreting either fit.
