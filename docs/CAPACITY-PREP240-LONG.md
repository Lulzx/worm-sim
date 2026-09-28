# Longer fit with audited preparation

The [201-evaluation fit](CAPACITY-PREP240.md) reaches 78.82% training-response
capture, misses the 90% gate and remains nonstationary. Unlike its earlier parent,
its endpoint gradients agree across 240/480/960-second preparation. The next
[declared run](../configs/capacity-prep240-long.json) increases the budget to
1,001 actual objective/gradient evaluations and at most 1,000 accepted updates,
with no other optimizer, model, data or numerical changes.

This starts from the same original warm parent as the 201-evaluation run,
repeating the prefix to rebuild L-BFGS history. It does not resume optimizer
state from the final checkpoint, and it is not an independent initialization.
Repeating the prefix costs evaluations; the total budget includes them.

```sh
.venv-jax/bin/python backends/jax/overfit_lbfgs.py \
  --model runs/level0-atlas-sign-seed1-fit/epoch-0.json \
  --graph runs/c302-audit.json --training runs/overfit-seed1-training.json \
  --warm-start runs/capacity-scaling-curvature-v1-eval201/last-accepted.json \
  --dt .005 --maxcor 100 --coordinate-scaling curvature-v1 \
  --preparation-seconds 240 --max-evaluations 1001 --max-iterations 1000 \
  --output runs/capacity-prep240-eval1001

python3 scripts/audit_lbfgs_prefix.py \
  --reference runs/capacity-prep240-eval201 \
  --candidate runs/capacity-prep240-eval1001 \
  --declaration configs/capacity-prep240-long.json \
  --output runs/capacity-prep240-long-prefix.json
```

Verify all 201 reference trial records and 190 accepted records, including
initialization, within the existing 1e-10 scalar tolerance. Inputs, parameter
lineage, fitting source hashes and configuration must match except declared
budgets; elapsed time is excluded. An interim partial receipt is explicitly
incomplete. Do not interpret the extra evaluations until the complete prefix
check passes.

Use the final accepted iterate for the unchanged 90% gate, and report best
accepted loss separately. Preserve all trials and nonfinite failure artifacts.
At termination repeat independent NumPy replay, step/preparation controls,
240/480/960-second gradient comparison, raw/scaled stationarity and frozen-gain
calibration. The longer budget is not a convergence guarantee. No budget increase,
new dynamics or held-out model selection is authorized by interim results.
The full specification and scientific acceptance remain incomplete.


## Repeated-prefix verification

The [complete prefix receipt](capacity-prep240-long-prefix.json) verifies all
201 reference objective/gradient trial records and all 190 accepted-state
records, including initialization, within the declared 1e-10 scalar tolerance.
The audit checked the declared input, parameter-lineage, fitting-source and
configuration agreement, allowing only the declared budget changes. Elapsed
time was excluded. The candidate snapshot contained 203 evaluation records and
191 accepted-state records; its file hashes identify that snapshot, not the
still-growing final history.

This passes the prerequisite for interpreting the additional optimization.
It does not establish bitwise parameter/optimizer-state equality, convergence,
the 90% capacity gate or held-out performance. The run remains in progress;
endpoint replay, numerical controls and stationarity checks are still required.
