# Reproducible sign initialization restarts

The [coupling diagnostic](ATLAS-INITIALIZATION-DIAGNOSTIC.md) found zero chemical
gate-to-voltage sensitivity at the original neutral initialization. The atlas
fitter now optionally starts tied chemical signs away from that state, without
changing sign priors, raw shrinkage centers, anatomy or response data.

```json
"sign_initialization": {"seed": 1, "reversal_magnitude": 0.5}
```

Each tied sign group receives one draw. Its positive-polarity probability is the
arithmetic mean of its member edges' declared prior probabilities: the molecular
overlay when present, otherwise the graph priors. Unknown edges therefore remain
probability 0.5 in sampling and in regularization. The draw chooses an initial
reversal of plus or minus the configured magnitude, implemented by raw sign
coordinates plus or minus `log((1+magnitude)/(1-magnitude))`. The magnitude must
be strictly between zero and one. This is a two-point optimization initialization,
not a posterior sample or a biological polarity assignment.

The uniform draw uses SHA-256 of the concatenation of:

1. The UTF-8 bytes of `wormsim-sign-init-v1` followed by a zero byte.
2. The seed as eight little-endian bytes.
3. The full UTF-8 tied-group name.

Interpret the first eight hash bytes as a little-endian unsigned integer, shift
right by 11 and divide by 2^53. Choose positive when this is strictly below the
group's prior probability. Name-keyed draws are independent of traversal order;
renaming a group intentionally changes its draw. Ties remain exact. Prior means
are preserved, including molecular tied-prior centers, and every non-sign
parameter is unchanged. The fixed preparation seed and subsequent
parameter-dependent unforced preparation retain their existing semantics.

Initialization has no access to fluorescence or detection labels. It adds no
trainable parameters. Predictions identify the sign seed in the model name and
seed field; split provenance remains in the unchanged split hash. Missing
configuration retains legacy initialization. The independent Python auditor
recomputes hashes, group probabilities, initial signs and unchanged prior centers
from the graph and optional source-audited molecular projection. Tests cover
repeatability, different seeds, endpoint probabilities, ties, preserved priors and
other parameters, invalid magnitudes, and held-out-fluorescence mutation through
the complete optimization trajectory.

## Frozen three-seed pilot

Configurations [seed 1](../configs/level0-atlas-sign-seed1-fit.json),
[seed 2](../configs/level0-atlas-sign-seed2-fit.json) and
[seed 3](../configs/level0-atlas-sign-seed3-fit.json) each use 25 updates, initial
observation gain 10, neutral graph priors, reversal magnitude 0.5, constant-rate
Adam at 0.01 and the existing trace-MSE plus pair-BCE objective. No correlation
loss, cosine schedule or AdamW is enabled. They otherwise match the frozen
[learned-gain control](../configs/level0-atlas-gain-fit.json). Source changes since
that control added optional features whose defaults preserve its fitting path.

Within each run, minimum validation trace MSE selects a checkpoint, including
epoch zero and earlier exact ties. Across the three runs, the predeclared primary
candidate is the one with minimum selected validation MSE, breaking exact ties
by seed in ascending order. Report every seed's result and dispersion, as well as
the selected candidate, against the gain control and LDS. Do not select a seed
using test outcomes. The runner emits each run's selected test predictions, but
those are excluded from the seed-selection rule. Numerical preparation/step
checks and independent replay must pass before interpreting outcomes.

This is a three-seed optimization pilot, not the specification's 10–50-member
uncertainty ensemble, a convergence proof or a fresh confirmatory test. Changing
initial signs may change basins and transient preparation; it does not guarantee
better forecasting. The two existing control fits continue unchanged. Results
for the sign-restart pilot are pending.

## Launch checks

All three runs launched from `a3768d1`. Independent hash-based checks reproduce
all 2,457 tied sign draws per seed and confirm unchanged prior centers and
non-sign initialization relative to the gain control. Positive/negative group
counts are 1,238/1,219 for seed 1 and 1,250/1,207 for seeds 2 and 3 (equal counts
do not mean equal assignments).

At each prepared epoch-zero state, all 3,638 chemical gate derivatives are
nonzero; unforced derivative norms are 1.19e-13, 1.22e-13 and 1.23e-13. The
independent coupling audits verify the local coefficients and preparation:
[seed 1](coupling-sign-seed1-initial-audit.json),
[seed 2](coupling-sign-seed2-initial-audit.json),
[seed 3](coupling-sign-seed3-initial-audit.json).
This verifies the intended initialization intervention, not equilibrium
uniqueness, optimization convergence or predictive improvement. Fits and final
comparisons remain pending.
