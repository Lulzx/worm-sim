# Molecular-prior compiler v0: declared capacity comparison

This declares the first test of a "neural compiler" in the sense of Davy's
*302-neuron shortcut* proposal: a fixed mapping from molecular and structural
measurements to model parameters, tested on the worm before anything larger. It
contains no results. Endpoint receipts are required before interpreting any arm.

## What the compiler does

[`compile_molecular_init.py`](../scripts/compile_molecular_init.py) is a fixed,
audited rule; nothing in it is learned. It starts from a random-sign epoch-zero
model and touches only tied `chemical_sign` groups:

1. Take the threshold-2 [molecular evidence](MOLECULAR-PRIORS.md). A group is
   labeled when its directional member edges agree. No threshold-2 group mixes
   directions.
2. Keep the label only if the wiring is reproducible: at least one directional
   member edge has a present left/right mirror, or has no defined mirror
   (midline cells cannot be tested). Reliability comes from the
   [mirror analysis](MIRROR-RELIABILITY.md).
3. Set each kept group to its label at the seed's own magnitude
   (`ln 3`, reversal ±0.5). Every other value is copied unchanged.

Threshold 2 labels 198 groups. Step 2 drops 121 of them, whose directional edges
all lack an existing mirror. The remaining 77 (4 excitatory, 73 inhibitory; 47
confirmed, 30 untestable) are compiled. Because the seed already agrees with about
half of them, 42, 39 and 34 groups change for seeds 1–3. The intervention is small:
about 1.6% of 2,457 sign groups. In the existing seed-1 capacity fit these groups
moved about twice the median amount (median |Δreversal| 0.068 against 0.033), so the
ADAL/ADAR objective does engage them.

The shuffled arm assigns the same labels to other groups, permuted within strata
of presynaptic transmitter signature and wiring status (permutation seed
100 + seed). Magnitudes, label counts and stratum composition match the molecular
arm exactly; only which groups receive which labels differs. Within the small
`GABA|confirmed` stratum (21 of 27 groups labeled) the shuffle can barely differ, and
overall 25–29 of the 77 assignments coincide with the molecular arm. This is the
price of controlling for transmitter class; the contrast rests on the other strata.

The [probe](MOLECULAR-SIGN-PROBE.md) found that fits do not drift toward molecular
signs on their own. Any benefit here must therefore come from changing which basin
the optimizer reaches, not from saving it a journey it would make anyway.

## Arms and protocol

| Arm | Start | Seeds |
| --- | --- | --- |
| random | `level0-atlas-sign-seed{1,2,3}-fit/epoch-0.json`, unchanged | 1, 2, 3 |
| molecular | compiler v0 applied to the same seed | 1, 2, 3 |
| shuffled | stratified label permutation applied to the same seed | 1, 2, 3 |

Model hashes are in the [configuration](../configs/capacity-molecular-init-comparison.json);
compiler receipts with every changed group are in [molecular-init/](molecular-init/).
All arms share the 50 ADAL/ADAR training trials, rest −0.2, dt 0.005, 240-second
preparation, `curvature-v1` scaling, L-BFGS with 100 curvature pairs, at most 401
evaluations and 400 accepted updates, and no priors. These are fresh starts, not
warm continuations of the 79.31% lineage, so the budget is larger than the
201-evaluation warm comparisons. Each training export differs from seed 1's only in
model hash and source commit.

Each arm runs in two stages. A one-step `overfit.py` run writes the epoch-zero
capacity checkpoint at the declared numerics; L-BFGS then starts from that
`epoch-0.json`. This needed one fitter fix: `overfit_lbfgs.py` assumed every parent
recorded a warm-start dictionary, while fresh `overfit.py` parents record `null`.
An end-to-end smoke run (compiled seed 1, two evaluations) moved training MSE from
0.050927 to 0.050673 at about 29 seconds per evaluation.

```sh
O=runs/molecular-init
for s in 1 2 3; do for arm in random molecular shuffled; do
  d=$O/$arm-seed$s
  .venv-jax/bin/python backends/jax/overfit.py --model $d/model.json \
    --graph runs/c302-audit.json --training $d/training.json --targets ADAL ADAR \
    --steps 1 --dt .005 --preparation-seconds 240 --output $d/start
  .venv-jax/bin/python backends/jax/overfit_lbfgs.py --model $d/model.json \
    --graph runs/c302-audit.json --training $d/training.json \
    --warm-start $d/start/epoch-0.json --dt .005 --preparation-seconds 240 \
    --maxcor 100 --coordinate-scaling curvature-v1 \
    --max-evaluations 401 --max-iterations 400 --output $d/lbfgs
done; done
```

The arm directories were populated by compiling each seed (`--arm molecular`, or
`--arm shuffled --permutation-seed 10$s`), copying the random start, and exporting
training with `export_atlas_training ... runs/randi-pairs.json`. Runs may execute
concurrently; timings are not controlled. At about 29 seconds per evaluation alone,
nine 401-evaluation arms are roughly 30 CPU-process hours.

## Decision rule, fixed before any fit

**Primary:** last accepted captured zero-start energy, molecular minus shuffled,
paired by seed. Compiler v0 is called *supported on this objective* only if
molecular exceeds shuffled in all three seeds **and** the mean paired difference
exceeds the largest absolute difference between any two random-arm seeds. The second
condition requires the effect to exceed ordinary seed-to-seed variation.

**Secondary:**

- shuffled minus random, paired by seed: does overwriting the same number of
  groups with any stratified labels matter?
- molecular-sign agreement of each endpoint, from the probe script restricted to
  groups above that run's median movement;
- cross-seed agreement of endpoint signs within each arm, for groups moving above
  median in every seed.

The 90% capacity gate is reported, not used for this decision. Endpoints get the
existing independent NumPy replay, half-step and longer-preparation controls, and
a check that inputs, backend sources and configuration match across arms except
model and training hashes. No retries, budget extensions, new dynamics, threshold
changes or held-out scoring. Nonfinite failures are kept and count as failed arms.

## What each outcome would mean

Three paired seeds support no significance claim: a sign test on three pairs
cannot fall below p = 0.125. A *supported* result would be the first evidence in
this repository that molecular data shapes which fit Level 0 reaches. It would
justify replication on an unrelated target pair and a threshold-4 sensitivity run
(177 compiled groups), still on training data.

A null result has three readings that this design cannot separate: the ionotropic
expression rule carries no usable sign information; two related targets do not
constrain these signs; or Level 0 uses signs differently from biology. Under the
compiler programme, the next step would then be the mechanism the rule cannot
express, neuromodulation, rather than more sign variants. Neither outcome bears on
held-out prediction, which still awaits an uninspected confirmatory cohort.
