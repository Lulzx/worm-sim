# Do fitted signs move toward molecular evidence?

This probe tests one claim behind the planned molecular-prior compiler: that
CeNGEN-derived chemical sign evidence carries information the training data also
supports. It fits nothing. It compares saved sign parameters at initialization and
endpoint with the [audited molecular evidence](MOLECULAR-PRIORS.md), so it opens no
recordings and uses no held-out labels.

```sh
S=runs/level0-atlas-sign-seed
.venv-jax/bin/python scripts/probe_molecular_sign_agreement.py \
  --evidence th2=runs/molecular-th2/evidence.json --evidence th4=runs/molecular-th4/evidence.json \
  --pair capacity=${S}1-fit/epoch-0.json,runs/capacity-prep240-eval1001/last-accepted.json \
  --pair seed1=${S}1-fit/epoch-0.json,${S}1-fit/epoch-25.json \
  --pair seed2=${S}2-fit/epoch-0.json,${S}2-fit/epoch-25.json \
  --pair seed3=${S}3-fit/epoch-0.json,${S}3-fit/epoch-25.json \
  --permutations 10000 --output docs/molecular-sign-probe.json
```

## Design

The unit is the tied `chemical_sign` group, not the edge: tied edges share one
fitted value, so counting edges would inflate the sample. 2,457 groups cover the
3,638 edges. A group is labeled when its directional member edges agree. No group
has conflicting directional members at either threshold. Threshold 2 labels 198
groups (6 excitatory, 192 inhibitory); threshold 4 labels 397 (74/323).

The statistic is the mean of label × change in effective reversal
`2σ(raw)−1`. Its null permutes labels among groups within strata of presynaptic
transmitter signature and initial sign. The transmitter stratum keeps the test from
rewarding "GABA synapses are inhibitory" alone. The initial-sign stratum removes
shrinkage: a sign prior moves every positive start down and every negative start
up, whatever the label. A shrinkage-adjusted variant subtracts a pooled linear fit
on the starting value. A planted-signal reference forces a fraction of labeled
groups to move toward their label, keeping magnitudes, and reruns the test.

## Results

| Comparison | Labels | Labeled groups moving toward label (above-median movement) | Alignment z | p |
| --- | --- | ---: | ---: | ---: |
| Capacity endpoint (79.31%) | th2 | 51 / 101 | −1.88 | 0.062 |
| Capacity endpoint | th4 | 112 / 201 | −0.40 | 0.68 |
| Seed 1, 25 updates | th2 | 55 / 94 | +0.38 | 0.73 |
| Seed 2, 25 updates | th2 | 50 / 96 | +0.57 | 0.38 |
| Seed 3, 25 updates | th2 | 48 / 95 | +0.06 | 0.96 |

Shrinkage adjustment changes no conclusion. The full receipt, including threshold-4
seed results and planted references, is [molecular-sign-probe.json](molecular-sign-probe.json).

**The capacity fit shows no agreement with molecular signs.** Its 98 sign flips
include 10 labeled groups, 6 of them toward the label. At threshold 2 the trend is
slightly against the labels, short of significance and not replicated at threshold
4. The planted reference bounds what this can exclude: forcing half the labeled
groups toward their labels (about 75% agreement overall) gives z = +4.4 at threshold
2 and +5.3 at threshold 4, but forcing a quarter does not reach significance. A
strong shared signal would have been visible; a weak one would not.

**The 25-update sign restarts carry no information about signs.** Under Adam with
sign-prior strength 0.01, essentially every group moves about 0.097 toward zero.
Only 2–3 of 2,457 groups per seed move against shrinkage, and none is labeled. Their
planted references are large only because movement magnitudes are uniform; they do
not indicate power about data-driven sign changes, because the data barely moved
any sign.

## Interpretation

This is a negative result with a narrow scope. The only informative fit is one
lineage trained on two related targets (ADAL/ADAR), so most groups receive little
data gradient. The labels cover 8–16% of groups and are mostly inhibitory, and the
ionotropic expression rule cannot express metabotropic or peptidergic effects.
Absence of agreement is therefore consistent with an uninformative molecular rule,
with two targets not constraining these signs, or with the Level 0 model class
using signs differently from biology. It does not show that the fit's signs are
biologically wrong, and expression evidence is not a sign measurement.

For the compiler experiment, the consequence is concrete. A molecular initialization
cannot be expected to help simply because fits drift toward it; any benefit has to
come from changing which basin the optimizer reaches. The declared comparison
therefore needs the stratified shuffled-label arm and several non-directional sign
seeds, and its sign readout should use edges whose fitted values actually move.
