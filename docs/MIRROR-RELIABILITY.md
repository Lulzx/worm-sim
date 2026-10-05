# Left/right reliability of connectome counts

Level 0 treats every c302 synapse count and gap-junction size as exact. This
analysis estimates how reproducible those numbers are, using the animal's bilateral
symmetry as a test-retest design. It adapts fly-brain's per-connection uncertainty
analysis to the worm's denser graph and lower count floor. It reads only the
graph and changes no model or benchmark.

```sh
.venv-jax/bin/python scripts/mirror_reliability.py --nulls 200 \
  --summary docs/mirror-reliability.json --edges data/c302-edge-reliability.json
```

## Design

A terminal L/R suffix is swapped only when the partner exists (AVAL ↔ AVAR,
RMDDL ↔ RMDDR), giving 99 mirrored pairs and 104 unpaired cells. Unpaired cells
such as AVL, PVR, RIR, DA1–9 and VB1–11 map to themselves. A connection a→b and its
mirror m(a)→m(b) form one pair; connections whose mirror is themselves are excluded.
This gives 2,236 chemical and 638 gap-junction pairs. A further 516 chemical and
227 gap edges have no defined mirror.

Disagreement between sides mixes reconstruction error, genuine asymmetry (ASE, AWC)
and, where the source reconstruction pooled animals, inter-animal variability.
**Every estimate is therefore a lower bound on measurement precision.**

Chance presence comes from 200 degree-preserving double-edge-swap rewirings. Each
edge keeps its count, so chance is computed within each count bin. Counts follow a
censored, zero-inflated Poisson-lognormal model. The two sides share a latent
log-rate θ ~ N(μ, τ²), and each side adds N(0, σ²) noise. Each side then drops out
with probability `sigmoid(c0 + c1·η)` of its own log-rate η; surviving counts are
Poisson. A pair is observed only when one side has a contact. Integrals use 40×40
Gauss-Hermite quadrature over unique count pairs. `gammaln` accepts the twelve
half-integer gap sizes, which appear to be upstream averages.

## Results: weak edges are unreliable in existence

| Chemical count | Edges with mirror | Mirror present | Rewired chance (95%) | Model |
| --- | ---: | ---: | ---: | ---: |
| 1 | 957 | 32.1% | 9.1% (7.1–11.3) | 36.2% |
| 2 | 570 | 43.2% | 9.8% (7.2–12.1) | 49.2% |
| 3 | 354 | 53.7% | 9.9% (7.1–13.3) | 59.6% |
| 4–5 | 362 | 68.8% | 10.1% (6.9–13.3) | 69.6% |
| 6–9 | 363 | 80.4% | 10.8% (7.4–13.8) | 80.8% |
| 10–19 | 320 | 92.5% | 9.8% (6.9–13.8) | 88.6% |
| ≥20 | 196 | 98.0% | 8.7% (5.1–13.8) | 94.9% |

Single-synapse connections are real structure: their mirrors appear 3.5 times as
often as in rewired graphs. Most still fail to reproduce. The fitted dropout
probability is 0.64 at an expected rate of 0.5 synapses, 0.49 at 1, 0.19 at 5
and 0.06 at 20. When both sides are present, counts agree well: log(1+count)
correlates at 0.74, and the latent reliability τ²/(τ²+σ²) is 0.98. The main
uncertainty is whether a weak connection exists, not how large it is once present.
This is the opposite emphasis from fly-brain's finding at its 3-synapse floor:
there, existence was reliable and weight was not.

Gap junctions show the same pattern with more noise: 28.7% mirror presence at size
1 against 7.6% chance, both-present correlation 0.46, latent reliability 0.94.

The model is approximate. It overpredicts presence by 4–6 points at counts 1–3 and
underpredicts it by 3–4 points at ≥10 chemical synapses; gap junctions at 10–19 are
overpredicted (73% observed, 90% modelled, n = 45). A constant dropout probability
or no dropout fits much worse; both were tried during development, and neither is
retained.

## Empirical-Bayes counts

[c302-edge-reliability.json](../data/c302-edge-reliability.json) gives each edge's
posterior mean latent rate, pooling its mirror where defined, plus flags for mirror
definition and presence. It follows the sorted chemical and canonical gap order.

| Observed chemical count | EB count, mirror present | n | EB count, mirror absent | n |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.64 | 307 | 1.05 | 650 |
| 2 | 2.42 | 246 | 1.61 | 324 |
| 3 | 2.97 | 190 | 2.26 | 164 |
| 5 | 4.49 | 115 | 3.78 | 53 |
| 10 | 9.43 | 55 | 8.31 | 7 |

Two single-synapse connections differ by a factor of about 1.6 in estimated strength
depending on whether the other side confirms them. These are not proposed
replacements for the counts in existing fits; they are an input for weighting.

## Consequence for molecular evidence

The edges with directional molecular labels are among the least reliable.
Threshold-2 inhibitory edges have median count 2, and 43% of those with a defined
mirror reproduce, against 57% for all chemical edges. Threshold-4 inhibitory edges
reproduce at 47%. The [sign probe](MOLECULAR-SIGN-PROBE.md) weighted every
labeled group equally. Any molecular-prior compiler should carry edge reliability
alongside expression evidence, so a label on an unconfirmed single-synapse
connection does not count as much as one on a reproducible connection.
