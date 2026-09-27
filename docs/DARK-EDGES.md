# Optional off-connectome chemical connections

Specification §5 permits a small L1-penalized set of extra edges. The JAX Level 0
API now accepts an explicit sparse candidate list. It does not discover candidates
or alter the imported anatomical graph. This is an experimental software feature;
no off-connectome biological connections or improved benchmark scores are claimed.

## Declaration and use

```python
from dark_edges import DarkEdges
from level0 import Level0, parameters

extra = DarkEdges(graph, {
    "schema_version": 1,
    "source": "synthetic example; not biological evidence",
    "max_edges": 1,
    "l1_strength": 0.03,
    "edges": [
        {"pre": "B", "post": "A", "group": "extra",
         "strength": 0.2, "reversal": 0.5}
    ],
})
engine = Level0(model, graph, times, dark_edges=extra)
theta = parameters(model, dark_edges=extra)
# Add this once to the full training objective, outside target/trace reductions:
penalty = engine.extension_penalty(theta)
```

This example requires a graph containing A and B without a B → A chemical edge.
`max_edges` is an explicit integer budget. Unknown neurons, self-connections,
duplicate candidates, existing directed chemical connections, missing provenance,
and inconsistent tied initial values are rejected. An anatomical gap junction on
the same pair is allowed: the exclusion mask concerns chemical connections.
The module binds to the canonical neuron order and directed chemical topology;
attaching it to a different topology fails.

Each candidate uses the existing presynaptic gate and contributes
`w * gate[pre] * (E - V[post])` to the postsynaptic current. No anatomical synapse
count is invented. Strength is `softplus(raw_strength) + 1e-9`; reversal is
`2 * sigmoid(raw_sign) - 1`. Declared strengths must exceed the numerical floor
and reversals must lie strictly inside (-1, 1). Group names tie both parameters.
When neuromodulation is enabled, the postsynaptic synapse multiplier also applies
to these connections.

The regularizer is `l1_strength * sum(w)` over **edges**, including each occurrence
of a tied group. It is not averaged over edges, targets, or samples. Its gradient
passes through the physical-strength transform. The positive floor means weights
can shrink toward zero but are never exactly zero; this is not a hard pruning or
candidate-selection algorithm. The explicit budget limits storage and connectivity.

## Validation and remaining integration

`test_dark_edges.py` checks independently calculated currents and penalties,
unchanged source anatomy, empty-mode baseline equivalence, tied-edge multiplicity,
invalid candidate masks, topology binding, composed rollout/penalty gradients
against finite differences, and an Optax step that reduces the composed loss.
The default model remains unchanged when this option is omitted.

The [extended fit runner](EXTENDED-FITTING.md) now serializes declarations and
learned parameters, adds this penalty once, and submits predictions to Rust
scoring. The native AtlasModel simulator cannot execute the extension and rejects
its separate checkpoint envelope. Candidate selection must use training evidence
only, with validation-selected budgets and penalties; test responses must not
select candidate edges. Native protocol operations are not yet available through
this JAX API. Biological validation remains open.
