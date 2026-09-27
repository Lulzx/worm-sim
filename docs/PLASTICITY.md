# Short-term synaptic plasticity

Specification §5's optional depression/facilitation is implemented in the JAX
Level 0 API. Types and edge assignments must be declared; none are inferred from
names, transmitter annotations, or missing measurements. This feature has synthetic
numerical checks, not a fitted C. elegans parameter set.

## Equations and modeling choice

The starting point is the utilization/resource model in [Wang et al., equations
6–8](https://pmc.ncbi.nlm.nih.gov/articles/PMC7947117/). That paper uses spike
arrivals. WormSim explicitly substitutes a continuous rate for graded release:
`q = rate_scale * release[pre]`, with `rate_scale` in inverse seconds. This is a
phenomenological adaptation, not evidence that those spike-based parameters apply
to worm synapses.

For each presynaptic neuron/type combination:

```text
dx/dt = (1 - x) / tau_depression - u * x * q
du/dt = (U - u) / tau_facilitation + U * (1 - u) * q
chemical_current = anatomical_or_extra_strength * x * u * gate[pre] * (E - V[post])
```

`x` is available resource and `u` is utilization. The `both` mode uses both
equations. `depression` fixes `u = U`; `facilitation` fixes `x = 1`. Initial values
are `x = 1`, `u = U`, recomputed from current parameters for every solve, so reverse
AD includes the initial-state dependence. Unforced preparation carries both
states into the trial. Other chemical edges retain multiplier one.

`U` uses a sigmoid; positive time constants and rate scales use softplus plus
`1e-9`. Initial declarations require `0 < U < 1` and positive values above that
floor. Enabled currents use `x*u`, without normalization by U: anatomical weights
therefore represent maximum conductance, not resting effective conductance.

For nonnegative rates, the ODE points inward at the boundaries of `[0,1]^2`.
Numerical solvers are not guaranteed to preserve this invariant at coarse steps;
there is no clipping that would silently alter the model or its gradients. Use
appropriate step sizes/tolerances and check trajectories.

## Configuration

```python
from plasticity import Plasticity
from level0 import Level0, parameters

spec = {
    "schema_version": 1,
    "source": "synthetic example; no biological assignment",
    "types": [{
        "id": "example", "mode": "both", "utilization": 0.3,
        "tau_depression": 0.4, "tau_facilitation": 0.2, "rate_scale": 3.0
    }],
    "edges": [{"pre": "A", "post": "B", "type": "example"}]
}
plasticity = Plasticity(graph, spec)
engine = Level0(model, graph, times, plasticity=plasticity)
theta = parameters(model, plasticity=plasticity)
```

The example requires an A → B chemical edge. For off-connectome candidates, pass
the same `DarkEdges` instance to both `Plasticity(..., dark_edges=extra)` and
`Level0(..., dark_edges=extra)`, and to `parameters`. Both kinds of connections can
share a type. Duplicate/missing edges, unknown/unused types, invalid values, and
mismatched topology are rejected. Modulation's postsynaptic synapse multiplier
composes with the plasticity multiplier; gap currents are unaffected.

## State representation and validation

K selected edges require two states per unique `(presynaptic neuron, type)` pair,
not two per edge. Edges sharing this pair have identical drive, parameters, and
initial states, making the reduction exact for this model. Different presynaptic
neurons retain independent states even if parameters are tied. Target-specific
resource pools or edge-specific initial conditions would invalidate this reduction
and are not supported. Type parameters occupy four raw scalars per type; inactive
mode parameters have zero gradient and should be frozen in a future fit interface.

A [storage audit](plasticity-storage.json) using the imported c302 graph and a
**synthetic single type on all 3,638 chemical edges** produces 292 source/type
slots: the float64 state vector occupies 4,672 bytes versus 58,208 bytes for two
states per edge. These are state-vector bytes only, excluding topology maps,
parameters, solver/AD workspace, and runtime memory. This is not a speed claim or
a biological type assignment; additional types increase the number of slots.

`test_plasticity.py` checks constant-drive analytic depression/facilitation,
zero-drive recovery, inward boundary derivatives, source/type sharing against
independent per-edge equations, unchanged empty-mode behavior, invalid mappings,
combined modulation/extra-edge currents, and finite-difference gradients through
the full prepared response. Euler step refinement is compared with a tight-tolerance
Kvaerno5 solution. The [extended fitting pipeline](EXTENDED-FITTING.md) additionally trains,
serializes, reloads, and submits these dynamics to independent Rust scoring.
Source-backed type assignments and biological validation remain open.
