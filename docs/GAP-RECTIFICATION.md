# Optional gap rectification

The JAX Level 0 engine supports the optional rectification flag in specification
§5. The default remains the original symmetric ohmic gap current. No gap pairs
are added, and no innexin composition or preferred direction is guessed.

For a canonical pair `a < b`, define `delta = V[b] - V[a]`. A selected edge uses:

```
rho = tanh(asymmetry[group])
factor = 1 + rho * tanh(delta / voltage_scale[group])
I_into_a = base_gap_conductance * junction_size * factor * delta
I_into_b = -I_into_a
```

Positive asymmetry favors current into a when b is more depolarized; negative
asymmetry favors the opposite voltage polarity. Both endpoints use the same
instantaneous current, preserving current conservation. The factor lies in
[0,2] including floating-point saturation, so `I_into_a * delta >= 0`: the junction
does not create electrical energy. Zero asymmetry recovers the existing model.
The orientation must match the canonical graph; reversed, missing or duplicate
pairs are rejected instead of silently changing the parameter's meaning.

`voltage_scale` is a fixed declared scale in normalized voltage units. Only the
signed asymmetry is learned, avoiding redundant fitting of an asymmetry/scale
ratio. Named groups share asymmetry across selected pairs. Conflicting initial
values or scales within one group are rejected. Unselected pairs retain factor
one. This smooth law is an implementation choice, not a validated model of a
particular innexin or a measured rectification ratio.

```python
from rectification import GapRectification
from level0 import Level0, parameters, response
import jax.numpy as jnp

spec = {
    "schema_version": 1,
    "source": "Synthetic mechanism example; not biological evidence",
    "edges": [{"a": "A", "b": "B", "group": "example",
               "asymmetry": 0.4, "voltage_scale": 0.2}]
}
rect = GapRectification(graph, spec)
engine = Level0(model, graph, times, rectification=rect)
theta = parameters(model, rectification=rect)
result = response(engine, theta, jnp.asarray(target_index))
```

The chosen pair must exist in `graph`; A/B here are synthetic identifiers. The
same option works with the adaptive solver and slow-modulation API. Parameters
are a JAX pytree under `theta['rectification']['asymmetry']`; ordinary reverse AD
handles the new mechanism. No separate hand-derived gradient is needed.

Tests check conservation and dissipation at both voltage polarities, zero-effect
trajectory parity, reverse gradients against finite differences, sparse edge
selection, tying and invalid-map handling. Biological innexin assignments,
mutation mapping, priors and held-out evaluation remain unimplemented. The
separate [extended population runner](EXTENDED-FITTING.md) saves this
extension and uses Rust to score JAX predictions. Native model loading rejects
the extended checkpoint rather than dropping rectification.
