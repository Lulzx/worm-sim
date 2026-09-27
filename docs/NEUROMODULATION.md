# Slow compartmental modulation

The JAX Level 0 engine supports the slow layer in specification §5 as an optional
extension. It adds one concentration state per declared species/compartment pair:

`tau[p] * dc[p]/dt = sum(alpha[j,p] * release[j]) + bath[p] - c[p]`.

The release used here is the neuron's sigmoid output, including any gain
modulation. Concentrations feed back into the fast network. No peptide or
monoamine maps are inferred from the connectome, neuron names or missing data.
The caller must supply a versioned map and source provenance. The supplied
[example](../backends/jax/examples/modulation-synthetic.json) is explicitly
synthetic and is not evidence for a worm signaling pathway.

```python
from modulation import Modulation
from level0 import Level0, parameters, response
from solvers import Adaptive
import jax.numpy as jnp

# names must be the exact canonical order used by the model.
slow = Modulation(names, modulation_spec)
engine = Level0(model, graph, times, Adaptive(method="kvaerno5"), slow)
theta = parameters(model, slow)
fluorescence = response(engine, theta, jnp.asarray(target_index))
# engine.solve(theta, target_index).ys also exposes the concentration states.
```

## Representation and parameter semantics

Release and receptor connections are sparse index arrays. Compartments are
independently well mixed; there is no implied transport or diffusion between
them. Multiple species use the same equations. Positive time constants, release
strengths and receptor dissociation scales use softplus plus 1e-9. Initial physical
values are transformed inversely so they retain their declared values. Named
release groups tie alpha; named receptor groups tie dissociation scale and signed
sensitivity. Conflicting initial values for a tied group, unknown cells/channels,
duplicate rows, invalid values and unrecognized fields are rejected.

Receptor activation is `a = max(c,0)/(Kd + max(c,0))`. The max handles small
numerical undershoots in a solver; the concentration ODE itself is not clipped.
For each neuron and affected parameter family, sensitivities accumulate as
`z = sum(beta * a)`. The multiplier is `exp(L * tanh(z/L))`, with explicit
`max_log_effect = L` (default 3). This smooth engineering bound keeps the
multiplier positive and limits runaway modulation; it is not a measured
physiological range. The families are:

- `gain`: multiplies the sigmoid slope.
- `leak`: multiplies the leak current toward the original resting potential.
- `synapse`: multiplies incoming chemical weights at the receptor-bearing cell.

Gaps, reversal potentials and time constants are otherwise unchanged. With all
sensitivities zero, every multiplier is exactly one and the fast model agrees
with the unmodulated path. Bath is a nonnegative constant contribution to the
concentration's equilibrium drive, expressed in the same normalized units as the
release sum. It is not a calibrated drug dose or a timed pharmacology protocol.

State order is voltage, calcium, synaptic gates, then concentration channels
in declared order, followed by optional plasticity states. `theta['modulation']` contains named arrays `raw_tau`,
`raw_release`, `raw_kd` and `sensitivity`. It is an ordinary JAX pytree. The existing
Euler and optional Diffrax solvers carry the coupled state and its gradients
through unforced preparation and stimulus intervals. No hand-written adjoint is
needed. The [extended fitting checkpoint](EXTENDED-FITTING.md) preserves both
the original specification and learned arrays, and Rust scores its predictions.
The native simulator does not execute this extension.

## Checks and remaining requirements

Tests cover the exact constant-release/bath concentration solution, zero-feedback
parity with the existing Level 0 engine, sparse/local and bounded receptor
effects, reverse gradients of all four slow parameter families against finite
differences, and invalid map/tie handling. Test maps and parameter values are
synthetic. No biological fit, phenotype, or improved forecasting is claimed.

Remaining acceptance work includes source-backed peptide/monoamine maps,
receptor-specific priors, timed drug/gene perturbations, multirate integration, spatial transport
if needed, and held-out comparisons against the model without modulation.
