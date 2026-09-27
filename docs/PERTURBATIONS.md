# Perturbation protocols

The Rust solver accepts `solve::Config` serialized as JSON or YAML. Run the waveform
example with:

```sh
cargo run --release -- simulate data/c302-herm.wsc examples/aval-waveform.yaml runs/aval-waveform.wst
```

`current_waveform` specifies one canonical neuron, absolute `times` in seconds,
and corresponding signed `amplitudes` in the model's current units. Values are
linearly interpolated between knots and zero outside their support. At least two
finite, strictly increasing times are required, all within the simulation interval.
Amplitudes must be finite and match the number of times. Negative values inject
inhibitory current; they do not change synaptic signs.

Waveforms and rectangular `stimulate` events add. Existing `silence` and `ablate`
semantics are retained: silence blocks outgoing release, while ablation removes
the cell's dynamics and incident coupling. Silence does not block injected current.

The integrator splits at every waveform knot, event boundary and save time. RK4
uses time-varying current at each stage; a stage ending at a support boundary uses
the left limit, and the next step uses the right limit. Euler uses the left-endpoint
current. This avoids smearing nonzero endpoint jumps across the event boundary.
Neuron indices and boundary times are resolved once before integration. No dense
current timeline is stored; interpolation uses the supplied knots.

Tests cover an analytic passive-cell ramp response, exact rectangular-pulse
compatibility under Euler/RK4, signed overlapping inputs, malformed protocols,
JSON round trips and forward parameter derivatives against finite differences.
Waveform knots/amplitudes are fixed `f64` protocol data; differentiating those
controls or event times is not implemented. Gradients through neural parameters
and initial state remain available through the generic scalar solver.

This implements configurable current waveforms in specification §9. Gene/drug mappings and the published phenotype
registry remain separate requirements. It does not establish optogenetic pulse
calibration or any biological phenotype match.


YAML files use `.yaml` or `.yml` (case-insensitive). Other filenames retain JSON
parsing. `protocol::parse` accepts an explicit format for in-memory callers.
Both formats use the strict Serde schema and the same simulation validation;
there is no retry in another format after a parse error. YAML supports comments
and block mappings through the standard [serde_yaml_ng parser](https://docs.rs/serde_yaml_ng/0.10.0/serde_yaml_ng/).
Unknown fields, duplicate configuration keys and multiple documents are rejected.
Neurons and event intervals are checked against the graph before integration.
The existing JSON examples remain supported. Tests verify YAML/JSON trajectory
identity; output metadata retains the normalized configuration, independently of
its source format. New perturbation operations are still required to complete §9.

## Conductance stimulation

`conductance_waveform` supplies `neuron`, strictly increasing absolute `times`,
nonnegative `conductances`, and a fixed finite `reversal`. The waveform is zero
outside its support and linearly interpolated at each integration stage. Example:

```sh
cargo run --release -- simulate data/c302-herm.wsc examples/aval-conductance.yaml runs/aval-conductance.wst
```

The voltage equation adds `g(t) * (E - V) / tau`. Conductance is relative to the
model's leak conductance and reversal uses normalized voltage units. Its effect
can depolarize, hyperpolarize or shunt, depending on voltage and reversal; a
conductance is never represented by a negative g. For overlapping inputs the
solver stores `sum(g)` and `sum(g * E)` per cell. Current and conductance inputs
combine; ablation overrides both, while silencing still only blocks output.
Constant two-knot waveforms implement rectangular conductance pulses, including
nonzero endpoint jumps. No receptor kinetics or experimentally calibrated
optogenetic conductance is implied.

The forward scalar-generic Rust dynamics preserve derivatives with respect to
neural parameters and state. Tests check the exact passive-cell response to
multiple overlapping conductances and the exact initial-voltage derivative under
a ramp (integrated shunting). Waveform values and reversal are fixed f64 protocol
inputs. The atlas-fitting adjoint and JAX atlas replay currently accept their
existing current-kernel inputs; conductance protocol fitting is not wired into
those interfaces.

## Prescribed voltage traces

`voltage_clamp` supplies `neuron`, absolute `times` and matching finite `voltages`
in the model's normalized voltage units. At least two strictly increasing knots
are required. Run `examples/aval-clamp.yaml` using the same `simulate` command.
A recorded voltage trace can be used directly after explicit time/unit alignment;
calcium fluorescence is not a voltage trace and is not silently converted.

The voltage is replaced by linear interpolation at every Euler/RK4 stage while
calcium, gates and connected neurons evolve normally. The constrained voltage
has no free ODE update. Steps split at all knots. A clamp starts with an immediate
voltage reset; samples at its onset show the new value. Its final value is retained
at release, after which the neuron resumes its own dynamics. Adjacent clamps are
allowed: the new clamp wins at a shared boundary, independently of protocol order.
Overlapping clamps on one cell and clamping an ablated cell are rejected. Current
and conductance injections cannot override a clamped voltage; silencing can still
suppress that cell's output.

Clamped values are fixed f64 observations, so voltage derivatives with respect to
free initial voltage and neural parameters are zero during a clamp. Parameters
controlling calcium, gates and other neurons remain differentiable. No derivative
with respect to the supplied trace values is exposed. The implementation uses
stage projection, not a high-gain penalty or an inferred clamp current. Tests check
onset/ramp/release, adjacent intervals, passive gap-neighbor response to constant
and ramp commands, and calcium value/gradient against analytic solutions.
These native protocol clamps are not yet integrated with the JAX fitting API.
