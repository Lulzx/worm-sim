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

This implements configurable current waveforms in specification §9. Conductance
waveforms, voltage clamps, gene/drug mappings and the published phenotype
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
