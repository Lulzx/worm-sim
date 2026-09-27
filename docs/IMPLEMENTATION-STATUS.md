# Specification implementation and acceptance ledger

The active objective is the complete [specification](../SPEC.md), with Rust
replacing its proposed Python/JAX core at the user's request. Optional Taichi
acceleration remains separately audited. A software feature, a synthetic numerical
check, and a successful biological benchmark are distinct acceptance levels.
This ledger does not declare scientific targets achieved merely because code exists.

| Spec area | Current evidence | Work required for full acceptance |
| --- | --- | --- |
| Data (§3) | Canonical identities, aliases/provenance, content-hashed c302 anatomy, WSC1/WST1 codecs, pinned Creamer operators; native HDF5 import of 21 freely moving animals with behavior | Direct Cook/Witvliet variation, CeNGEN, peptide network, atlas trial recordings, further moving cohorts/preprocessing validation; graph exports |
| Neurons (§4) | Differentiable Level 0 | Mixed Level 1 conductance channels and published Level 2 adapters with source-specific tests |
| Coupling (§5) | Anatomical chemistry, relaxed signs, symmetric gaps, shared equivalent gates | Dark edges/penalty, plasticity, rectification, peptide/monoamine compartments and receptor modulation |
| Numerics (§6) | Euler/RK4, exact events, Rust forward AD, Taichi reverse AD, segment checkpoints | Stiff/adaptive and multirate solvers, recursive checkpoints/continuous adjoint, truncated training, heterogeneous batches, multi-GPU orchestration |
| Parameters (§7) | Positive/bounded transforms, named parameter groups, explicit class maps and optional L/R suffix sharing, raw shrinkage and graph-sign priors | Source-backed class/transmitter/receptor annotation, hierarchical animal offsets, annealed discrete choices |
| Training (§8) | Masked confidence-weighted MSE, small synthetic Adam fit; full-network history-only shooting and block-projected Kalman state inference | Two-epoch shooting and filtering fits completed; validation rejected trained epochs in both (LEVEL0-FIT.md, LEVEL0-FILTER.md); correlation/classification/behavior losses, AdamW schedules/group rates, ensembles/Laplace/SBI, held-out evidence |
| Perturbations (§9) | Current pulses, silence, ablation; JSON protocols | Waveforms/conductance input, gene mappings, drug/modulator input, voltage clamps, YAML, source-backed phenotype registry |
| Body (§10) | Not implemented | Motor/muscle map, differentiable reduced body, sensor/environment interface, replay and Sibernetic adapter |
| Tasks 1–2 (§11) | Pinned pretrained Creamer inference reproduced; Rust group-disjoint splits and common Task 1/2 scorers implemented; upstream Creamer split differs from requested held-out neurons | Controls, AR(1), animal bootstrap and a fitted stable latent LDS are scored (LATENT-LDS.md); shooting and filtering Level 0 population fits failed validation; useful Level 0 forecasting, GRU and atlas neuron split remain |
| Tasks 3–5 (§11) | Not demonstrated | Behavior decoder comparison, mutant/ablation direction scoring, untrained closed-loop gait statistics |
| Performance (§12) | Native CPU/Metal and storage receipts; checkpoint tradeoff measured | Spec workload targets: GPU 100 s <1 s, batch 64 gradient <2 s, atlas fit <12 h, ensemble runs |
| Engineering (§12) | Public MIT repository, local/CI regression tests, manifests and receipts | Complete tutorial notebook, experiment tracking adapter, package/API documentation and reproducible full benchmark runs |

## Build sequence and completion gates

1. Prioritize Task 2 fitting: controls and animal-level intervals, stable LDS,
   302-neuron initial-state inference, tied parameters/sign priors, then a real
   Level 0 fit through the same scorer. Defer backend/performance expansion.
2. Add GRU and equal behavior inputs, then ingest the Task 1 atlas and publish
   its held-out stimulated-neuron split; complete staged comparisons.
3. Extend coupling with slow modulation; compare against the same held-out tasks.
4. Add mixed-fidelity neuron models and stiff/multirate numerical support.
5. Complete perturbation protocols and the measured-phenotype registry.
6. Add reduced-body/replay/high-fidelity interfaces and Tasks 3–5.
7. Optimize measured bottlenecks, execute uncertainty/performance batteries,
   document reproducibility, and close every specification requirement individually.

Performance claims must specify workload, hardware, precision, timing boundaries,
and excluded storage. Benchmark success requires actual held-out data and named
baselines. Unknown metadata, missing measurements, unavailable adapters, or failed
thresholds remain visible; none may be replaced by synthetic success claims.
