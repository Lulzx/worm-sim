# WormSim — Rust foundation

Native Rust implementation of the first build stage in [SPEC.md](SPEC.md).
The specification's Python/JAX stack is replaced by Rust at the user's request.
This is a tested Level 0 Rust foundation with an experimental Taichi Metal
backend and a reproduced pretrained linear baseline, not a fitted whole-worm model.

## Run

```sh
cargo test
cargo clippy --all-targets -- -D warnings
RUSTFLAGS='-C target-cpu=native' cargo build --release --examples --bins
./target/release/examples/fit_small
mkdir -p runs
./target/release/wormsim simulate data/c302-herm.wsc examples/aval-pulse.json runs/aval.wst
./target/release/wormsim bench
./target/release/wormsim bench data/c302-herm.wsc
```

Create `runs/` first if starting from a fresh checkout. `.wst` output writes a
compressed trajectory plus a `.wst.json` run manifest. A `.json` output path
writes readable voltage/fluorescence arrays. The manifest includes graph hash,
fixed neuron ordering, configuration, raw parameters, package version, and seed.
This implementation is deterministic. Set `WORMSIM_COMMIT` at build time to
record a revision; otherwise the manifest explicitly says `unversioned`.

`fit_small` fits one chemical strength to synthetic fluorescence with exact
forward-mode automatic differentiation and Adam, then checks a different pulse.
It demonstrates plumbing, not biological validation or parameter identifiability.

## Implemented

- Validated, versioned graph schema and canonical ordering; SHA-256 graph IDs;
  explicit alias reconciliation; provenance and confidence; missing trace masks.
- Level 0 graded neurons; anatomical chemical conductances and relaxed reversal
  signs; symmetric gap junctions; learned calcium time constants and scales.
- Euler and RK4 with exact event boundaries; stimulation/inhibition through
  signed current, piecewise-linear current waveforms, silencing, and ablation.
  Declarative protocols are JSON; see [perturbations](docs/PERTURBATIONS.md).
- Softplus constraints and forward-mode AD for every implemented parameter;
  confidence-weighted fluorescence loss and a small-circuit Adam example.
- Compact incoming sparse rows, u16 indices, shared presynaptic gates, contiguous
  stage workspace, and no allocation in RHS evaluations or ordinary steps.
- WSC1 graph and WST1 trajectory codecs, checksums, lossless numerical storage,
  selected-window trace decoding, and measured compression baselines.
- Pinned c302 identity/anatomy import and [upstream audit](docs/UPSTREAM-REVIEW.md).

- Pinned [Creamer linear baseline](docs/BASELINE.md): complete inference operators,
  compact export, Rust evaluation, and independent NumPy parity.
- Experimental [DiffTaichi-style reverse gradients](docs/TAICHI.md) on CPU and
  Apple Metal, audited against Rust for all 10,169 parameters of the 302-neuron
  anatomy, using short synthetic-target trials.
- [Checkpointed reverse differentiation](docs/CHECKPOINTING.md), audited over
  1,024 steps on CPU/Metal; trades recomputation for smaller state storage.

- Native [WormWideWeb HDF5 import](docs/WORMWIDEWEB.md): 21 labeled baseline
  animals, fixed animal splits, behavior channels, [trivial and AR controls](docs/TASK2-FITTING.md),
  animal-bootstrap uncertainty, a fitted [stable latent LDS](docs/LATENT-LDS.md)
  and a [connectome-free GRU](docs/GRU.md). The [source audit](docs/PREPROCESSING-AUDIT.md)
  identifies this benchmark as retrospective whole-recording-normalized signals.
- Native [Randi atlas trial ingestion](docs/RANDI-ATLAS.md): 3,166 individual
  stimulation trials, a fixed held-out-neuron split, and independent sample checks.
  [Published pair-level labels](docs/ATLAS-CLASSIFICATION.md) are independently
  checked; Task 1 model fitting and comparison remain outstanding.
- Full-network [history state inference](docs/INITIAL-STATE.md), tied-parameter
  [Level 0 population fitting](docs/LEVEL0-FIT.md), and an approximate
  [history filter](docs/LEVEL0-FILTER.md). Scientific acceptance remains open.

## Scientific and numerical conventions

Time is seconds. Voltages and currents are normalized consistent units, **not**
calibrated mV/pA. Default parameters are illustrative. The imported anatomy has
unknown sign priors (0.5), neuron classes, sides and types; do not interpret a run
as a validated C. elegans prediction. Calcium is a first-order low-pass of
sigmoidal release, scaled into arbitrary fluorescence units, not calibrated ΔF/F.
Initial voltage is the leak rest value; gates and calcium start at the matching
single-cell steady state, not a solved coupled-network equilibrium.

Ablation removes chemical/gap interactions and freezes membrane voltage.
Silencing clamps transmitter output and outgoing chemical current, while leaving
electrical coupling intact. Event intervals are `[start,end)`. Save grids include
zero and the exact final time. Integration steps split at saves and events; a
final RK stage uses the forcing from the interval being integrated.

Shared gates are mathematically exact only with the currently shared synaptic
kinetics and consistent initial gates. Heterogeneous edge kinetics, receptor
classes or short-term plasticity require separate source/kinetic-class states.
Parameter ordering stays canonical even when runtime edges reorder by target.

## Reproduce the imported anatomy

```sh
python3 scripts/fetch_c302.py
./target/release/wormsim import-c302 runs/upstream/herm_full_edgelist.csv \
  data/c302-neuron-ids.json 6cd861f8ca4d3241ee9cf4627884caa930dab53c \
  data/c302-herm.wsc --mean-mirrors
```

The stdlib-only downloader parses, but never executes, upstream Python. Simulation,
import validation, compression, fitting, and tests are Rust. Source URLs and
hashes are in `docs/c302-fetch-receipt.json`; upstream MIT notice is retained.
The import preserves all 302 canonical identities, including isolated neurons.
Numbered-neuron zero-padding aliases (for example DA01 → DA1) are mapped and
logged. Remaining endpoints outside the canonical manifest are excluded and listed. Self gaps carry zero current and
are excluded with a count. Equal mirrored electrical rows become one undirected
edge; unequal mirrors require `--mean-mirrors` (recorded in the import report).
Use `--strict` to reject those conflicts. Repeated rows in the same direction sum.
Chemical signs/types are not inferred from anatomy or names.

## Specification progress

[Implementation ledger](docs/IMPLEMENTATION-STATUS.md) tracks the complete
specification and its acceptance gates. [Benchmark contracts](docs/BENCHMARKS.md)
provide Rust neuron/animal splits, leakage checks, trace/response metrics, and
1/10/30-second forecast scoring. Real-data benchmark success remains unproven.

## Remaining work

The GPU performance targets and Tasks 1–5 in the spec are **not met**. Forward AD
needs one rollout per selected parameter and is a reference/audit path, not the
full-scale training implementation. Full-network Metal training, streaming observations,
stiff/adaptive solvers, class/hierarchical tying, calibrated observation models,
real recording ingestion and held-out benchmarks, peptides, mixed fidelity,
YAML protocols, mutants/drugs/clamps, uncertainty, and body feedback remain.
Output accumulation is still in memory; incremental trajectory writers and
on-the-fly loss accumulation are needed for long recordings.

See [performance and formats](docs/PERFORMANCE.md) for measurements, representation
contracts, and the next optimization experiments. See [upstream review](docs/UPSTREAM-REVIEW.md)
for reusable work and explicit evidence boundaries.
