# Performance and compact representations

## Machine and measurement contract

Measured locally on an Apple M4 Pro: 12 CPU cores, 16 GPU cores, 24 GiB unified
memory, aarch64 macOS. Rust 1.97.1; release profile, thin LTO, one codegen unit,
`RUSTFLAGS='-C target-cpu=native'`. The current solver uses **one CPU thread**.
The GPU has not been used or benchmarked. See `benchmark-environment.json` for
compiler, hardware, source, lockfile, and binary fingerprints.

Forward timings include parameter preparation, workspace allocation, integration
and saved output allocation; they exclude graph import/compilation, executable
startup and compilation. Three repetitions follow a short warm-up. This is a
local engineering benchmark, not a controlled cross-project speed comparison.
The forward test runs 100 simulated seconds, dt=1 ms, RK4, and saves every second.
Defaults are uncalibrated and this test has no stimulation. A separate trace-codec
workload uses a two-second pulse trial, with 401 saved fluorescence frames.

Numerical correctness gates: analytical passive-cell response, exact event
boundaries, symmetric-gap conservation, independent per-edge gate comparison,
all-parameter AD/finite-difference comparison, masked observations, and lossless
codec/corruption checks. Faster layouts must continue to pass these gates.

## Measurements

Exact receipts: [synthetic](benchmark-m4-pro.json),
[c302 anatomy](benchmark-c302-m4-pro.json), and the earlier
[edge-scatter baseline](benchmark-m4-pro-edge-scatter.json).
The synthetic graph has 302 nodes, 6,946 chemical edges and 906 gap pairs.
The c302 import has 302 nodes, 3,638 chemical edges and 1,080 gap pairs.
Edge counts are distinct aggregated neuron pairs, not total synapse counts.

| Workload | Median forward wall time | Canonical JSON | Zstd JSON | WSC1 |
| --- | ---: | ---: | ---: | ---: |
| Synthetic | 3.100 s | 1,285,366 B | 16,037 B | 1,816 B |
| Imported c302 anatomy | 2.524 s | 1,395,996 B | 30,674 B | 14,382 B |

The earlier synthetic edge-scatter median was 5.258 s; the incoming-row layout is 1.70× faster in these local runs.

| Trace workload | Raw f64 | Zstd raw f64 | WST1 |
| --- | ---: | ---: | ---: |
| Synthetic | 968,816 B | 660,885 B | 646,626 B |
| Imported c302 anatomy, illustrative parameters | 968,816 B | 885,989 B | 729,925 B |


The synthetic graph is unusually repetitive; its compression ratio should not
be projected onto biological data. Both graph measurements include provenance
and annotations. The raw trace baseline is little-endian row-major f64 compressed
using the same Zstd level. WST1 adds missing-data support, chunk boundaries and
checksums; it does not promise to beat every general codec on every signal.
Codec encode/decode timings are single samples, not statistical estimates.

The spec's <1-second **GPU** target has not been demonstrated. Full-gradient
performance has also not been demonstrated: forward-mode audit gradients run
one simulation for each selected parameter. The synthetic fitting example is
small-circuit engineering evidence only.

## Runtime representation

- Cold, canonical metadata remains outside the numerical loops. Source neuron
  order is lexicographic. Parameters use this canonical edge ordering.
- Runtime chemical edges are sorted by postsynaptic cell in incoming sparse
  rows; each target accumulates into a local scalar. A permutation maps learnable
  parameters to runtime edges. u16 source indices suffice for this organism.
- Each undirected gap pair is stored once; its current is added/subtracted at its
  endpoints. No dense 302×302 voltage-difference matrix is formed.
- Each neuron has one voltage, one calcium state, and one presynaptic gate.
  Since all outgoing edges obey the same initial condition and gate ODE, those
  gates are identical by uniqueness of the ODE solution. Sharing them is exact
  under the present assumptions. For the synthetic workload, this reduces state
  from 7,550 to 906 f64 values (60,400 to 7,248 bytes).
- RK4 stage arrays and release scratch are allocated once and reused. Positive
  parameter transforms are evaluated once per rollout. RHS calls allocate zero
  heap objects. Save frames still allocate; output storage grows with saved time.

Extending the model with heterogeneous receptor kinetics must allocate one gate
per `(presynaptic neuron, kinetic class)` or revert to edge states. Do not apply
this optimization indiscriminately to imported detailed models.

## WSC1 graph container

This is an application-specific format combining standard encoding primitives:

1. Four-byte magic `WSC1`.
2. Eight-byte little-endian uncompressed payload size (maximum 64 MiB).
3. 32-byte SHA-256 of the uncompressed payload.
4. Zstd level-3 payload.

The payload holds unsigned LEB128 metadata length, JSON metadata, chemical edge
count, gap count, followed by sorted edge records. Metadata stores canonical
neurons plus a dictionary of repeated provenance/receptor records. Each edge
key is `pre*N + post` (for a gap, canonical `a*N + b`); unsigned deltas encode
successive keys. Counts/sizes and sign priors encode their f64 bits XORed with the
previous value in their column, using unsigned LEB128. Dictionary indices are
unsigned LEB128. Chemical records contain key/count/sign/metadata; gap records
contain key/size/metadata. Predictors reset between chemical and gap sections.

Decoding validates bounds, lengths, checksums, endpoints, duplicates and numerical
constraints before compilation. Numeric edge values are bit-exact; JSON numeric
metadata uses serde_json's float-roundtrip support. Canonical graph hashes are
computed from canonical serialized graphs, independent of the packed container.
Metadata list order remains significant for that hash.

## WST1 trace container

1. Magic `WST1`, u32 little-endian row count, u32 column count.
2. Independent chunks of up to 256 time rows, in order.
3. Per chunk: u32 compressed length, 32-byte SHA-256, Zstd level-3 payload.
4. The digest covers magic, total rows, columns, starting row (all dimensions as
   u32 little-endian), and uncompressed payload, detecting shape reinterpretation.

Each payload begins with a presence bitmap, indexed `column*chunk_rows + row`,
least-significant bit first. Present values follow in column/time order as
unsigned LEB128 of the f64 bits XORed with the previous present value. Predictors
start at zero for each column in each chunk. Missing values consume no numeric
code and leave the predictor unchanged. Signed zero is preserved; NaN/infinity
are rejected (missingness is explicit). Current allocation limit: 16 million
matrix entries. More capacity requires a streaming interface.

`decode_range` skips unrelated compressed chunks, validates all chunk framing,
and checks payload hashes only for decoded chunks. It scans chunk headers to
locate a range; it has no persistent index yet. The caller currently supplies the
archive bytes, so this is selective decompression, not memory-mapped I/O.
The simulation CLI uses columns `[voltage..., fluorescence...]` and writes a
JSON sidecar containing times, neuron order, parameters and configuration.

## Next measured experiments

1. **Reverse gradients:** implement checkpointed discrete adjoints; match the
   existing AD audit before interpreting any training speedup.
2. **Apple GPU:** batch independent trials in shared-memory Metal buffers, test
   fused multi-step dispatch against CPU for batch sizes 1/16/64. Do not assume
   a GPU wins for one tiny 302-neuron step.
3. **Precision:** compare f32/mixed precision with f64 at matched trajectory and
   gradient error. No lossy quantization or fast-math shortcuts enabled yet.
4. **Integration:** stiff/adaptive and exponential/IMEX alternatives compared at
   equal error, not merely equal dt. Long-range behavior needs step convergence.
5. **Memory:** streaming losses and chunk writers, checkpoint/recompute sweeps,
   shared immutable graph metadata across ensembles, and a mapped chunk index.
6. **Storage:** compare WSC/WST with column-major raw Zstd, bitshuffle, additional
   codecs, and real recordings; choose measured size/decode-time tradeoffs.

Minimum bytes and minimum latency are separate objectives. Packed archival data
is decoded once into arithmetic-friendly arrays; the integrator does not decode
varints or decompress data during timesteps.
