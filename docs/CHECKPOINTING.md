# Checkpointed reverse differentiation

The optional Taichi backend now retains one differentiable state window plus
primal checkpoints. Reverse evaluation replays each window once, carries the
terminal voltage/calcium/gate adjoint into the preceding window, and accumulates
parameter derivatives. It computes the complete first-order gradient; it does
not truncate backpropagation or quantize the differentiable state.

Rust generates the graph, parameters, trajectories, and independent forward-AD
reference. `CheckpointLevel0` reuses the audited Euler kernel. CPU keeps dynamic
CSR traversal; Metal keeps the static slots required by its gradient regression.

## Reproduce

Using the environment from [TAICHI.md](TAICHI.md):

```sh
mkdir -p runs
cargo run --locked --release --example export_taichi_reference -- runs/c302-1024.json data/c302-herm.wsc 1024
.venv-taichi/bin/python backends/taichi/audit.py runs/c302-1024.json --arch cpu --checkpoint-steps 23 --validate --output runs/checkpoint-validation.json
.venv-taichi/bin/python backends/taichi/audit.py runs/c302-1024.json --arch cpu --checkpoint-steps 23 --output runs/checkpoint-cpu.json
.venv-taichi/bin/python backends/taichi/audit.py runs/c302-1024.json --arch cpu --output runs/full-cpu.json
.venv-taichi/bin/python backends/taichi/audit.py runs/c302-1024.json --arch metal --checkpoint-steps 23 --output runs/checkpoint-metal.json
```

The exporter accepts an optional step count after the graph path; `-` selects the
small synthetic graph. `--checkpoint-steps 0` retains the original full tape.
Positive intervals exceeding the trajectory length are clamped to that length.

## Measured result on the M4 Pro

The imported anatomy has 302 neurons and 10,169 parameters. All derivatives,
voltage/fluorescence frames, and masked loss pass against Rust for **1,024 Euler
steps (1.024 seconds)**, with synthetic targets and illustrative parameters.
Five repetitions follow the first reverse evaluation; validation uses three.
The final partial window contains 12 steps and is checked too.

| Path | Median forward + reverse | Maximum gradient error |
| --- | ---: | ---: |
| CPU f64, full tape | 131.40 ms | 6.39e-16 |
| CPU f64, 23-step checkpoints | 173.66 ms | 6.39e-16 |
| CPU f64, checkpoints + access validator | 614.09 ms | 6.39e-16 |
| Metal f32, 23-step checkpoints | 552.26 ms | 4.55e-9 |

CPU uses one thread. Metal fallback is disabled. Timings include dispatch,
checkpoint construction, recomputation, gradient accumulation, synchronization,
and final gradient readback. All-state comparison is a separate untimed replay.
Rust's independent full forward-AD reference took 96.97 seconds; it performs a
separate simulation for each parameter and is not a matched reverse-mode solver.

CPU checkpoint storage accounting:

| Allocation | Bytes |
| --- | ---: |
| Window state and adjoints | 347,904 |
| Primal checkpoints (45 boundaries) | 326,160 |
| Boundary adjoint | 7,248 |
| Extra parameter-gradient accumulator | 81,352 |
| **Total of the above** | **762,664** |
| Full-tape state and adjoints | 14,858,400 |
| Target and weight arrays, unchanged | 4,952,800 |

The accounted state/adjoint/checkpoint allocations plus the extra accumulator are
**19.48 times smaller**, at **32.2% more CPU runtime**. Including the unchanged
target/weight arrays reduces the ratio to about 3.47. These are logical field
sizes, not measured process peak memory: graph/parameter fields, runtime/compiler
allocation, AD stacks, allocator padding, and the host audit fixture also use
memory. Metal field sizes are half the CPU f64 sizes for the same shapes.

Receipts are `taichi-full-1024-cpu-timing.json`,
`taichi-checkpoint-cpu-timing.json`, `taichi-checkpoint-cpu-validation.json`, and
`taichi-checkpoint-metal-audit.json`. A separate two-copy, four-neuron Metal audit
is in `taichi-checkpoint-metal-batch-audit.json`; batches still replicate one
trial, not distinct stimuli.

## Storage and derivative ownership

For T steps, interval K, B trials, N neurons, and scalar width d, the resident
state/checkpoint/carry fields occupy:

```
3 * B * N * d * (2*(K+1) + ceil(T/K) + 1)
```

An additional P*d bytes accumulates parameter gradients, where P is the parameter
count. Targets and weights remain O(T*N). Ignoring integer rounding and the
constant terms, K near sqrt(T/2) minimizes this state-storage formula; the fastest
interval can differ and has not been exhaustively tuned. Twenty-three is this
storage-based choice for 1,024 steps.

A boundary observation belongs to the preceding segment; the initial frame is
included only in the first segment. Every sample contributes once to the masked
loss. Each segment's reverse pass differentiates a terminal state dot product
with the carried adjoint, in addition to its local observed loss. The initial
state is differentiated through its parameter-dependent initialization only in
the first segment. Parameters, constants, and checkpoint states remain unchanged
during each value/gradient call. Higher-order derivatives are not implemented.

## Regression coverage and limits

The CPU tests compare one-step windows, a non-dividing interval, one whole window,
and an oversized interval with the full tape. They cover repeated calls, changed
parameters, two replicated trials, and a primal-only replay after validation.
CI also audits every derivative of the full anatomy with seven-step windows.
The independent 1,024-step CPU/Metal receipts were generated locally.

Taichi 1.7.4's `Tape.__exit__` restores kernel modes in call order. For a kernel
called repeatedly, that can leave it in validation mode after the tape and make
a later primal replay fail on stale access checkbits. `replay_safe_tape` restores
the recorded modes in reverse order after normal tape execution. Validation
remains enabled inside each differentiable window. This compatibility helper
uses pinned Taichi tape internals and must be reassessed when upgrading Taichi.

The checkpoint path remains optional. It does not establish stable training over
biological recording durations, total-process memory scaling, heterogeneous
stimulus batching, real-data fitting, or body feedback. Streaming observation
windows and reducing host/kernel dispatch overhead are the next memory and speed
opportunities. CPU remains faster than Metal for the measured single-trial case.
