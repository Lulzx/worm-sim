# Experimental differentiable Metal backend

We adopt the kernel-level reverse differentiation and tape approach of
[DiffTaichi (ICLR 2020)](https://arxiv.org/abs/1910.00935), using
[Taichi](https://github.com/taichi-dev/taichi) 1.7.4. The
[DiffTaichi repository](https://github.com/taichi-dev/difftaichi) now contains
examples; automatic differentiation lives in Taichi itself. No example source
was copied. This is an optional Python-hosted backend; Rust owns the graph,
parameter contract, compressed formats, baseline evaluation, and reference solver.

## Reproduce

```sh
uv venv --python 3.12 .venv-taichi
uv pip install --python .venv-taichi/bin/python -r backends/taichi/requirements.txt
mkdir -p runs
cargo run --release --example export_taichi_reference -- runs/taichi-reference.json
.venv-taichi/bin/python backends/taichi/audit.py runs/taichi-reference.json --arch cpu --validate --output runs/cpu-audit.json
.venv-taichi/bin/python backends/taichi/audit.py runs/taichi-reference.json --arch metal --output runs/metal-audit.json
.venv-taichi/bin/python backends/taichi/audit.py runs/taichi-reference.json --arch metal --batch 32 --output runs/metal-batch-audit.json
```

The fixture exercises all 45 raw parameters of a four-neuron circuit, eight
chemical edges, four gap junctions, 64 Euler steps, nonuniform resting voltages,
current injection, missing observations and confidence-weighted loss. Rust
forward-mode AD supplies an independent derivative reference. All batch members
are checked, but currently repeat the same trial with shared parameters.

On the M4 Pro, CPU f64 maximum gradient error was 8.7e-17 with Taichi's global
access validator enabled. Metal f32 passed without CPU fallback: maximum gradient
error 2.3e-8 for one trial and 3.1e-8 for 32 replicated trials. Receipts are
`taichi-cpu-audit.json`, `taichi-metal-audit.json`, and
`taichi-metal-batch-audit.json`. Tolerance is absolute 2e-8 plus relative 0.002
on Metal, fixed before the implementation correction. These are small-circuit
correctness checks, not biological validation or full-network performance results.

## Metal regression and representation

The original nested dynamic CSR loops produced accurate forward values but
incorrect reverse gradients on Metal (maximum error 3.1e-5, including missing or
doubled chemical derivatives). CPU f64 passed. The failed receipt is retained in
`taichi-metal-dynamic-loop-failure.json`. This is an observed formulation/backend
interaction, not a fully isolated compiler bug report.

Static neighbor slots guarded by each target's actual degree fix this fixture
without changing tolerances. Storage remains CSR; static unrolling increases
compiled code with maximum degree. Measure compile size/time and gradients
when changing anatomy or scaling the horizon. A per-edge kernel with explicit
current reductions is an alternative if unrolling becomes expensive.

Time-indexed voltage, calcium, and shared presynaptic gate fields respect
[Taichi's autodiff access rules](https://docs.taichi-lang.org/docs/differentiable_programming).
The default full-tape path retains every step and its adjoint: `6*(T+1)*B*N*sizeof(float)`
bytes, excluding parameters, constants, tape/compiler/runtime overhead. This is
not a compressed training tape. An optional [checkpoint path](CHECKPOINTING.md)
now recomputes windows and carries boundary adjoints. State remains differentiable
floating point; the lossless Rust trajectory codecs serve archival output.

[Metal supports f32, not f64](https://docs.taichi-lang.org/docs/type). CPU f64
therefore remains the numerical reference. Metal timings on this tiny fixture
are dominated by overhead and do not demonstrate acceleration. Long-horizon
stability, mixed precision, distinct batched stimuli,
and end-to-end fitting still need measurement beyond the audited cases. Metal
AOT/C API integration is not established here; the documented stable/master
support differs, so this backend uses the verified Python runtime.

## Complete anatomy audit (2026-09-27)

```sh
cargo run --locked --release --example export_taichi_reference -- runs/taichi-c302-reference.json data/c302-herm.wsc
.venv-taichi/bin/python backends/taichi/audit.py runs/taichi-c302-reference.json --arch cpu --validate --output runs/c302-cpu-validation.json
.venv-taichi/bin/python backends/taichi/audit.py runs/taichi-c302-reference.json --arch cpu --output runs/c302-cpu-timing.json
.venv-taichi/bin/python backends/taichi/audit.py runs/taichi-c302-reference.json --arch metal --output runs/c302-metal.json
```

The exporter now accepts a WSC1 graph. The pinned c302 anatomy has 302 neurons,
3,638 chemical edges, 1,080 undirected gaps, and 10,169 raw parameters. This audit
checks **every derivative**, all voltage/fluorescence samples, and the masked loss
against independent Rust forward AD. It uses 64 Euler steps (64 ms), periodic
nonuniform resting voltages/confidence/target offsets, and current injected into
the first canonical neuron. Targets are synthetic, not experimental recordings.
All parameters are compared, including derivatives that are zero or smaller than
the absolute tolerance; passing does not establish identifiability.

Measured on the M4 Pro, five timed repetitions after the first reverse pass:

| Backend | Median forward + reverse | First reverse including compilation | Maximum gradient error | State + adjoints |
| --- | ---: | ---: | ---: | ---: |
| CPU f64, one thread | 8.45 ms | 0.46 s | 3.01e-17 | 942,240 bytes |
| CPU f64, access validator enabled | 30.34 ms | 0.80 s | 3.01e-17 | 942,240 bytes |
| Metal f32 | 38.24 ms | 15.83 s | 4.37e-10 | 471,120 bytes |

Receipts: `taichi-c302-cpu-timing.json`, `taichi-c302-cpu-audit.json`, and
`taichi-c302-metal-audit.json`. Large receipts retain vector hashes and the 16
largest errors rather than duplicate entire gradient arrays. Offline compilation
cache is disabled for these runs. Timings include Python dispatch, synchronization,
and gradient readback; the state-byte estimate excludes AD stacks and runtime
allocations. The Rust full forward-AD reference took 6.99 s for 10,169 separate
parameter directions. That difference measures differentiation algorithms as well
as runtimes and is not a GPU-versus-Rust solver speedup.

CPU is about 4.5 times faster than Metal for this short single-trial reverse pass.
The audit CLI now defaults to CPU for this workload; larger batches and longer horizons require separate
measurements. Metal's scalar loss differs from Rust by about 6.4e-7, while state
and gradient errors remain much smaller. The f32 loss accumulation needs further
accuracy work before using tightly converged loss values as a stopping criterion.

### CPU stack failure and backend-specific loops

The first full-anatomy CPU attempts exited with native signals 11/10 under
Taichi's automatic AD stack sizing (`ad_stack_size=0`), with both static and
dynamic neighbor loops. A dynamic-loop attempt also failed without the access
validator and with the offline cache disabled. Explicit capacity 128 completed;
capacity 256 passed the complete gradient audit with the validator. This narrows
the failure to a stack-sizing-sensitive configuration, without claiming an
isolated upstream compiler defect.

The audit runner now defaults to `--ad-stack-size 256`; this is a tested capacity
for these fixtures, not a universal bound for arbitrary graphs. CPU uses dynamic
CSR traversal; Metal keeps static slots to avoid the earlier dynamic-loop
reverse-gradient regression. Maximum c302 chemical/gap degrees are 63/47.
Both the original four-neuron regression and the full-anatomy CPU validator run
are included in CI. Metal is audited locally because hosted CI has no Metal GPU.

This establishes short-horizon gradient parity on the imported anatomy. It does
not establish stable long-horizon training, heterogeneous trial batching,
measured total peak memory, biological fitting, or body feedback.
The subsequent [checkpoint experiment](CHECKPOINTING.md) audits 1,024 steps and
measures the recomputation/storage tradeoff. Metal dispatch/reduction costs remain
optimization targets, with the complete derivative audit retained as the
correctness gate.
