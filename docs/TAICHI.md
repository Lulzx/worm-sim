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
compiled code with maximum degree. Validate compile size/time and gradients
before extending it to the complete anatomy. A per-edge kernel with explicit
current reductions is an alternative if unrolling becomes expensive.

Time-indexed voltage, calcium, and shared presynaptic gate fields respect
[Taichi's autodiff access rules](https://docs.taichi-lang.org/docs/differentiable_programming).
They currently retain every step and its adjoint: `6*(T+1)*B*N*sizeof(float)`
bytes, excluding parameters, constants, tape/compiler/runtime overhead. This is
not checkpointed or a compressed training tape. State remains differentiable
floating point; the lossless Rust trajectory codecs serve archival output.

[Metal supports f32, not f64](https://docs.taichi-lang.org/docs/type). CPU f64
therefore remains the numerical reference. Metal timings on this tiny fixture
are dominated by overhead and do not demonstrate acceleration. Long-horizon
stability, mixed precision, distinct batched stimuli, real graph gradients,
checkpoint/recompute, and end-to-end fitting still need measurement. Metal
AOT/C API integration is not established here; the documented stable/master
support differs, so this backend uses the verified Python runtime.
