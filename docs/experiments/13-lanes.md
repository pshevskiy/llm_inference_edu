# T13 — Four accumulation lanes in one projection

The T11 Qwen diagnostic attributed 108.91 ms of 406.71 ms to tied vocabulary projection. This selected the operation to investigate. The original `linear` visits contiguous weight rows, with one increasing-index accumulator. Compiler assembly already vectorizes products but then performs scalar additions through one dependency chain.

The candidate retains row-major W[O,I] and caller-owned outputs. Four independent partial sums expose instruction-level independence. Safe Rust chunk iteration handles tails and avoids alignment or architecture intrinsics. LLVM 21.1.1 on this M1 Pro generated two pairs of `fmul.2s`/`fadd.2s` per group of four. No unsafe code, threads, dependencies or fast-math flags were added. Original scalar `linear` remains callable.

Release experiment: synthetic public FP32 inputs, actual rectangular Qwen dimensions; three warmups and ten alternating-order samples per kernel, one token. Independent FP64 quality checks use the unchanged operator gate atol1e-5 + rtol1e-4 before any timing. One initial overlapping run was retained but excluded; the following run was isolated from agent builds/tests and real-model work. No CPU affinity or cache flush was used.

| I → O | Scalar median ms | Four lanes median ms |
|---|---:|---:|
| 896 → 151936 | 108.445500 | 27.035333 |
| 896 → 4864 | 3.465937 | 0.875958 |
| 4864 → 896 | 3.946813 | 0.991479 |
| 896 → 896 | 0.640458 | 0.160208 |
| 896 → 128 | 0.091209 | 0.022917 |

Evidence: ignored `docs/evidence/t13-kernels-isolated/{identity,completion,summary}.json`, `raw.jsonl`, and `t13-{scalar,lanes}-asm.txt`. Actual host Apple M1 Pro/MacBookPro18,3, 16 GB, macOS26.6.2, AC power, Rust1.90.0/LLVM21.1.1. Identity records actual executable hash and runtime source/compiler snapshots; these do not certify build provenance. Inputs are defined by the recorded formula and dimensions; no pretrained weights enter this microbenchmark. Payload admission includes FP64 expectations, two output rows and inputs/weights within 6 GiB. Physical RSS and energy are not measured here. Ten samples do not establish a robust p95.

Decision: retain the candidate for vocabulary projection only, subject to unchanged tiny/real numerical checks. Other transformer projections remain the original scalar path despite the shape results. Synthetic kernel ratios are not full-model speedups or sustained thermal observations. Proposed long-context and full-model repeated measurements remain deferred pending explicit opt-in. A portable safe candidate can vectorize differently on another compiler/target and needs new measurements there.

Reproduce from a fresh release build:

```sh
cargo build --offline --release --example kernel_benchmark
target/release/examples/kernel_benchmark docs/evidence/NEW-kernels
cargo rustc --offline --release --lib -- --emit=asm
```

Inspect the exact `linear` and `dot_lanes4` symbol definitions in the emitted assembly, not a call site. Keep build/executable identity beside the excerpt.
