# T14 — Prefill projection chunks: a negative result

Compare the T13 four-lane matrix-vector row kernel with a projection-only prototype that reads a weight group once across a token chunk before advancing. This does not implement batched decoder prefill or populate KV. The production decoder and its serial prefill are unchanged.

Three actual Qwen rectangular shapes and token counts1/4/8/16/32 were tested using public synthetic FP32 values; chunks1/2/4/8, three warmups and ten rotating-order samples. All outputs match the lane-order reference exactly and independent FP64 sums satisfy the frozen atol1e-5/rtol1e-4 operator gate. Construction and reference work stay outside timing. The final development-only prototype is out of line for inspection; final binary/workload/source identity and raw results are preserved.

| I→O | Tokens | Independent rows median ms | Chunk2 ms | Chunk4 ms | Chunk8 ms |
|---|---:|---:|---:|---:|---:|
| 896→4864 | 1 | 0.874167 | 3.725437 | 3.723416 | 3.719875 |
| 896→4864 | 8 | 7.139063 | 20.778229 | 17.105688 | 14.988479 |
| 896→4864 | 32 | 28.542146 | 83.074042 | 68.540751 | 60.006833 |
| 4864→896 | 1 | 0.992916 | 3.727917 | 3.720667 | 3.730104 |
| 4864→896 | 8 | 8.051854 | 21.296813 | 17.361249 | 15.110666 |
| 4864→896 | 32 | 32.066729 | 85.186292 | 69.413167 | 60.474479 |
| 896→128 | 1 | 0.022917 | 0.097958 | 0.097979 | 0.097959 |
| 896→128 | 8 | 0.183042 | 0.546188 | 0.452500 | 0.394813 |
| 896→128 | 32 | 0.734396 | 2.183438 | 1.800791 | 1.579250 |

Decision: reject production adoption. Larger chunks improve this prototype relative to its chunk1 mode, but every candidate remains slower than independent optimized rows. For T32/chunk8, logical weight-loop visits fall from32 to4 while latency still rises. Lower loop visits do not prove lower measured DRAM traffic or a faster kernel. The emitted prototype has vector multiply/add but also accesses its stack partial sums and runs dynamic token/index loops. Loop/control/scratch traffic appears to outweigh its reuse benefit in these tests; that is an inference, not a hardware-counter attribution. No universal compute-bound/bandwidth-bound claim is made.

The rejected implementation remains only in `examples/support/prefill_kernel.rs` with isolated tests. It cannot be selected by the production decoder. No extra dependency, BLAS, unsafe code, threads, transposition, KV-layout change or speculative prefill framework was added. A future true matrix-prefill implementation must separately preserve per-layer causal mask, absolute RoPE positions and cache commit/failure contracts. This experiment does not certify it.

Actual host M1 Pro/MacBookPro18,3,16GB/macOS25G83, AC power, Rust1.90/LLVM21.1.1. Logical maximum payload is checked before weight allocation against6GiB; a fixed128-byte stack accumulator bounds chunk scratch. No physical memory, energy, sustained thermal, Apple accelerator or mobile result is claimed. Short public numerical measurements are distinct from sustained real-model tests, which still await opt-in. Full-model longer contexts remain outside current32-token admission. No theoretical bandwidth number is substituted for measured data.

Evidence: ignored `docs/evidence/t14-final/{identity,completion,summary}.json`, `raw.jsonl` (750 samples), `t14-chunk-asm.txt`, and `t14-final-run.txt`. Initial production-prototype measurement retained separately; final results use the final development-only executable.

```sh
cargo build --offline --release --example prefill_experiment
target/release/examples/prefill_experiment docs/evidence/NEW-prefill
cargo rustc --offline --release --example prefill_experiment -- --emit=asm
```

No claim that ten samples provide robust p95. Inspect immutable identity and completion records before consuming partial results. Input/workload hash, actual executable hash and runtime source/compiler snapshots are retained; rebuild first.

Evidence limitation discovered in review: the executed recorder used duplicate
`memory` keys, and its immutable identity omitted the actual RAM field. Host16GB
was verified on the same machine in T11; the timing records remain valid. The
recorder now uses `host_memory` and `memory_scope` separately. Existing identity
was preserved unchanged; no new timing run is required by that metadata-only
correction. Future runs use the corrected recorder.
