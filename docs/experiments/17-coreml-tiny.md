# T17 — Tiny FP32 Core ML CPU qualification

Actual MacBookPro18,3/M1 Pro/16 GB, macOS26.6.2, arm64. Python3.12.8,
Core ML Tools9.0, PyTorch2.6.0 and NumPy2.3.2, pinned reference-only environment.
Public Core ML conversion/compile/predict APIs; CPU_ONLY throughout. No Rust
runtime dependency, FFI, GPU inference, private ANE API or pretrained export.

Verified the original NumPy fixture byte-for-byte after extracting its callable
oracle. PyTorch source/traced graph checked four explicit lengths1..4 against
that independent oracle. Converted the full tiny two-layer decoder with35named
outputs, FP32 typed compute/output and int32 token IDs, enumerated shapes,
macOS13 deployment floor (iOS16 equivalent). Compilation deferred until a
separate process. CPU prediction checked all140named arrays within original
atol1e-5/rtol1e-4; maximum absolute error9.536743164e-7, maximum allowed-error
fraction.040932612. All10teacher-forced greedy IDs agree. Adding a future token
preserves the first3logit rows within the same gate (maxabsolute4.172325134e-7).

| Phase | Single scoped observation | Child peak RSS | Child peak footprint |
|---|---|---:|---:|
| Convert / save | .610990375s / .001613s | 335,806,464 B | 257,770,360 B |
| Compile saved package | .069516583s | 313,294,848 B | 237,126,520 B |
| Load compiled model | .144861459s | 319,504,384 B (whole check process) | 240,747,480 B (whole check process) |

One prediction per shape took1.768875,1.469667,1.360875,1.375417ms. These are
correctness diagnostics, not distributions or an end-to-end speed comparison.
First-call initialization/shape effects are included; no cold-cache claim.
Each child peak includes Python imports, and memory of separate system compiler
services is not attributed. Package files total123,673bytes; Rust logical FP32
weights82,560bytes. Neither number is physical residency. No energy, sustained
thermal or ANE placement result is claimed.

Final evidence is ignored `docs/evidence/t17-final/`: identity, immutable phase
results, artifact hashes, compiled/package artifacts, raw phase process logs,
completion. The initial run `t17-tiny` passed but its identity omitted the NumPy
oracle's imported RoPE source hash. Astra found that gap; added the hash and
reran in a fresh directory. Initial evidence retained. No numerical rule changed.

Reproduce with a fresh result directory using lesson17. Applicable Rust checks
passed (88 integration+3unit), original independent fixture verification and
Python syntax compilation passed. Astra final code/evidence review found no remaining actionable issues. Owner
review remains pending.

T17's CPU-only gate passes for this narrow synthetic graph. T18 must independently
qualify pretrained state, cache positions and precision with the installed public
APIs. T19 must separately measure/attribute placement before any ANE claim.
