# T18 — Short pretrained Core ML cache qualification

Actual MacBookPro18,3/M1 Pro/16 GB, macOS26.6.2. Public macOS15/iOS18 stateful
ML Program APIs; installed Command Line Tools SDK26.5, Swift6.3.3. Core ML
Tools9.0/PyTorch2.6.0/NumPy2.3.2 pinned isolated reference environment. Same pinned
Qwen revision/assets as T05/T06; no model download, GPU or private API.

The public minimal probe rejected FP32 state (`State only support fp16 dtype`)
and accepted FP16 state. A FP32-cache source control passed all392reference arrays
and eight greedy IDs. Single FP16 KV storage failed frozen numerical gates:
worst allowed-error fraction21.85594461, logit error.02270013. All greedy IDs still
agreed; this alone was insufficient. No gate/tolerance was widened.

An explicitly declared second candidate uses FP16 leading/residual planes and
reconstructs in FP32. It passed source and traced qualification, then exported/
compiled CPU_ONLY checkpoint qualification. All392arrays passed unchanged gates
(logit atol.001, intermediate atol.005, rtol.0001); maximum allowed-error
fraction.6105585152 (Rust is, position0, layer21 output; maxabsolute.0057373047).
Maximum logit error.000674247742. All8greedy IDs exact. Three interleaved native
states share one compiled model; complete state hashes survive invalid token/
position admission. Fresh-state reset replay passes. Portable fake-predictor
checks cover capacity admission and partial/nonfinite failure invalidation.

This representation is approximate, narrower-range than FP32 and costs exactly
the original786,432logical cache bytes at C32 (96FP16planes). FP32 weights,
activations and accumulation remain explicit. No quantization acceptance or
INT8 decision was bypassed. The selected artifact is a development qualification
path, not a CLI/default precision change or production native integration.
Only short positions0..2 are numerically qualified; capacity32 is allocated,
not evidence that every32-token prefix or large value range is qualified.

| Phase | Scoped single time | Child peak RSS | Child peak footprint |
|---|---|---:|---:|
| Export conversion / save | 12.227460166s / .697897417s | 3,244,670,976 B | 2,852,948,656 B |
| Compile saved artifact | .689875s | 361,709,568 B | 262,013,936 B |
| Load compiled artifact | 4.869456667s | 4,307,697,664 B (check process) | 276,465,392 B (check process) |

Eight diagnostic predictions took97.332084,43.054750,38.008667,37.329583,
38.325916,37.535958,37.121334,37.898667ms. Outputs include all49intermediates;
this is not a policy-matched generation benchmark. No distribution, comparative
speedup, cold-cache, energy, sustained thermal or ANE execution claim. Child
memory includes imports/mapped pages; separate compiler services are not
attributed. File-backed resident pages and footprint differ; do not substitute
one metric for the other. Planning admitted three expanded-weight copies +1GiB
against8GiB, with disk-space check; that estimate is not measured residency or
an enforced process-memory limit. Highest measured child RSS4.31GB stays below
that planning budget. Package files1,977,422,232bytes; source BF16 file unchanged
988,097,824bytes, FP32 logical weights1,976,131,072bytes.

Artifacts/evidence remain ignored under `docs/evidence/t18-qwen/`: immutable
identity, source/traced/native checks, package/compiled hashes, per-phase process
logs and completion. Final preflight `t18-final-preflight` includes host model/
chip/RAM and control/single-half/paired results. Initial single-half preflight
`t18-preflight` omitted those host fields; its numerical negative result is
preserved, and final run provides corrected identity. Source hashes identify all
measured graph/tool versions. No credentials/signing/toolchain changes.

Ordinary fmt/Clippy/tests passed; portable wrapper tests and syntax checks passed.
Astra final code/native-evidence review found no remaining actionable issues.
Owner review remains pending. T19 must independently
compare execution policy/placement before any ANE claim; CPU_ONLY qualification
says nothing about ANE work attribution or iPhone14 validation.
