# T27 — A bounded Metal residual-add slice

Current run: `docs/evidence/t27-metal-final/`, external process log
`t27-metal-final.txt`, exit0. M1 Pro/MacBookPro18,3,16GiB,
macOS26.6.2/build25G83, Rust1.90.0/Homebrew LLVM21.1.1. Actual Metal device:
Apple M1 Pro, unified memory=true, pipeline execution width32. Global developer
directory stays CommandLineTools. Runtime shader compilation succeeded without
installing a toolchain or changing global Xcode configuration.

The package pins public `metal=0.33.0`, `objc=0.2.7` and
`foreign-types=0.5.0`; ten new dependency packages are locked (three direct,
seven transitive). The last two expose
autorelease/object ownership and nil-checked selectors. No MPS, private API,
GPU framework, decoder integration or model download. Root scalar crate still
forbids unsafe. Production inference remains CPU.

Shader: one FP32 residual addition per thread, contiguous lane access, tail guard.
Pipeline/queue/three buffers reused, fast math=false. All ten deterministic input
lengths1/31/32/33/255/256/257/4,097/262,144/1,048,576 match scalar exactly under
the frozen `1e-6 + 1e-6×abs(reference)` gate. Hand-derived `[3,1,-5,0]` passes.
Invalid lengths/capacity/nonfinite inputs and finite-input overflow preserve
output; successful reuse follows. Signed zeros and cancellation-to-zero agree
bitwise. Subnormals flush: CPU expected bits2/4,194,304, GPU actual0/8,388,608.
This limitation is disclosed, not hidden by a full IEEE-equivalence claim.

Three warmups and ten samples per length, one bounded process. Medians are the
mean of the fifth/sixth sorted observations. Representative microseconds:

| Elements | Public Metal call | Encoding/submission/wait | GPU command timestamps | Scalar public call |
|---:|---:|---:|---:|---:|
| 4,097 | 203.000 | 197.479 | 9.813 | 0.459 |
| 262,144 | 637.729 | 258.521 | 40.854 | 33.500 |
| 1,048,576 | 2,133.771 | 626.542 | 94.354 | 136.271 |

The full safe Metal call loses to scalar on these workloads. Kernel timestamps
alone would misrepresent the usable operation. The scopes differ intentionally:
public call includes validation/CPU copies/output checks, submission timing does
not, and GPU timestamps exclude most host work. No observed DRAM-bandwidth or
PCIe-transfer claim. Tiny scalar samples reach clock resolution, including zero;
no speed ratios are derived from them. Fixed ordering/cache/power conditions are
not a robust backend/application comparison or a sustained thermal benchmark.
Energy not measured.

Final setup took36,195,416ns; an earlier unchanged kernel/pipeline run took
242,891,167ns, so neither establishes cold compiler latency. Logical shared-buffer
payload12,582,912bytes; maximum RSS59,244,544bytes; macOS peak process
footprint50,397,760bytes. File size, logical payload and both physical figures
are distinct. Whole-process time0.73s includes observations and reference work.

The first passing run `t27-metal-v1` is preserved. Astra found assertions could
bypass failed completion evidence. The qualifier now returns errors and saves
cases/workloads incrementally to `progress.json`; identity is captured before
device creation, completion/failure separately. Final source/binary identity and
workload hashes belong to the final run; the earlier source is not mislabeled.
Ordinary fmt, warning-denied Clippy and112unit/integration tests pass. Astra
final review approved, no remaining actionable findings; owner review pending.
No command-failure/device-loss fault was induced; that
conservative invalidation path is code-reviewed, not claimed hardware-tested.

Reproduce with lesson27's commands. No iPhone validation: owner explicitly
deferred it. This evidence qualifies only the first Mac operation.
