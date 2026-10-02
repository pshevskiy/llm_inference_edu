# Apple Inference Engine — Codex Implementation Plan

**Revision:** 0.2 — learning-first  
**Prepared:** 2026-10-01  
**Status:** Implementation in progress; see individual ticket records.
**Primary hardware:** Mac M1 Pro, 16 GB unified memory.  
**Later physical test device:** iPhone 14; iPad support follows device qualification.  
**First ticket:** T00. Do not execute the whole roadmap in one run.

**Current owner authorization — 2026-10-02:** Continue the roadmap sequentially,
one commit per ticket, with a GPT-6 Astra review before each commit. Account for
mechanical sympathy in layout and buffer ownership; performance claims still
require measurements. This explicitly permits advancing through multiple
tickets in this run. The owner also approved the pinned Qwen model download and
isolated reference/export setup. Hardware qualification and owner-understanding
gates still need their actual evidence; approval to proceed is not a passed gate.

> Build an inference engine that its owner can understand, debug, optimize, and maintain without an AI coding agent. Performance is a goal to measure, not a property to assume.

This plan supersedes the **integration-first implementation sequence** in `apple_inference_ecosystem_plan.md`. That draft put a Core ML wrapper first. This revision puts our own numerical implementation first, while preserving the ecosystem, ANE, model-search, and later-GPU requirements. The CPU reference is an intentional learning milestone, not a claim to have delivered ANE acceleration.

## 1. Requirements, defaults, and exclusions

### Confirmed direction

| ID | Requirement | Implementation consequence |
|---|---|---|
| R1 | Deeply understand inference-engine internals. | Implement the forward pass and cache ourselves; do not hide them behind a pretrained-model runtime. |
| R2 | Simple, human-maintainable design and code. | Small increments, concrete types, ordinary builds, explicit ownership, and few dependencies. |
| R3 | High token-generation performance with low resource use. | Measure decode, prefill, memory, latency, and quality independently; energy where reliably measurable. |
| R4 | Rust is the primary engine language. | Python is reference/export tooling; Swift is later native integration. |
| R5 | Start on the existing M1 Pro with 16 GB. | Use tiny fixtures, one small pretrained model, one sequence, and bounded working memory. No hardware purchase prerequisite. |
| R6 | Initial acceleration direction is CPU + Neural Engine. | Keep CPU-only reference and Core ML/ANE experiments distinct; no hidden GPU model execution. |
| R7 | GPU execution is a later extension. | No Metal dependency in the initial runtime. GPU-enabled external baselines are explicitly labeled development tools. |
| R8 | Eventually support macOS, iPhone, and iPad. | Portable numerical core; real iPhone 14 validation later. Do not assume desktop resource budgets apply on mobile. |
| R9 | UI searches for MLX models on Hugging Face. | Discovery is separate from execution compatibility and conversion. |
| R10 | Ecosystem includes an agent and educational apps. | They reuse the engine; they do not determine its initial numerical architecture. |
| R11 | Long-context, multi-step agent workloads matter. | Correct incremental state and continuation first; larger contexts and advanced cache policies later. |

### Proposed implementation defaults—not additional user commitments

Use a tiny two-layer decoder first. Then qualify **`Qwen/Qwen2.5-0.5B-Instruct`** as the first pretrained checkpoint. It is a deliberately small, established candidate, not a claim to be the latest or best model. Pin its repository revision before downloading weights. Its Core ML/ANE suitability remains unverified.[^qwen-config]

Start with FP32 CPU arithmetic, batch size one, and no padding. For the real model, validate short prompts first; progress toward a **2,048-token total capacity**, including generated tokens, only after memory and correctness checks. Begin with one loaded model and one active generation. Context size, resource budget, and final precision remain measured choices.

Do not build training, autograd, a general tensor framework, a graph compiler, broad model-family support, distributed execution, continuous batching, an HTTP server, a vector database, a plugin system, or a multi-agent framework. Do not add empty packages to anticipate them.

## 2. How Codex must work

The unit of work is **one ticket**, not an entire milestone. At each ticket:

1. State the purpose, relevant inputs/outputs, and a short intuition. Identify the files likely to change.
2. Implement the smallest complete, testable slice. Keep unrelated code untouched.
3. Verify it independently; run applicable tests and report the actual results.
4. Explain the core implementation, its buffer layout, and one small numerical or execution trace.
5. Give one bounded exercise or question for the owner. Do not automatically solve the exercise.
6. Update the ticket with evidence and stop. Do not start the next ticket without a new instruction.

The owner may explicitly request an exercise-only interaction instead; in that case, supply the contract/tests and review their implementation rather than completing it for them. Rust remains the production language in either mode.

`AGENTS.md` contains the short persistent working rules. This file is the authoritative roadmap and progress record. Codex supports project instructions in `AGENTS.md`; keep that file concise and explicitly direct it to read the current ticket here.[^codex]

**Done means:** implementation, applicable tests, a reproducible command, a short explanation, and stated limitations. A hardware-only check that could not run is **blocked/not run**, not passed. Code completion and owner review are separate: Codex may report completion but must not claim the owner has reviewed or understood it.

## 3. Minimal code organization

Start with **one Rust package** containing a library. Add a CLI in the same package when text generation exists. The following is an incremental destination, not a request to scaffold it today:

```text
AGENTS.md
docs/
  PLAN.md                   # this roadmap and its ticket evidence
  lessons/                  # one compact explanation per completed mechanism
  experiments/              # hypotheses and measured results, added when needed
src/
  lib.rs
  ops.rs                    # readable scalar operations, split only as necessary
  model.rs                  # one concrete decoder architecture
  weights.rs                # validated checkpoint loading
  kv_cache.rs               # explicit cache storage and position rules
  generation.rs             # token history, sampling, stop conditions
  main.rs                   # CLI, added later
reference/                  # tiny NumPy oracle; later pinned Transformers reference
fixtures/                   # small checked-in test data, never full model weights
benchmarks/                 # harness and small result manifests
apple/                      # added only at the native milestones
```

No `Device` trait, generic `Module` hierarchy, tensor operator overloading, dynamic backend registration, or cross-language bridge is needed for T00. Ordinary functions over slices are sufficient. Introduce a narrow backend interface only after two actual execution paths demonstrate its required operations.

### What we implement versus reuse

| Implement for understanding | Reuse rather than reinvent |
|---|---|
| Linear operations, normalization, RoPE, causal attention, feed-forward block, residuals, model forward. | Existing tokenizer implementation and checkpoint tokenization assets. |
| KV cache, incremental generation, sampling, memory accounting, benchmarks. | `safetensors` parsing, JSON parsing, half-precision decoding when needed. |
| Selected, measured CPU and later Metal optimizations. | Core ML runtime/compiler for public ANE deployment; native UI and OS services. |

Hugging Face supplies Rust tokenization support and the safetensors ecosystem.[^tokenizers][^safetensors] Add each dependency only when its ticket needs it. Pin the chosen toolchain and dependency resolution in the repository. No Candle, MLX, llama.cpp, or Core ML `predict()` call may stand in for our Rust forward pass. External runtimes remain useful references.

## 4. Numerical and memory contracts

### 4.1 Start with explicit contiguous storage

For a linear layer, store input `X[T, I]`, weights `W[O, I]`, and output `Y[T, O]` in row-major order:

```text
Y[t, o] = sum_i X[t, i] * W[o, i]
X offset = t * I + i
W offset = o * I + i
Y offset = t * O + o
```

This makes weight-row access visible and matches the proposed checkpoint interpretation without a hidden transpose. Validate multiplication overflow, dimensions, and buffer lengths before writing outputs. Initially reject zero dimensions explicitly. Do not build arbitrary strided views just to represent these arrays.

Weights are immutable after loading. Output and scratch buffers have explicit owners. Later, reuse scratch space in steady-state decode; do not claim the entire process is allocation-free merely because one kernel performs no allocations.

### 4.2 Tiny model specification

Use these **synthetic test dimensions**, not a claimed real checkpoint:

| Property | Test value |
|---|---:|
| Vocabulary | 64 |
| Layers | 2 |
| Hidden dimension | 32 |
| Query heads | 4 |
| KV heads | 2 |
| Head dimension | 8 |
| Feed-forward dimension | 64 |
| Initial context capacity | 32 tokens |

Use explicit deterministic weights with nontrivial values, serialized once into small fixtures. Rust and Python must consume the same values; relying on equal random seeds across different RNG implementations is not sufficient.

The target block is pre-normalization, grouped-query causal attention with rotary positions, a residual connection, then normalization, SwiGLU feed-forward, and a second residual. Document conventions rather than calling it universally “Llama-compatible.” For real Qwen weights, verify every architectural detail against the pinned reference, including Q/K/V biases, rotary pairing, normalization epsilon, and tied output weights.[^qwen-impl]

### 4.3 Correctness before fluent-looking output

Compare operator results and intermediate activations, not only generated text. Maintain two independent references: small NumPy equations for our synthetic model, and the pinned Transformers implementation for the real checkpoint.

For tiny FP32 fixtures, begin with `atol=1e-5`, `rtol=1e-4` as **provisional tolerances**. Report maximum absolute error and a scale-aware relative error. Investigate failures; never simply widen tolerances until tests pass. Establish separate, justified gates for full-model FP32, conversion, and quantization. Test finiteness explicitly.

Teacher-forced comparisons feed identical token IDs at each position so an early sampling difference cannot disguise later numerical errors. Greedy-token equality is useful but insufficient; near-tied logits can change argmax without a large numerical error.

### 4.4 Cache invariants

Keep two distinct concepts: token history and the prefix actually evaluated into cache. A newly sampled token is in history before its K/V values necessarily exist. Never equate “emitted token count” with “cached position count.”

Start with contiguous, fixed-capacity K and V arrays per layer, logical layout `[kv_head, position, head_dim]`. Write rotated keys and unrotated values according to the pinned model semantics. For GQA, map query heads to their KV group without physically duplicating the entire cache.

For a conventional K/V cache, the calculated payload is:

```text
2 * layers * capacity * kv_heads * head_dim * bytes_per_element * sequences
```

The proposed real checkpoint declares 24 layers, hidden dimension 896, 14 query heads, and 2 KV heads, giving head dimension 64.[^qwen-config] At 2,048 positions and FP32, the resulting **calculated logical K/V payload is 48 MiB**. FP16 would be 24 MiB. Neither figure includes padding, weights, activations, or framework memory; neither is an observed footprint.

On cancellation or failure partway through a multi-layer update, do not expose a partially updated cache as valid. The first safe recovery is to invalidate and replay from committed token history. More elaborate transactional cache updates are not required initially.

## 5. Roadmap and executable tickets

All boxes start unchecked. When completing a ticket, add evidence directly beneath it: changed files, commands, results, limitations, and owner-review status. Do not create a separate duplicate task database.

### M0 — One operation we can completely explain

- [x] **T00 — Repository check and scalar linear layer.** Inspect the existing repository and preserve changes. Create a minimal Rust library only when absent. Implement the contract in Section 4.1 using safe scalar FP32 loops, caller-owned output, and no runtime dependencies. Include exact hand-calculated tests, shape-error tests, and overflow checks. Add a tiny runnable example and a short lesson showing flattened offsets. Capture available toolchain/platform facts without assuming the runner is the target Mac.

**T00 completion record — 2026-10-01**

- **Status:** implemented; owner review pending. This record supersedes the
  initial plan-only status for T00. T01 and all later tickets remain unstarted.
- **Repository inspection:** `pwd`, `ls -la`, `rg --files --hidden -g '!target'
  -g '!.git'`, and reads of `AGENTS.md`, this plan, and `docs/FIRST_TASK.md`.
  No ancestor `AGENTS.md` was found in the inspected parent directories.
  There was no Rust package. `git status --short` exited 128 with "not a git
  repository"; no Git repository was initialized. Existing `AGENTS.md` and
  `docs/FIRST_TASK.md` were preserved during initial implementation; the owner
  later requested removal of `docs/FIRST_TASK.md` from the repository.
  Only T00 was edited in this plan.
- **Files changed:** added `Cargo.toml`, generated dependency-free `Cargo.lock`,
  `rust-toolchain.toml`, `.gitignore`, `src/lib.rs`, `tests/linear.rs`,
  `examples/linear.rs`, and `docs/lessons/00-linear.md`; updated this T00 record.
  The five evidence logs below remain local in the ignored `docs/evidence/`
  folder and are excluded from Git history. No dependencies were introduced.
- **Observed host/toolchain:** macOS 26.6.2, build 25G83, Darwin 25.6.0, arm64;
  MacBookPro18,3, Apple M1 Pro, 10 CPU cores (8 performance + 2 efficiency),
  16 GB memory. `rustc -vV` reported Homebrew Rust 1.90.0, commit
  `1159e78c4747b02ef996e55082b704c09b970588`, host `aarch64-apple-darwin`,
  LLVM 21.1.1. Ordinary checks used `/opt/homebrew/bin/cargo` and
  `/opt/homebrew/bin/rustc`; Cargo 1.90.0, rustfmt 1.8.0, Clippy 0.1.90.
  `rustup toolchain list` showed an already installed `1.90` toolchain, and
  `rustup run 1.90 rustc --version` confirmed the same Rust 1.90.0 commit;
  `rust-toolchain.toml` selects that installed version for rustup users.
  No toolchain was installed. The installed `stable` alias reported Rust
  1.86.0, so it was not selected as the project toolchain.
- **Host commands actually run:** `uname -a`, `sw_vers`, `rustc --version`,
  `cargo --version`, `rustc -vV`, `command -v rustc`, `command -v cargo`,
  `command -v rustup`, `rustup show active-toolchain`, `rustup toolchain list`,
  `rustup run stable rustc --version`, `rustup run stable cargo --version`,
  `rustup run 1.90 rustc --version`, `rustup run 1.90 cargo --version`,
  `rustup run 1.90 cargo fmt --version`, `rustup run 1.90 cargo clippy --version`,
  `cargo fmt --version`, and `cargo clippy --version`.
  `sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu` was blocked by the
  sandbox ("Operation not permitted"). `/usr/sbin/system_profiler
  SPHardwareDataType` supplied the hardware facts instead, while warning
  "Error fetching hw.cpufamily: -1". The saved hardware output selects model,
  chip, core count, and memory fields. Local log: `docs/evidence/t00-host.txt`.
  No hardware follow-up is required for T00; to retry the blocked query outside
  the sandbox, run `sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu`.
- **Verification commands actually run:** `cargo fmt --all` (exit 0), then:

  | Command | Actual result | Local log (ignored) |
  |---|---|---|
  | `cargo fmt --all -- --check` | exit 0, no diagnostics | `docs/evidence/t00-fmt.txt` (empty on success) |
  | `cargo clippy --all-targets -- -D warnings` | exit 0, no warnings | `docs/evidence/t00-clippy.txt` |
  | `cargo test` | exit 0; 8 integration tests passed, 0 failed; 0 unit/doc tests | `docs/evidence/t00-test.txt` |
  | `cargo run --example linear` | exit 0; `Y = [[-2.0, 4.0], [-2.0, 13.0]]` | `docs/evidence/t00-example.txt` |

- **Numerical/layout explanation:** `linear` first rejects zero dimensions,
  checks all three shape products, and validates exact slice lengths before
  writing. Then its `t`, `o`, `i` loops compute each dot product in FP32 and
  overwrite the caller's output. For `Y[0,1]`, input offsets 0, 1, 2 hold
  1, 2, 3; weight offsets 3, 4, 5 hold 2, 1, 0. The sums progress
  0 → 2 → 4 → 4, then 4 is written at output offset 1. The innermost loop
  reads consecutive elements from both rows. The weight layout `[output,input]`
  is described mathematically by `Y = X * W^T`, with no transposed allocation.
  See the [lesson](lessons/00-linear.md) for the execution order and offset table.
- **Test coverage:** the required example, single-token input, distinct
  nonsquare dimensions, negative values, finite inputs with zero weights,
  short and long lengths for every buffer, every zero dimension, and overflow
  in each of the three shape products. All rejection cases assert unchanged
  sentinel output. Expected arrays are hand-calculated, independent of the
  implementation, and exactly representable; no tolerance was needed.
- **Known limitations:** arithmetic follows ordinary FP32 semantics without
  finiteness validation; this is one scalar operation without bias. No
  performance or physical memory measurements were made. Model/reference,
  ANE, GPU, and iPhone tests were not run because they are outside T00.
  No commit or push was performed.
- **One learning exercise:** for `tokens=2`, `input_dim=4`, `output_dim=3`, list
  the four input offsets and four weight offsets read for `Y[1,2]`, the output
  offset written, and which accesses are contiguous. The answer remains for
  the owner to supply.
- **Owner review:** pending; implementation completion does not establish
  owner understanding or approval.
- **Next ticket:** T01 proposed only; not started. Wait for a new instruction.

**Gate:** `cargo test`, formatting, and Clippy pass where the toolchain exists; otherwise explicitly report what could not run. Demonstrate:

```text
X = [[1, 2, 3], [4, 5, 6]]
W = [[1, 0, -1], [2, 1, 0]]
Y = [[-2, 4], [-2, 13]]
```

**Learning checkpoint:** Why is the stored weight shape `[output, input]`, and what is contiguous in the innermost loop? Stop here on the first Codex run.

### M1 — A tiny full forward pass

- [x] **T01 — Numerical oracle and basic operations.** Add the minimal NumPy reference environment and fixed fixtures. Implement embedding lookup, residual addition, RMSNorm, SiLU, and numerically stable softmax. Test extreme logits, invalid token IDs, and zero-valued normalization input with epsilon. A masked softmax must reject an all-masked row rather than silently create NaNs.

**T01 completion record — 2026-10-01**

- **Status:** implemented; owner review pending. T00's implementation and
  passing checks satisfied the prerequisite. T02 and later tickets were not
  started. This record supersedes the earlier plan-only status for T01.
- **Files changed:** `src/lib.rs` exports the new `ops` module; added
  `src/ops.rs`, `tests/basic_ops.rs`, `examples/basic_ops.rs`,
  `fixtures/t01-inputs.json`, `fixtures/t01-expected.json`,
  `reference/basic_ops.py`, `reference/requirements.txt`, `reference/README.md`,
  and `docs/lessons/01-basic-ops.md`; updated `Cargo.toml`, `Cargo.lock`,
  `.gitignore`, and this T01 record. The T00 scalar linear implementation and
  its tests were preserved. Logs remain in ignored `docs/evidence/`.
- **Dependencies:** no runtime dependencies. Test-only `serde_json=1.0.149`
  parses the shared JSON fixtures, avoiding a custom parser. Its dependency
  resolution is in `Cargo.lock`; existing cached crates were sufficient.
  Optional oracle dependency: `numpy==2.3.2`. Python and NumPy were already
  installed; no Python, toolchain, model, or ML runtime was installed.
  `reference/README.md` documents isolated virtual-environment setup and
  explicit fixture regeneration. No virtual environment was created in this run.
- **Observed runtime:** arm64 Darwin 25.6.0, macOS 26.6.2 build 25G83;
  MacBookPro18,3, Apple M1 Pro, 10 CPU cores, 16 GB memory. Homebrew Rust/Cargo
  1.90.0, Python 3.12.8, NumPy 2.3.2. Commands: `uname -a`, `sw_vers`,
  `/usr/sbin/system_profiler SPHardwareDataType` (selected hardware fields),
  `rustc --version`, `cargo --version`, `python3 --version`, and
  `python3 -c 'import numpy; print("NumPy", numpy.__version__)'`.
  `system_profiler` reported the same hw.cpufamily warning as T00 while
  supplying the listed facts. Local log: `docs/evidence/t01-host.txt`.
- **Commands actually run and results:**

  | Command | Actual result | Local log (ignored) |
  |---|---|---|
  | `python3 reference/basic_ops.py --write` | exit 0; generated the expected fixture using NumPy 2.3.2 | Saved oracle: `fixtures/t01-expected.json` |
  | `python3 reference/basic_ops.py` | exit 0; byte-for-byte regeneration check passed | `docs/evidence/t01-oracle.txt` |
  | `cargo fmt --all` | exit 0; formatted new code | No diagnostics |
  | `cargo check --offline --lib` | exit 0 | No persisted log |
  | `cargo test --offline --test basic_ops -- --nocapture` | exit 0; 20 T01 tests passed | No persisted log |
  | `cargo fmt --all -- --check` | exit 0; no diagnostics | `docs/evidence/t01-fmt.txt` |
  | `cargo clippy --all-targets -- -D warnings` | exit 0; no warnings | `docs/evidence/t01-clippy.txt` |
  | `cargo test` | exit 0; 28 integration tests passed (20 T01 + 8 T00), 0 failed; 0 unit/doc tests | `docs/evidence/t01-test.txt` |
  | `cargo test --test basic_ops -- --nocapture --test-threads=1` | exit 0; numerical evidence printed in serial order | `docs/evidence/t01-numerics.txt` |
  | `cargo run --example basic_ops` | exit 0; displayed all five operations | `docs/evidence/t01-example.txt` |
  | `cargo tree --edges normal` | only `llm_inference`; confirms no runtime dependencies | No persisted log |
  | `git diff --check` | exit 0; no whitespace errors | No diagnostics |

- **Independent numerical evidence:** both Rust and Python read the explicit
  values in `fixtures/t01-inputs.json`; there is no shared RNG assumption.
  NumPy evaluates direct equations in FP64 on FP32-rounded inputs, then saves
  FP32-rounded expected values. Rust uses scalar FP32 arithmetic. Ordinary
  Rust tests use saved fixtures without invoking Python or accessing a network.
  The unchanged provisional gate is `abs_error <= 1e-5 + 1e-4 * abs(expected)`.
  All fixture comparisons check finiteness. Relative error below means
  `abs_error / max(abs(expected), 1e-5)`, using the absolute tolerance as the
  scale floor. Row copies and exact residual sums additionally require equality.

  | Fixture | Maximum absolute error | Maximum scale-aware relative error |
  |---|---|---|
  | Embedding | 0 | 0 |
  | Residual addition | 0 | 0 |
  | RMSNorm | `1.19209290e-7` | `8.58405969e-8` |
  | SiLU | `6.30584309e-44` | `6.30584309e-39` |
  | Softmax | 0 | 0 |
  | Masked softmax | `5.96046448e-8` | `6.76712588e-8` |

- **Mechanism and trace:** `embedding` copies table rows in token order;
  residual addition combines matching elements; RMSNorm normalizes row
  magnitude and applies per-feature weights; SiLU multiplies by a sigmoid;
  softmax subtracts the maximum, exponentiates, and divides by the sum. All
  shape/ID/numerical input validation occurs before writes. Row-major offsets
  are `row * width + feature`. For logits `[1000,1001,1002]`, subtracting the
  maximum gives `[-2,-1,0]`, exponentials approximately
  `[0.13533528,0.36787944,1]`, sum `1.50321472`, and probabilities approximately
  `[0.09003057,0.24472847,0.66524096]`. The example separately demonstrates
  `[3,4]` → residual `[4,2]` → RMSNorm `[1.2649105,0.6324552]` →
  SiLU `[0.98646605,0.41302234]` → probabilities `[0.6395574,0.3604426]`.
- **Boundary tests:** repeated/ordered token IDs, late invalid IDs, short/long
  buffers, zero dimensions and checked product overflow, zero RMSNorm rows,
  invalid epsilon, large finite RMSNorm and SiLU input, extreme finite logits,
  uniform/single-entry softmax, shift invariance, masked maxima, ignored masked
  non-finite values, active non-finite rejection, and a late all-masked row.
  Every tested rejection asserts unchanged output.
- **Known limitations:** these are small operation fixtures, not a full decoder
  or a pretrained-model qualification. Finite checks apply to RMSNorm inputs/
  weights, SiLU inputs, and allowed softmax logits; embedding copies values and
  residual addition retains ordinary FP32 semantics. Final RMSNorm weights or
  residual sums can still overflow. The scaled RMSNorm equation avoids squared
  input overflow; SiLU uses a sign-dependent exponential to avoid exponential
  overflow. No performance, physical memory, model, ANE, GPU, or mobile tests
  were run; those are outside T01. Owner understanding is not inferred from the
  request to proceed. Changes were not committed or pushed in this run.
- **One learning exercise:** for logits `[8,9,10]` and mask `[true,false,true]`,
  write the maximum used, shifted allowed logits, exponentials, and normalized
  output. Explain the middle probability. The exercise is unsolved; see
  `docs/lessons/01-basic-ops.md` for the supporting explanation.
- **Owner review:** pending.
- **Astra review — 2026-10-02:** GPT-6 Astra found no actionable issues and
  independently verified formatting, Clippy, all 28 integration tests, the
  NumPy regeneration check, and whitespace checks. No review fixes were needed.
- **Mechanical sympathy:** contiguous row access, no per-kernel allocation,
  caller-owned outputs, and output reuse for softmax exponentials. Finite-input
  validation and stable RMSNorm deliberately add passes; no speed claim is made.
- **Commit preparation — 2026-10-02:** repeated ordinary checks and the NumPy
  fixture check passed. The owner authorized one commit per ticket and
  continuation to T02 after this review.
- **Next ticket:** T02 (rotary positions), authorized next.
- [x] **T02 — Rotary positions.** Implement RoPE with explicit head dimensions, pairing convention, absolute positions, and configurable base. Test position zero, nonzero offsets, and agreement with the independent oracle. Preserve vector norm within declared floating-point tolerance.

**T02 completion record — 2026-10-02**

- **Status:** implemented; owner review pending. Astra found no actionable
  issues and independently verified all 35 tests, formatting, Clippy, and the
  NumPy fixture check before commit.
- **Files:** `src/rope.rs`, `src/lib.rs`, `tests/rope.rs`,
  `fixtures/t02-inputs.json`, `fixtures/t02-expected.json`, `reference/rope.py`,
  `docs/lessons/02-rope.md`, and this record. No new dependencies.
- **Contract:** row-major `[tokens,heads,head_dim]`, even dimensions, split-half
  pairs `(j,j+head_dim/2)`, positive configurable base, checked absolute position
  arithmetic. Reject integer positions above 2^24 rather than silently rounding
  consecutive positions in FP32. Shapes, lengths, finiteness, and angles are
  checked before output writes. Ordinary final FP32 overflow remains possible.
- **Commands/results:** `python3 reference/rope.py --write` generated independent
  FP64-equation fixtures from FP32-rounded inputs. `python3 reference/rope.py`
  verified saved output. `cargo fmt --all -- --check`,
  `cargo clippy --all-targets -- -D warnings`, and `cargo test` passed; 35
  integration tests (7 T02) passed. An initial compile error from negating a
  usize before conversion was corrected before these passing checks.
  `cargo test --test rope -- --nocapture --test-threads=1` reports numerical
  errors in ignored `docs/evidence/t02-numerics.txt`. Tests use the unchanged
  `atol=1e-5, rtol=1e-4`, and verify per-head squared norm, identity, offset
  equivalence, both dimension products, malformed lengths/positions/base,
  invalid angles and non-finite inputs. Rejection tests retain sentinel output.
  Both fixtures had maximum absolute error `2.38418579e-7`; maximum relative
  errors were `5.40462395e-7` and `6.20858657e-7` respectively.
- **Mechanical sympathy:** no allocations or transposed copies; share
  sine/cosine across heads. Head-strided access and repeated frequency
  calculation remain explicit tradeoffs, not measured optimizations.
- **Trace:** at position 1, head dimension 4, base 10000, the first pair has
  frequency 1. `[1,0]` becomes approximately `[0.5403023,0.8414710]`;
  position zero leaves each head unchanged.
- **Exercise:** identify split-half pairs for head dimension 6 and explain
  which rotates fastest for base 10000. Unsolved; see the T02 lesson.
- **Limitations:** scalar FP32 operation, not a full model/context qualification;
  no performance or hardware acceleration claim. Actual runtime is the same
  M1 Pro/Rust 1.90.0/Python 3.12.8/NumPy 2.3.2 recorded in T01.
- **Next:** T03, after Astra review and this ticket's commit.

- [x] **T03 — Causal attention and GQA.** Implement Q/K/V projections, per-head scoring scaled by inverse square root of head dimension, causal mask, softmax, value weighting, and output projection. First use transparent scalar code and small score buffers. Test that modifying a future token cannot change an earlier position's output; verify each query-to-KV-head mapping.

**T03 completion record — 2026-10-02**

- **Status:** implemented; owner review pending. GPT-6 Astra found no actionable
  issues and independently passed all 39 integration tests, formatting, Clippy,
  and the NumPy fixture check before commit.
- **Files:** `src/attention.rs`, `src/lib.rs`, `tests/attention.rs`,
  `fixtures/t03-inputs.json`, `fixtures/t03-expected.json`,
  `reference/attention.py`, `docs/lessons/03-attention.md`, and this record.
  No new dependencies. Projection biases are absent in this synthetic slice;
  real-checkpoint semantics remain a separate qualification.
- **Commands/results:** `python3 reference/attention.py --write`, followed by
  `python3 reference/attention.py`, passed generation/verification.
  `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, and `git diff --check` passed. 39 integration tests (4 T03)
  passed; ignored log `docs/evidence/t03-test.txt`. Targeted
  `cargo test --test attention -- --nocapture` compared every named intermediate.
  Q/K/V matched exactly. Rotated Q/K, context, and output had max absolute error
  `1.19209290e-7`; largest scale-aware relative error was `9.63204570e-7`.
  Tolerances remain `atol=1e-5, rtol=1e-4`.
- **Trace/layout:** Q/context are `[tokens,query_heads,head_dim]`, K/V use
  compact kv_heads. For 4 query heads and 2 KV heads, mapping is `[0,0,1,1]`.
  A score is the rotated Q/K dot product divided by sqrt(head_dim). At token
  zero the one-entry probability is exactly 1, so context copies its group's
  value row. A prefix slice enforces causality; altering the third token leaves
  the first two outputs exactly unchanged.
- **Mechanical sympathy:** consecutive features in each dot/value loop; one
  score/probability row reused across tokens and heads; no full T*T score
  allocation or duplicated K/V. Named activations retain owned scratch for
  inspection; this path allocates and is not claimed to be optimized.
- **Failure contract:** invalid shape/grouping/length/checked arithmetic or
  non-finite inputs/intermediates returns no valid trace. Caller input and
  weights are immutable. Aggregate scratch-byte arithmetic is checked.
- **Exercise:** map 6 query heads to 2 KV heads; identify keys visible at token
  index 2. Unsolved; see `docs/lessons/03-attention.md`.
- **Limitations:** synthetic uncached attention; no performance, full-model,
  memory-footprint, or hardware-acceleration claim. Same observed T01 runtime.
- **Next:** T04, authorized after this reviewed ticket's commit.

- [ ] **T04 — Decoder and logits.** Compose the two-layer model, feed-forward block, residuals, final normalization, and output projection. Save named intermediate values for one fixture. Compare every block and final logits against NumPy. Add a tiny token-ID generation demo; synthetic weights are not expected to produce useful language.

**Gate:** One complete token-to-logit trace is understandable without an external model runtime. The readable scalar implementation remains the correctness anchor after optimization.

### M2 — One real pretrained model

- [ ] **T05 — Checkpoint inventory and validated loading.** Inspect the selected Qwen repository; pin an immutable revision and record license, hashes, tensor inventory, tokenizer/config/generation assets, and tool versions. Reuse safetensors parsing; validate tensor names, shapes, dtype, shard coverage when present, file bounds, and dimensional arithmetic. Decode the checkpoint's BF16 weights into FP32 for the first reference. Share tied embedding/output storage rather than copying it. Do not load the Python reference and Rust model simultaneously during memory measurements.
- [ ] **T06 — Tokenization and real-model parity.** Reuse tokenizers and implement only the selected template behavior, validated against its pinned reference. Test whitespace, Unicode, special tokens, and message boundaries; avoid double insertion of BOS/EOS. Compare selected intermediate values and teacher-forced logits on short prompts. Keep the reference small: no full `[long_sequence, vocabulary]` logits dump unless necessary.
- [ ] **T07 — Uncached generation CLI.** Implement a simple greedy loop that recomputes the prefix as a correctness baseline. Respect configured stop tokens and output/context limits. Print timing separately from generated text. Generate a short answer from the real checkpoint; document that uncached execution is intentionally inefficient.

**Gate:** A real model runs through our Rust mathematics, and its behavior is validated against a pinned independent implementation. A model's advertised maximum context does not become our validated capacity.

### M3 — Cached, incremental inference

- [ ] **T08 — KV storage and one-token decode.** Implement explicit cache capacity and processed-position accounting. Compare cached and uncached logits at every teacher-forced step. Test exact capacity, one-token overflow, reset, repeated decode, and GQA. Use the invariants in Section 4.4.
- [ ] **T09 — Prefill and incremental append.** Start with a correct token-by-token prefill using the cache. Then add small chunk processing only when that path passes. For a chunk starting at position `p`, query row `j` can attend only through position `p+j`. Verify one-shot, chunked, and token-by-token results match. Measure them separately; do not describe a serial prefill loop as optimized matrix-matrix prefill.
- [ ] **T10 — Generation state and sampling.** Separate model stepping from sampler policy. Add deterministic greedy behavior, seeded temperature sampling, and one filtering strategy only when needed. Validate stop conditions, bad parameters, and non-finite logits. Preserve token IDs while decoding text without corrupted partial Unicode. Test cancellation and the sampled-but-not-yet-cached token case.

**Gate:** Correct cache reuse, no hidden truncation, and reproducible continuation. Restarting the model for every request or tool return fails this milestone.

### M4 — Explain and improve the measured bottleneck

- [ ] **T11 — Reproducible benchmark harness.** Implement Section 6 before changing performance-critical code. Record load, prefill, steady decode, continuation, and memory separately. Add layer/operator timing in diagnostic mode. Preserve raw results and configuration identity.
- [ ] **T12 — Scratch reuse and allocation audit.** Use measured allocation profiles to remove redundant copies and per-step large allocations. Preallocate attention workspace to the admitted capacity. Keep diagnostic tracing out of timed paths. Test repeated generation/reset and failure cleanup.
- [ ] **T13 — One CPU-kernel improvement.** Select one expensive operation from the profile. First investigate loop ordering/blocking and compiler output; then one explicit vectorized or threaded implementation where justified. Keep a callable scalar reference. Benchmark realistic model shapes, not only convenient square matrices. Do not add both threading and SIMD in one unexplained change.
- [ ] **T14 — Prefill versus decode experiment.** Compare matrix-vector decode and chunked matrix-matrix prefill, varying the chunk size within the memory budget. An Accelerate/BLAS comparison may be a labeled development benchmark, not a replacement for the learning implementation. Record where data movement, arithmetic, or launch/setup cost appears to dominate.

**Gate:** At least one reproducible optimization experiment, including a negative result when that is what measurements show. Keep an optimization only with evidence or a clear simplification benefit. No obligation to beat mature runtimes at this stage.

### M5 — Understand quantization, then choose a representation

- [ ] **T15 — Weight-only quantization experiment.** Start with one explicit groupwise symmetric INT8 teaching implementation and a documented group size/scale rule. Test rounding, clamping, all-zero groups, tails, and reconstructed-value error. Keep norms and other sensitive small tensors in FP32 initially. Use float activations and FP32 accumulation. Separate a dequantize-then-multiply reference from a fused dequantize-and-dot candidate; avoid a full FP32 weight reconstruction per token.
- [ ] **T16 — Quality/memory/latency decision.** Compare unquantized and quantized teacher-forced logits, held-out token loss, memory, and both prefill/decode. Decide whether INT8 earns a retained execution path. A subsequent INT4 experiment is a new authorized ticket only after these results; use one documented existing packing scheme or clearly label an internal experiment, not a new general model format.

**Gate:** Lower file size is not reported as lower resident memory or faster inference without measurements. Freeze the quality acceptance rule before selecting the winning experiment. Core ML compression is a separate representation and must be qualified independently.

### M6 — Core ML and Neural Engine qualification

- [ ] **T17 — Tiny-model export experiment.** Add isolated, pinned Python conversion tooling. Export the already-verified synthetic graph first, using supported public Core ML APIs. Test a narrow shape range and CPU-only correctness before investigating ANE placement. Record export/compilation peak memory and load cost separately from inference. Do not invent private ANE bindings.
- [ ] **T18 — Pretrained cached Core ML path.** Qualify the real checkpoint only after T17. Evaluate persistent state and causal positions against Rust/reference results. Audit the actual OS/SDK requirements of the selected state APIs. Prefer an artifact that retains cache state; do not assume separately loaded prefill/decode artifacts share either weights or state.[^stateful]
- [ ] **T19 — Placement and end-to-end experiment.** Explicitly allow CPU + Neural Engine and exclude GPU inference. Compare CPU-only Core ML, CPU+ANE Core ML, and our CPU implementation with identical inputs and disclosed precision. Capture Core ML performance reports/available traces. Attribute actual expensive work; “ANE allowed” and supported/preferred-device metadata alone do not establish measured ANE execution.[^compute-units][^coreml-profile]

**Important boundary:** MLX documents CPU/GPU execution; it is not our ANE execution route.[^mlx] Core ML permits CPU+ANE without GPU, but does not promise every operation runs on ANE.[^compute-units] Apple's transformer deployment guidance supplies experiments involving layouts, copies, and bandwidth—not universal performance guarantees.[^ane]

**Gate:** A written ANE result: numerical quality, observed placement, memory, speed, and maintenance cost. If conversion or useful ANE execution fails, preserve the CPU engine, produce a minimal reproducer, and mark acceleration blocked. Do not quietly enable GPU or call CPU fallback an ANE success.

### M7 — Native integration and iPhone 14 checks

- [ ] **T20 — Native boundary and one execution worker.** Add the smallest Swift integration that runs our CPU engine. Use opaque handles and explicit buffer/error ownership; prevent panics crossing FFI. The boundary is a request/chunk/token step, not individual transformer operators. Cancellation is explicit and event delivery bounded. Keep heavy work off the UI thread. Add Core ML as a second concrete path only after T19; keep its state native, not mirrored in a second Rust cache.
- [ ] **T21 — Physical iPhone 14 validation.** Build and sign with the owner's development setup; never modify account/signing credentials automatically. First run synthetic parity, then a short real-model generation with a measured admission budget. Test load/unload, memory pressure behavior, cancellation, background/foreground transitions, and sustained runs. Stop/suspend predictably when execution cannot continue. Label simulator/build-only results separately from physical-device results.

**Ownership decision:** Rust owns tokenization integration, token history, sampling, and policies. CPU sessions own Rust KV storage; Core ML sessions own native state. A thin native driver may invoke the shared Rust generation policy and the selected execution path. Do not duplicate the production sampler or keep two complete model copies just to cross the language boundary. Finalize this small contract from the working implementations, not from speculative traits.

**Gate:** Same numerical/generation contracts across paths, with declared exceptions. Desktop success does not certify the phone. iPad remains unqualified until a device is actually tested.

### M8 — macOS Workbench and Hugging Face discovery

- [ ] **T22 — MLX model search.** Use the Hub API and its current metadata/filter capabilities, not HTML scraping.[^hub] Support text search, pagination, cancellation, errors, and optional publisher/library filters. Treat missing metadata as unknown. Do not assume `mlx-community` is the entire catalog.
- [ ] **T23 — Compatibility and installation.** Display separate identities for discovered repository, source checkpoint, and execution artifact. Label entries as runnable, recipe available, discovery only, or unsupported. A Qwen source or converted Core ML counterpart is not silently presented as the exact MLX quantization. Pin revisions; check disk space, integrity, interrupted downloads, and atomic install. Never execute repository code. Gated-model authentication is deferred until explicitly scoped.
- [ ] **T24 — Minimal native UI.** Add search, installed-model details, local chat, cancellation/unload, context budget, and diagnostics. Disclose execution policy and validation status. Run offline after installation. Keep inference math outside SwiftUI.

**Gate:** The search requirement works without falsely promising that every MLX result runs on ANE or in our Rust loader.

### M9 — A small agent and one educational application

- [ ] **T25 — Agent loop and cache-preserving continuation.** Implement a bounded sequential model → validate tool request → execute → append result → model loop. Start with deterministic mock tool calls, then qualify the model's actual tool-use capability. Use a calculator or read-only user-selected content; no arbitrary shell. Enforce typed arguments, allowlists, step/output budgets, cancellation, and permissions. Tool text is untrusted data. Persist transcript separately from volatile cache.
- [ ] **T26 — Educational application.** Choose one exercise type with the owner before adding content-specific structure. Reuse the engine API. Keep curated lessons, deterministic grading, generated explanations, and progress storage separate. Generation quality is a product qualification gate, not a consequence of numerical correctness.

**Gate:** Agent continuation reuses the exact valid token prefix; editing history forces invalidation/replay. No new inference implementation appears in either app. The small first model is not presumed to be a capable coding agent or a reliable teacher.

### M10 — Later GPU and long-context research

- [ ] **T27 — First custom Metal execution slice.** Requires explicit authorization to enter GPU scope. Implement one operation against the scalar reference, then extend toward a full forward pass. Reuse numerical fixtures and benchmark contracts, not undocumented buffer-sharing assumptions. Learn dispatch, synchronization, lifetime, and memory layout before fusion. No compiler framework required.
- [ ] **T28 — Larger-context continuation.** Increase capacity incrementally within the selected model's valid positional configuration and device budget. Measure prefill, decode, cache bytes, and model → tool → model continuation. Begin with contiguous cache and exact prefix reuse. Consider quantized KV, paging, branching, or speculative decoding only as individually justified experiments; neither arbitrary 128K support nor a context-extension method is promised.

**Gate:** Each new backend/context/device combination earns independent correctness and resource evidence. A faster kernel is not automatically a faster application.

## 6. Benchmark contract

### Test groups

**Numerical references:** scalar Rust, independent NumPy, and pinned Transformers with identical token inputs. They answer correctness questions, not market-performance questions.

**Policy-matched execution:** our CPU path, minimal Core ML CPU-only, and Core ML CPU+ANE. Record execution policy, precision, state handling, and measured placement.

**External product references:** pinned MLX-LM and llama.cpp configurations on the same Mac. GPU execution is allowed only for these explicitly labeled development comparisons until M10. Both provide useful inference/benchmark facilities; align what is timed rather than treating all reported tokens/second as interchangeable.[^mlx-lm][^llama-bench]

Do not compare the owner's M1 Pro against published M4/A100 results and call the difference an engine optimization. Do not promise a percentage improvement before collecting a baseline.

### Workloads and measurement

Start synthetic tests at 1–32 tokens. For the real model, propose prompt lengths 32, 128, 512, and 1,024, with 128 generated tokens where capacity permits. Longer tests are conditional on memory and runtime feasibility. Use fixed token IDs and hash the workload. Separate normal EOS-respecting generation from fixed-length teacher-forced throughput tests.

Report:

| Metric | Definition |
|---|---|
| Cold preparation | Download, conversion, compilation, and initial loading, separately. |
| Warm TTFT | Request submission to first token; also retain first-visible-text timing. |
| Prefill | Uncached prompt tokens divided by elapsed prompt-processing time. |
| Decode | For N > 1, `(N - 1) / (last_token_time - first_token_time)`. |
| Continuation | Appended tool-result/new-turn submission to next token with valid prefix retained. |
| Memory | Logical weights/cache/scratch accounting plus measured process footprint and memory pressure. |
| Sustained behavior | Time-series throughput and thermal/power observations during an opt-in repeated workload. |
| Quality | Numerical error, teacher-forced loss, and later task-level results. |

Separate backend timings from tokenization, sampling, text conversion, FFI, and UI delivery. Synchronize asynchronous work before stopping its timer. Keep cold and warm caches explicit. For short benchmarks, propose three warmups and ten repetitions; report median and dispersion. Do not label ten request samples a robust p95 estimate.

Record actual device/chip, memory, OS build, power conditions, compiler/dependency revisions, model revision/hashes, storage and compute dtype, quantization scheme, context capacity/used positions, threads, prompt hash, output length, and cache state. Preserve raw JSONL and a short Markdown report. Never populate a measured field with theoretical bandwidth/FLOPS or an external published result.

Measure energy only with a documented instrument and scope. Otherwise write `unavailable`, not an invented joules/token number. Whole-device estimates are not engine-only measurements.

### Memory protection

Estimate weights + allocated cache + workspace + tokenizer/transcript/app buffers before admission. Use checked arithmetic. The **planning default** for Mac experiments is a conservative 6 GiB engine working-set budget, adjustable after observing actual headroom; it is not a Core ML allocator guarantee or an iPhone budget. Record actual footprint and system pressure. Stop an oversized conversion rather than repeatedly provoking out-of-memory behavior.

Do not allocate an attention matrix or all-position vocabulary logits for a long prompt without including it in the estimate. Do not count shared CPU/GPU pages twice. A small file or memory map does not prove a small resident working set.

## 7. Test and development rules

Portable tests must run without model downloads, Apple hardware, Python, or network access. Reference regeneration and real-model/hardware tests are explicit opt-ins. Tiny fixtures may be checked in; pretrained weights, compiled models, credentials, and sensitive user prompts may not.

Once applicable, use ordinary commands:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Document release-mode benchmark commands when the harness exists; do not list fictional commands as already implemented. Pin the tested Rust toolchain without upgrading unrelated code. Separate the lightweight NumPy environment from optional Transformers/conversion tools so T00 never installs an ML stack.

Do not use `unsafe` in the initial numerical implementation. Later SIMD/FFI use requires a narrow boundary, documented safety invariants, and reference-based tests. Prefer no runtime dependency when a clear standard-library implementation is adequate; do not reinvent security-sensitive parsers or tokenizers to meet an arbitrary dependency count.

Retain a short experiment record for optimizations: hypothesis, implementation difference, workload, correctness results, measurements, interpretation, and retain/revert decision. Remove unsuccessful complexity rather than leaving permanent unused switches.

## 8. Review gates and unresolved choices

The owner reviews the tiny forward pass before proceeding to real weights; cached parity before performance tuning; benchmark evidence before adopting an optimization; and ANE evidence before representing the engine as accelerated.

These choices remain unresolved and have designated decision points:

| Choice | Resolve at |
|---|---|
| Exact Mac CPU/GPU configuration, installed OS/toolchains | T00 on the actual host; do not infer from “M1 Pro.” |
| Immutable first checkpoint revision and reference versions | T05. |
| Full-model numerical tolerances and quality floor | T06/T11, before performance selection. |
| Final CPU quantization and any 4-bit scheme | T15–T16 results. |
| Core ML graph/state representation and deployment floors | T17–T19 against installed public SDKs. |
| iPhone admission budget and validated context | T21 on the physical device. |
| Educational subject and intended audience | Before T26; do not invent them now. |
| Whether advanced KV/Metal work is worth its complexity | Individual experiments after authorization. |

No speed, device compatibility, ANE-placement, conversion-success, or mobile-memory result has been measured in preparing this plan.

## 9. Ticket completion record

Append this compact record under each completed or blocked ticket:

```text
Status: implemented / blocked / not run
Files changed:
Commands actually run:
Results and evidence paths:
Numerical or memory-layout explanation:
Known limitations:
One learning exercise:
Owner review: pending / acknowledged by owner
Next ticket: proposed only, not started
```

Keep the roadmap plain Markdown. Building, testing, and understanding the repository must never depend on access to Codex.

## Sources and verification boundaries

Sources reviewed on 2026-10-01. They establish APIs, model properties, or reference-tool behavior, not performance results for this project. Implementation proposals and calculated examples are identified as such above. Recheck and pin relevant versions when each ticket starts.

[^codex]: OpenAI, **Custom instructions with AGENTS.md**. Project instruction discovery and concise repository guidance. `https://developers.openai.com/codex/guides/agents-md`

[^qwen-config]: Qwen, **Qwen2.5-0.5B-Instruct configuration**. Model type, dimensions, BF16 storage declaration, tied embeddings, and positional settings. `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json`

[^qwen-impl]: Hugging Face Transformers, **Qwen2 documentation and linked implementation**. Use a pinned implementation for exact operator conventions. `https://huggingface.co/docs/transformers/model_doc/qwen2`

[^tokenizers]: Hugging Face, **Tokenizers**. Existing tokenization implementation, including Rust. `https://huggingface.co/docs/tokenizers/index`

[^safetensors]: Hugging Face, **Safetensors**. Existing tensor storage and parsing ecosystem. `https://huggingface.co/docs/safetensors/index`

[^stateful]: Apple, **Stateful Models — Core ML Tools**. Stateful model representation and cache examples; validate selected SDK/OS requirements. `https://apple.github.io/coremltools/docs-guides/source/stateful-models.html`

[^compute-units]: Apple, **Core ML Tools ComputeUnit definitions**. `CPU_AND_NE` allows CPU and Neural Engine but excludes GPU. `https://raw.githubusercontent.com/apple/coremltools/main/coremltools/__init__.py`

[^coreml-profile]: Apple, **Deploy machine learning and AI models on-device with Core ML**, WWDC24. State, performance reports, and compute-plan inspection. `https://developer.apple.com/videos/play/wwdc2024/10161/`

[^mlx]: MLX maintainers, **MLX README**. CPU/GPU execution and unified-memory model. `https://github.com/ml-explore/mlx`

[^ane]: Apple Machine Learning Research, **Deploying Transformers on the Apple Neural Engine**, 2022. Historical layout/copy/bandwidth guidance to test, not unconditional modern-chip rules. `https://machinelearning.apple.com/research/neural-engine-transformers`

[^hub]: Hugging Face, **Hub API Endpoints**. Current model discovery API reference. `https://huggingface.co/docs/hub/api`

[^mlx-lm]: MLX-LM maintainers, **MLX-LM README**. Reference inference and prompt-cache functionality. `https://github.com/ml-explore/mlx-lm`

[^llama-bench]: llama.cpp maintainers, **llama-bench README**. Benchmark modes and timing scope; tokenization and sampling are not included in its core timing. `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/llama-bench/README.md`
