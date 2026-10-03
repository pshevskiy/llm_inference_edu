# T19 — Matched policy experiment; ANE acceleration unqualified

The experiment is implemented. Retain our Rust CPU engine. This artifact has
not earned an ANE backend: allowed devices and anticipated placement are not
execution proof, and allowing NE showed no useful improvement in this short run.
This negative outcome does not pass the M6 acceleration gate. Owner review pending.

Actual host: MacBookPro18,3, Apple M1 Pro,16GB, macOS26.6.2/build25G83,
Xcode27.0/build27A266a. The owner cleared the license gate and then authorized
bounded repeated tests with “do it autonomously.” Global developer selection
remains Command Line Tools; Xcode is selected per command. GPU inference excluded.
No dependencies or production numerical implementation changed.

## Hypothesis and interface

Compare identical public token workloads on our single-thread Rust CPU path,
Core ML CPU_ONLY, and CPU_AND_NE. Core ML system-managed threading and Python
bridge/policy costs are explicitly included; this is not a comparison of identical
kernel implementations. Rust uses FP32 KV; native state uses approximate FP16
high/residual pairs. Both have FP32 weights/compute and capacity32.

The original T18 artifact returned49diagnostic arrays. For timing, change only
its public output lists to logits. Restoring the lists must produce an identical
protobuf; weight bytes must remain identical. No arithmetic, precision, cache,
checkpoint or reference gate is silently changed. Compile the new interface once
and qualify it separately before measurements. This removes extra output
marshalling; it does not establish why Core ML is faster.

## Independent correctness

Pinned Transformers4.46.3/torch2.6.0 CPU FP32, eager attention, one thread, no cache
or network builds independent full-prefix logits and greedy histories. Actual
versions are checked against the frozen T06 reference, not copied as labels.
The oracle includes original public cases and both complete benchmark histories.

All three paths pass24teacher-forced logit positions at the unchanged gates
(atol.001, rtol.0001), all greedy IDs exact. Worst gate fraction: Rust.23070911,
both native policies.58195120. Earlier diagnostic runs also passed392named-array
checks per native policy. T18 remains the intermediate-activation qualification;
the new output interface necessarily exposes only logits.

Every warmup/repetition matches the independent emitted IDs, EOS, continuation
next token and processed/pending counts. Greedy emits `[358,2776,264,48948]`,
processed5before continuation and8after, next token614pending. Teacher-forced
emits `[358,1058,279,0]`, same counters, next token264pending. No truncation.

## Measured short workload

`benchmarks/qwen-short.json`: prompt `[9707,0]`, output budget4, continuation
`[264,1172]`,3warmups and10repetitions per mode/path. Each process runs alone;
180s cap per benchmark path,240s preparation cap. Timeout kills only its freshly
created child process group. Actual full child durations Rust75.22s, CPU_ONLY28.08s,
CPU_AND_NE30.98s include qualification/loading and later diagnostic work.

Greedy request medians (MAD in parentheses):

| Path | Prefill ms | TTFT ms | Decode tokens/s | Continuation ms |
|---|---:|---:|---:|---:|
| Rust CPU | 650.491 (.889) | 651.074 (.890) | 3.067 (.003) | 976.034 (.891) |
| Core ML CPU_ONLY | 73.246 (.467) | 73.371 (.467) | 27.521 (.314) | 108.704 (.806) |
| Core ML CPU_AND_NE | 73.733 (.321) | 73.895 (.337) | 27.589 (.079) | 108.346 (.397) |

First text chunk ready medians651.080/73.389/73.913ms; no UI delivery included.
Reset/cache construction, model load and initial tokenization are outside request
timing. Prefill evaluates both prompt IDs; TTFT includes greedy selection. Token
emission timestamps include intervening text preparation; the final token's text
conversion follows its emission timestamp. Decode uses `(N-1)/(last-first)`.
Continuation evaluates the pending token before appended IDs, then samples next.
Teacher-forced throughput has the same output count; its TTFT is unavailable.
Backend timings remain separate from sampling/text timings in raw records.

Warmups include lazy initialization. Filesystem/runtime caches are uncontrolled;
backend order is fixed, not randomized. Python0.20.3 and Rust0.21.4 tokenizers use
an equivalent checked UTF-8 stream contract; Python lacks the newer streaming
entry point, so the development wrapper reproduces that established algorithm.
Portable tests cover split bytes, hidden special tokens, ties, EOS and admission.
This wrapper is reference tooling, not a second production engine or sampler.

The native CPU execution is much faster on this scoped short workload. The small
CPU_ONLY/CPU_AND_NE differences do not establish a meaningful NE improvement or
statistical equivalence. No p95, long-context, product-wide, sustained thermal,
mobile or backend-independent speed claim follows from these ten samples.

## Memory and environmental scope

Logical FP32 weight bytes1,976,131,072; KV786,432/session on both paths. Paired
half state saves no logical KV bytes. Rust retains its explicit reusable workspace
and caller logits buffer. Native internal scratch/layout is system-managed.

| Child path | Peak RSS B | Peak footprint B |
|---|---:|---:|
| Rust | 3,054,485,504 | 3,054,160,656 |
| CPU_ONLY | 5,463,293,952 | 318,477,472 |
| CPU_AND_NE | 5,388,337,152 | 323,949,584 |

These peaks include imports, mappings, qualification and, for native, post-timing
compute-plan inspection. They are not engine payload, file size or total-system
memory. Compiler/runtime services are not attributed. Different accounting
between mapped pages and footprint does not prove the native model consumes
less memory. No double-counted shared pages or inferred mobile admission budget.
AC power snapshots are saved. `pmset` reported no recorded thermal/performance
state; this is unavailable thermal qualification. Calibrated energy unavailable.

## Observed CPU work and unresolved framework tracing

Xcode's public Core ML and Neural Engine instruments are available. Earlier Python
recordings lacked framework events. A native Swift tiny control captured8unique
custom signpost events (load plus3predictions), ruling out total signpost failure.
Core ML events remained empty, so the issue is not specific to the Python bridge.

A native control of the *matched logits-only artifact*, CPU_AND_NE, passed all8
frozen greedy checks and captured18unique custom events: load plus8synchronous
prediction intervals. The public XML analyzer resolves backtrace references,
filters the target PID, handles missing stacks and deduplicates alternate event
views. It found330sampled Core ML prediction stacks;325also contain
`BNNSGraphContextExecute_v2` and `BnnsCpuInferenceOperation::ExecuteSync`.
All325fall within our8marked prediction intervals. This attributes actual graph
execution to CPU. These counts are not wall-time percentages or proof of every
operator's device. The first prediction can include lazy initialization.

Framework Core ML and ANE tables still have zero rows. Both controls report
`Data stream: Time Mapping`; its internal numeric severity is not interpreted.
Custom interval timestamps remain ordered and encompass the CPU samples. Exact
framework tracing cause is unresolved. Empty ANE tables do not prove zero ANE.
The lean anticipated compute plan has2280CPU-preferred operations and2266unknown
(2254constants plus12reshape/transpose operations), no NE-preferred work. This is
supporting metadata, not a replacement for observed stacks or device placement.

Astra independently verified CPU-stack and interval attribution. The decision is
**ANE acceleration unqualified/blocked for this artifact**. A future ANE attempt
must separately qualify any new layout/precision artifact and provide healthy
attributed device traces. Keep the scalar Rust reference and explicit CPU policy.

## Evidence and validation

Ignored `docs/evidence/t19-benchmark-v1/`: source/host/artifact/oracle identities,
compile hashes, per-position logits and gates,20rawsamples/path, environment
snapshots, process logs, comparison and completion. `t19-control/`: actual binary
hashes, custom/native traces, XML, stdout and `qwen-cpu-correlated.json`.
Earlier `t19-final`, `t19-instruments` and license-readiness results are retained
as history. Trace bundles contain incidental process/device metadata; never Git.

Release build, formatting, Clippy with warnings denied,88integration+3unit Rust
tests, Python syntax and9portable Python tests passed. Logs are listed in PLAN.
Astra reviews cover code, contracts and evidence; owner understanding remains
pending. Reproduction and one unsolved exercise are in lesson19.
