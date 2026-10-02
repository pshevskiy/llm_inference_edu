# Reproducible CPU measurements

Build from the repository root immediately before running:

```sh
cargo build --offline --release --example benchmark
target/release/examples/benchmark tiny fixtures/tiny-model.json benchmarks/tiny.json docs/evidence/NEW-tiny
```

Result directories must be fresh. `raw.jsonl` preserves each repetition;
`metadata.json` contains the public token workload, actual host, model/lock/source
hashes and executable hash. Runtime source/compiler snapshots are not certified
build provenance. Initial metadata is written before repetitions; `run_status`
marks incomplete runs. Inspect it before consuming results. `summary.json`
reports median/min/max/MAD. Ten samples do not support a robust p95 claim.
`diagnostic.json` measures operator scopes separately from timed repetitions.
No new dependency or instrumentation is enabled in ordinary generation.

The Qwen checkpoint must already be explicitly downloaded and verified:

```sh
cargo build --offline --release --example benchmark
/usr/bin/time -l target/release/examples/benchmark qwen models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775 benchmarks/qwen-short.json docs/evidence/NEW-qwen --single
```

`--single` is a smoke test, not a distribution or optimization claim. Remove it
only after owner opt-in for sustained hardware tests. The checked-in short
workload stays within the currently qualified 32-token capacity. Proposed
32/128/512/1024-token prompts plus 128 outputs are not yet admitted. Never shrink
or truncate an oversized workload silently. Reserve prompt + requested output +
continuation + its next emitted token, including emitted-but-not-evaluated state.

The model-load scope includes verification/BF16 expansion/decoder validation;
tokenizer loading is separate. Filesystem cache state is uncontrolled: this is
an initial load, not a certified cold load. Download/conversion/compilation are
preparation, not inference. Fixed teacher-forced output is separate from greedy
EOS-respecting generation. True TTFT exists only for greedy; fixed output records
logits readiness. Decode is `(N-1)/(last emission-first emission)`; a one-token
output has no decode throughput. Serial prefill processes every prompt token;
its tokens/s is prompt count divided by `prefill_ns`. Backend times exclude
sampling/text conversion; text readiness is a complete decoded chunk, not UI
rendering. Continuation evaluates the last pending emitted token, then appended
IDs, then samples one new token that is again pending.

Logical FP32 weights, allocated KV and conservative live-vector scratch payload
are distinct from file bytes, point RSS, peak RSS and macOS peak footprint.
`/usr/bin/time -l` captures process-wide peak memory including loading; it is not
a steady decode working-set measure. Sandbox restrictions can make `ps` RSS
unavailable. Thermal commands may return error text despite success status;
that is unavailable evidence, not a normal thermal reading. Metadata preserves
these responses. Energy is unavailable without an instrument. No ANE, iPhone,
GPU, external-runtime or sustained thermal claim follows from this CPU baseline.

T12 adds `allocation_audit` as a development-only executable. See
`docs/lessons/12-workspace.md`. Its allocator counters must remain outside latency
measurements. Benchmarks now reuse the caller logits row and session workspace;
metadata records resident workspace separately from the conservative scratch
estimate (workspace plus up to two legacy caller rows). Historical T11 records
retain their original allocation pattern and accounting.

T13's short projection experiment uses public synthetic values and actual Qwen
rectangular dimensions, not whole-model timing:

```sh
cargo build --offline --release --example kernel_benchmark
target/release/examples/kernel_benchmark docs/evidence/NEW-kernels
```

An immutable `identity.json` precedes samples; `completion.json` identifies a
completed run. Inspect both. `docs/experiments/13-lanes.md` records the decision
and limitations. Each kernel remains independently callable.

T14 preserves a rejected, development-only chunked projection experiment:

```sh
cargo build --offline --release --example prefill_experiment
target/release/examples/prefill_experiment docs/evidence/NEW-prefill
```

It rotates independent row calls and chunk1/2/4/8 on fifteen public numerical
workloads. `docs/experiments/14-prefill.md` records the negative result. Production
model prefill remains serial; projection timing does not certify cached batched
prefill or sustained model throughput.
