# 28 — Larger contiguous CPU context, earned in stages

Context capacity is a storage/admission parameter, not a guarantee of useful
answers. `Decoder::from_qwen` now accepts explicit CPU experiment capacities up
to 2,048, bounded by the checkpoint's positional configuration. Existing native
callers retain their separate eight-position admission and 32-position allocated
cache. The CLI defaults to 32; its optional final argument selects a larger
cached context. The uncached CLI baseline remains bounded to 32 positions.

For example, after the relevant T28 stage passes:

```sh
cargo run --offline --release -- \
  models/installed-qwen/7ae557604adf67be50417f59c2c2f167def9a775 \
  'What is 2 plus 3?' 16 'Be brief.' cached 256
```

The whole prompt and output reserve must fit. There is no silent truncation,
rolling window, precision change or model substitution. Cache capacity cannot be
resized under an active session: choose it before construction; changing history
requires reset/replay. Changing the capacity changes per-head stride, so tests
compare the same prefix in differently sized caches instead of assuming stride
has no effect.

## What changes in memory

Each layer owns contiguous K and V buffers in `[kv_head, capacity, head_dim]`
order. Increasing capacity extends the position dimension; weights remain
immutable, with no duplicated vocabulary output head. Attention reads only the
valid prefix, never uninitialized future positions. The reusable score and
probability vectors grow linearly in capacity. The production cached path does
not materialize a square attention matrix or all-position vocabulary logits.

For pinned Qwen, logical FP32 KV payload is
`2 × 24 × capacity × 2 × 64 × 4` bytes:

| Capacity | Logical KV payload |
|---:|---:|
| 32 | 0.75 MiB |
| 128 | 3 MiB |
| 256 | 6 MiB |
| 512 | 12 MiB |
| 1,024 | 24 MiB |
| 2,048 | 48 MiB |

The FP32 weights still occupy 1,976,131,072 logical bytes. File sizes, calculated
payload, sampled resident memory, maximum RSS and macOS peak process footprint
are recorded separately. The 6 GiB planning estimate includes scratch and
allowances; it is not an operating-system memory limit or a mobile qualification.

## Verification in execution order

`reference/long_context.py` verifies pinned assets/versions, records run identity,
then uses CPU FP32 eager Transformers. Cached evaluation exports one vocabulary
row at each teacher-forced position to an ignored file. The Rust harness streams
those rows into two reusable vocabulary buffers and compares every finite value
with unchanged `atol=0.001`, `rtol=0.0001`, plus exact greedy IDs. It rejects wrong
revision, workload/hash, shape, row size, gate and raw bytes before execution.

Selected independent full-prefix calibrations use only the final logits row, via
pinned `num_logits_to_keep=1`. Eager reference attention still creates quadratic
scores during those calibrations: their explicit memory estimate includes that
cost. Such matrices are not silently attributed to the production cached path.
Identity is captured before long work; completion/failure records are separate.
Interrupted runs retain identity and partial raw observations. Binary identity is
captured at startup rather than re-read from a path that a rebuild could replace.

After numerical qualification, one bounded request measures prefill, one decode
step and continuation. A separate independent oracle follows the actual sampled
IDs and appended suffix; the teacher-forced workload alone would not prove this
request after its first sampled token. Timing excludes comparison work. These
long request timings are single samples, without a robust distribution/sustained
thermal/energy claim. Source identity, executable identity, host, pressure and
raw position records accompany each stage; device evidence does not transfer to
an iPhone or a different backend.

## A continuation trace at capacity 512

A 500-token prompt evaluates 500 positions. Two emissions leave evaluated/history
lengths 501/502: the second emitted ID is pending. Appending four result IDs
consumes that pending ID at position 501, then results at 502–505. The valid prefix
and history both reach 506. The next emitted ID creates history length 507 with
processed length still 506. The 501 already evaluated positions are retained.

The actual calculator probe is separate from fixture suffix timing. It renders
the previously qualified plain-message chat branch, requests one strict JSON
calculator call, uses explicit EOS IDs, validates/executes only the calculator,
and appends the trusted numeric envelope as a user data message. The emitted EOS
is already in history; result framing does not duplicate it. Three public fixed
arithmetic probes are checked against independent FP32 logits and exact greedy
IDs at every emitted position. Tool-selection/schema/answer quality is recorded
separately. Even three successful probes would not establish a reliable general
agent, teacher, tool template, or coding assistant.

See the current T28 record in `docs/PLAN.md` for stages actually run, numerical
results, measurements and the calculator outcome; table capacities are formulas,
not a claim that every stage has passed.

Exercise: at capacity 512, calculate the **per-layer** K-buffer offset for KV head
1, position 501, channel 7, with head dimension 64. Then explain why copying just
the first `processed × head_dim` floats would not preserve both KV heads when
moving to a differently sized cache. Do not solve it by running the engine.
