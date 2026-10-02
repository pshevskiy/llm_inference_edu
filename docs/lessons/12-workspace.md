# 12 — Own and reuse the working buffers

The allocation audit found 363 System allocator requests per Qwen cached step,
allocating 2,655,104 bytes cumulatively. That is allocation traffic, not peak RSS.
The scalar arithmetic still dominates the observed T11 profile, so this ticket
claims simpler ownership and less allocation traffic, not a pretrained speedup.

`CachedSession::new` allocates one `StepScratch` for the session. All layers use
it sequentially. Hidden/norm/residual/FFN buffers are H-wide; attention keeps
Q and rotated Q H-wide, K/V/rotated K KV-wide, and context/output H-wide. Three
FFN vectors are F-wide. Scores and probabilities reserve the admitted capacity,
then attention sees only `[..=position]`. No capacity tail enters a softmax.
The checked resident payload is `(8H + 3KV + 3F + 2C) * 4` bytes: 88,832 bytes
for Qwen at capacity 32. It is separate from compact per-layer KV storage and
caller-owned vocabulary logits. The session checks KV + scratch against its
128 MiB bound before either allocation. Benchmark admission also includes
weights, output rows and tokenizer/metadata allowances within its 6 GiB plan.

`step_into` first validates session, token, capacity and exact output length.
Embedding overwrites hidden; normalization and projections overwrite their
whole outputs. Attention explicitly zeroes context because it accumulates value
rows. RoPE outputs and the current KV row are overwritten. FFN and final logits
then overwrite their buffers in the same arithmetic order as before. Processed
length increments only on complete success. Errors after writes invalidate the
cache; reset changes logical state, and the next valid step overwrites scratch.
Stale bytes never become a valid prefix.

`prefill_into` and `append_into` reuse the same caller logits on each serial step.
The convenient Vec-returning methods allocate one final vocabulary row per call.
`GenerationSession` keeps its row across resets and distinguishes allocated storage
from ready logits with a Boolean. An emitted token clears readiness but retains
storage. Cooperative cancellation and history/processed counts retain T10 rules.
Greedy numerical generation performs no allocator requests in audited scopes.
Temperature sampling and text decoding can allocate; normal forward inspection,
loading, construction, diagnostic profiles and output records are separate.

After the change, both tiny and Qwen audits measured zero allocation/reallocation
requests for buffered step, buffered three-token prefill and greedy next-token.
Legacy step and prefill each allocate one output row. Qwen session construction
now makes 65 requests instead of 49, moving reusable work into setup. The
isolated test checks repeated resets, exact reference logits, invalid admission,
and reset recovery after a forced FFN overflow. Numerical tolerances are unchanged.

Reproduce the audit separately from latency measurements:

```sh
cargo run --offline --release --example allocation_audit
target/release/examples/allocation_audit models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775
cargo test --offline --test workspace
```

Instrumentation is the pinned, dependency-free development crate
[`stats_alloc` 0.1.10](https://docs.rs/stats_alloc/0.1.10/). No production allocator
or Rust toolchain changed. Counters cover Rust global System allocation requests,
not arbitrary native-library allocation or a whole-process allocation-free claim.
Do not time this instrumented executable as a performance baseline.

**Trace:** With prompt `[1,5,7]`, prefill overwrites the same 64-value tiny logits
row three times and leaves processed length 3. Sampling emits 41 and marks logits
unready. The next call evaluates 41 into that same row at position 3, emits 53,
and leaves processed length 4/history length 5. No row is copied or reallocated.

**Exercise:** If capacity is 8 and processed length is 3, which score/probability
indices may be read while evaluating the next token? Which scratch buffer must
be cleared before accumulation, and which outputs are completely overwritten?
