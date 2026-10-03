# 20 — Rust ownership through a native boundary

Swift submits a whole prompt or one token step. Rust keeps weights, tokenizer,
KV, logits, sampler and history together on one worker. We do not cross the
language boundary for each matrix multiplication or duplicate model/cache state.
The ordinary numerical crate still forbids unsafe code. Only the small
`apple/cpu_bridge` wrapper has documented unsafe pointer reads/writes.

Execution order:

1. `aie_open` validates a bounded payload and copies it, then returns a checked
   integer handle. The caller's memory can be released immediately after return.
   A process-wide reservation admits only one loading/live/closing worker.
2. That worker verifies and loads the model, then sends Ready or a terminal Error.
   Swift polls on its delivery queue, never performing numerical work on main.
3. Begin copies bounded IDs or UTF-8. Rust tokenizes text and validates the whole
   prompt/output budget before prefill. This first contract accepts plain text;
   it does not add chat/tool rendering or change the C32 qualification.
4. Next invokes our existing `GenerationSession`, evaluates the pending token
   when needed and emits a small event. The event has kind, status, token ID,
   evaluated-prefix count and history count, all fixed-width integers.
5. Each admitted command produces one response. Rust channels have capacity one;
   Swift also rejects another request until the current response reaches main.
   A rejected Swift request returns false and queues no callback. An admitted
   invalid request gets one failure callback. Oversized UTF-8 is scanned/copied
   only up to 65,537 bytes on the background queue and rejected, never truncated.
6. Cancellation uses an atomic flag independently of the request channel. It
   takes effect at a token boundary; it does not interrupt a matrix loop or model
   load. Clearing it explicitly allows safe continuation.
7. Close immediately invalidates the handle and requests cancellation. It never
   blocks main waiting for a worker. Another model remains Busy until the worker
   has actually dropped the old model/session, including an in-flight load.

Concrete trace, tiny prompt `[1,5,7]`, four new tokens:

| Event | Evaluated prefix | History length | Emitted ID |
|---|---:|---:|---:|
| Started | 3 | 3 | none |
| Token | 3 | 4 | 41 |
| Cancelled before next | 3 | 4 | none |
| Resume, Token | 4 | 5 | 53 |
| Token | 5 | 6 | 41 |
| Token | 6 | 7 | 50 |
| Stopped on output budget | 6 | 7 | none |
| Reset | 0 | 0 | none |

The final emitted token remains unevaluated. Cancellation preserves that exact
state, rather than pretending all emitted IDs already have K/V. Numerical
failure after a K/V write is terminal and reports zero valid prefix. Tests build
an independent finite model whose feed-forward arithmetic overflows after K/V
writes to exercise this boundary.

The C caller must supply real readable/writable, aligned allocations for the
stated lengths. Null/length/alignment checks cannot validate arbitrary addresses.
Panics unwind only inside Rust catchers; invalid pointers, OOM, aborts and signals
are not recoverable error codes. No foreign pointer survives a call. Stale IDs
never dereference freed allocations and are not reused.

The static library and native development client run on the tested Mac. The
current build explicitly targets macOS 26.5 because the existing Rust tokenizer
objects were compiled with that deployment floor. This is not an iOS build or a
portable older-macOS binary. Swift 6 checks concurrency; the narrowly documented
`@unchecked Sendable` client uses a lock for flags, immutable integer handle,
one polling queue and main-queue completion. Polling sleeps two milliseconds
only off main and closes on its bounded 60-second deadline. This is an ownership
check, not a latency or energy benchmark.

Core ML/ANE is still unqualified after T19's negative result. We expose our own
CPU engine here, rather than creating a second native state or hiding a fallback.

Exercise (do not implement yet): after the first tiny token event, main is busy
for 100 ms. Identify which layer prevents another request, how many response
events can be waiting, and the prefix/history counts if cancellation arrives
before the next token step.
