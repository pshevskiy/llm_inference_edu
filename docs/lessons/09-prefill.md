# 09 — Prefill and append without changing the causal contract

`prefill(ids)` requires an empty valid session. `append(ids)` admits a nonempty
chunk against token range and remaining capacity before any writes. Both return
only the last position's vocabulary row. Each admitted token uses `step`, so a
chunk starting at absolute position p gives local row j position p+j and reads
only through that position. There is no attention to future tokens in a chunk.

Trace: prefill `[1,5,7]` commits positions 0,1,2. Appending `[41,53]` evaluates
41 at 3 and 53 at 4; the session commits processed=5. Its last logits equal
uncached `[1,5,7,41,53]`. A later invalid ID or over-capacity request is rejected
before touching the prefix. If the second token of an admitted chunk fails
during execution, the earlier token and every old prefix are invalidated: a
partially updated cache cannot masquerade as a valid result.

The smallest implementation is serial. Chunk calls group validation and API
ownership, but do not batch linear projections or implement matrix-matrix
prefill. This keeps one-token and chunk semantics identical before optimizing.
Fixed compact K/V is reused; scratch and vocabulary output still allocate per
token. Intermediate chunk logits are dropped rather than saved as T*V.

Tests compare each partition boundary with uncached forward, one-shot serial
prefill with explicit steps, nonzero-position appends, exact capacity,
whole-chunk admission, reset and partial failures. The opt-in Qwen example
preserves every prior independent numerical check and adds exact prefill/append
comparisons for all three prompts.

`cargo run --release --example prefill` recorded tiny timings: full-prefix
97 us, serial prefill 108.541 us, serial chunk calls 104.708 us and explicit
steps 102.208 us. Qwen Hello! (two tokens) recorded 0.836729 s with uncached
inspection, 0.815178 s serial prefill, 0.812071 s chunk calls. These are single
samples; inspection overhead differs, so these are not fair performance
comparisons or an optimization claim. The T11 harness will define controlled
measurement. All final logits are exactly equal.

Exercise: a chunk starts at absolute position 5. For its third query row,
identify the greatest cache position it may read and explain where that bound
comes from. Leave the answer for owner review.
