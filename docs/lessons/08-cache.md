# 08 — One evaluated token and its KV state

`CachedSession` borrows one immutable decoder for its entire lifetime. A cache
cannot be reused with a different model that happens to have the same dimensions.
Each layer owns contiguous K and V arrays with logical shape
`[kv_head, capacity, head_dim]`. Offset is `(head * capacity + position) * D + i`.
Keys are stored after rotation at the absolute position; values are unrotated,
including projection biases. Query heads read their KV group without duplication.

Construction checks capacity and byte arithmetic, reserves storage fallibly,
and enforces a 128 MiB cache-only planning budget. This is separate from weights,
scratch and physical memory. At 32 Qwen positions logical KV is 786432 bytes.
No cache growth or silent eviction occurs. Per-step temporary buffers still
allocate; this is the correctness slice, not the scratch-reuse milestone.

`step(id)` first checks valid state, token range and spare capacity. It then
embeds one token, runs each layer, rotates its Q/K at `processed`, writes that
layer's K/V row, attends only through this position, completes SwiGLU/residuals,
and projects final logits. Only after every layer and output succeeds does
`processed` increase. Admission errors preserve the valid prefix. An arithmetic
or other execution failure may have written some layers, so it invalidates the
whole prefix and sets processed to zero. Another step requires explicit reset.

Reset changes logical state without reallocating or clearing storage. Stale
entries cannot be read because attention reads only through the newly written
position. Every layer writes position zero before a reset session commits its
first token. This is logical invalidation, not secure memory erasure.

Trace: evaluating token 1 at position zero writes each layer's first K/V row
and returns logits; processed becomes 1. Sampling token 41 does not call step,
so processed stays 1. Calling step(41) rotates at position 1, attends to positions
0 and 1, and commits processed=2. History and evaluated positions are distinct.

Six cache tests cover exact cached/uncached equality over every position,
GQA/biases, exact capacity and overflow, reset/repeated IDs, preserved state on
bad IDs, a forced FFN overflow after KV writes, and admission before a large
allocation. The opt-in Qwen parity example compares all eight teacher-forced
positions exactly with uncached Rust while retaining the independent reference
checks. No new numerical tolerance was needed.

Exercise: with capacity 8, head dimension 4 and KV head 1, locate feature 2
of position 3 in one cache array. Then say which positions a query at position 3
may read. Leave both answers for owner review.
