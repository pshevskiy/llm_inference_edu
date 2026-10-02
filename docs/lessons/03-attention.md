# 03 — Causal grouped-query attention

`src/attention.rs` composes three projections, rotary positions, causal scores,
stable softmax, value weighting, and an output projection. Its result contains
named intermediate buffers so the owner can inspect every stage. It allocates
scratch deliberately; this is the reference path, not a benchmarked fast path.
The projection matrices are `[output,input]` row-major, currently without bias.

Q has `[tokens,query_heads,head_dim]`; K/V have
`[tokens,kv_heads,head_dim]`. Query head `h` selects
`h / (query_heads/kv_heads)`. With 4 query heads and 2 KV heads, the mapping is
`[0,0,1,1]`. No expanded KV copies are needed in Rust.

After Q/K rotary positions, token `t` computes dot products only against keys
`0..=t`. Divide by `sqrt(head_dim)`, normalize that prefix with softmax, and
form a weighted sum of the matching V rows. The head results occupy contiguous
pieces of the context row. The final linear projection mixes their features.

At token zero every head has one allowed key, so its probability is exactly 1
and its context is precisely its KV group's first value row. At later tokens,
future inputs cannot affect earlier contexts: no future key is scored.

Mechanical sympathy: dot products read consecutive head features; scores and
probabilities reuse one row across every token/head. There is no full quadratic
score allocation or replicated K/V. Returned traces retain projection and
rotation buffers for learning. Repeated validation, scalar reductions and owned
scratch remain costs to measure and improve later.

`reference/attention.py` uses NumPy matrix multiplication and einsum with an
explicit triangular mask, independently of Rust's loops. Its small fixture
checks Q/K/V, both rotated projections, context and output. Other tests modify
a future token, compare prefix calls, and use distinct KV rows to expose the
mapping for all four query heads. Tolerance remains `atol=1e-5, rtol=1e-4`.

Exercise: with 6 query heads and 2 KV heads, write the KV head selected by
each query head. At token index 2, which key positions may each head attend?
Do not implement another kernel to answer this.
