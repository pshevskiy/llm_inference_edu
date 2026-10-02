# 02 — Rotary positions

RoPE encodes position by rotating pairs of features in every query/key head.
There is no embedding row to add. Each pair has a frequency, and its angle is
absolute position times that frequency. A rotation preserves a vector's norm
apart from floating-point rounding.

`src/rope.rs` takes explicit `[tokens,heads,head_dim]` row-major buffers, a
starting absolute position, and a positive base. The offset is
`(token * heads + head) * head_dim + feature`. This implementation uses
**split-half pairing**: feature `j` pairs with `j + head_dim/2`. It requires
even head dimensions and never silently substitutes adjacent pairing.

For `head_dim=4`, the pairs are `(0,2)` and `(1,3)`. With base 10000 their
frequencies are 1 and 0.01. At position 1 the first pair `[1,0]` becomes
approximately `[0.5403023,0.8414710]`. Position zero is the identity.

Execution order: validate dimensions/products, checked position addition,
lengths, finite input and angles; then visit token, frequency pair, and head.
Compute a sine/cosine pair once for every position/frequency and share it
across heads. No output is written until validation has completed. Positions
above 2^24 are rejected because arbitrary consecutive integers beyond that
boundary are not exact in FP32; this is a numerical bound, not a supported
model-context claim.

The loops avoid allocations and share trigonometry across heads. Accesses
between heads have a head-width stride; whether this loop order is faster
than alternatives must be measured at the benchmark milestone. The scalar
reference is retained. Input validation and repeated frequency evaluation
remain visible costs.

`reference/rope.py` computes the rotation with independent NumPy FP64 array
equations, from the same FP32-rounded explicit input fixture. Generate with
`python3 reference/rope.py --write`; verify with `python3 reference/rope.py`.
Rust tests consume the saved fixture offline, check the existing tolerance,
per-head squared norms, exact identity, and equivalence of absolute offsets.

Exercise: for `head_dim=6`, identify all split-half pairs. With base 10000,
which pair rotates fastest, and why? Leave the answer on paper.
