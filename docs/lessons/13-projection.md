# 13 — Break the accumulation dependency chain

`linear_lanes4` validates nonzero dimensions, all three checked products and exact
buffer lengths before the first write. For each input row and weight row,
`dot_lanes4` visits full groups of four consecutive values. Four independent
FP32 sums accumulate their corresponding products. It then reduces lanes left
to right and adds any final one-to-three tail products in increasing-index order.
No scratch allocation or model-weight copy occurs. Only tied vocabulary
projection uses this candidate in the decoder, including inspection and cached
one-token paths. Other projections and original scalar `linear` remain unchanged.

The weight row is contiguous and the input row is small enough to be repeatedly
read without a transpose. The original accumulator prevents independent additions
from progressing until its previous result is ready. Splitting the reduction
exposes independent work to the compiler. On this M1 Pro, generated vector
instructions accumulate two pairs of lanes. We examined that output and measured
actual Qwen rectangular dimensions; portable Rust alone does not guarantee any
particular SIMD instruction or speedup. See `docs/experiments/13-lanes.md`.

Reduction order changes floating-point rounding. Finite independent FP64 expected
values and the fixed operator gate qualify the kernel; frozen tiny and real-model
activation/logit gates qualify adoption. The full-model reference and exact
teacher-forced IDs remain the acceptance checks. NaN, infinity and overflow still
follow the explicit ordinary arithmetic order; compare to the scalar path when
debugging differences. No tolerance was widened for this change.

**Trace:** For X=`[1,2,3,4,5,6,7,8]`, W=`[1,-1,2,0,-2,1,0,1]`, the two full
groups produce lane totals `[-9,4,6,8]`. Reduction yields `((-9+4)+6)+8=9`.
The independent full dot is `1-2+6+0-10+6+0+8=9`. There is no tail here.

**Exercise:** For input width seven, identify which product indices enter each
lane and which enter the scalar tail. Draw the addition order for the candidate
and the original scalar function. Explain why equal real-number equations can
produce slightly different FP32 values. Do not change tolerances to hide them.
