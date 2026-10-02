# 16 — Decide whether compressed weights earn a model path

T16 rejected the pretrained INT8 candidate: KL, greedy agreement and maximum
logit change failed the frozen quality gate. Default builds expose only FP32
model weights. Reproducing the rejected experiment requires the nondefault
`int8-experiment` development feature; it is not a selected runtime precision.

With that feature, each large matrix is either an FP32 vector or a validated
`GroupwiseInt8` matrix. `WeightMatrix` has just these two cases. JSON/checkpoint
loading still creates FP32 weights; `Decoder::into_int8(32)` explicitly consumes
that decoder and converts one matrix at a time. Norm weights and Q/K/V biases
remain FP32. A live cache borrows its decoder, so Rust prevents replacing those
weights while that cache is in use. FP32 remains the default.

Execution starts by validating matrix dimensions and all embedding IDs before
writes. Embedding lookup reconstructs only the requested rows. Attention and
feed-forward projections reconstruct each weight in a register, multiply it by
an FP32 activation, and accumulate in FP32. The tied vocabulary head uses four
accumulator lanes, matching the existing FP32 head's order. Other projections
keep the scalar accumulation order. The same attention, RoPE, cache, sampling
and generation code runs for both representations. No full matrix is reconstructed
per token. Buffered cached decode remains allocation-free.

Groups never cross rows. With G=32 and input width I, row o's weight i lives at
`values[o*I+i]`; its scale lives at
`scales[o*ceil(I/32)+i/32]`. This preserves sequential byte reads in a dot product.
An aligned group reads one scale for 32 byte weights. Reduced weight traffic is
useful only if it outweighs integer conversion and multiplication overhead. The
fused implementation still reconstructs weights on each invocation. Mechanical
sympathy supplies a hypothesis; model measurements decide whether it helps.

`into_dequantized_reference()` consumes a quantized decoder and reconstructs its
weights once, outside inference. Exact activation/logit comparisons to this
reference detect arithmetic/layout bugs separately from the quality difference
between quantized and original weights. Synthetic tests compare every named
intermediate, cached positions, resets, arbitrary group sizes and width tails.
The pretrained experiment also checks all held-out logit rows exactly against
this reconstructed reference. Original FP32 numerical gates stay unchanged.

**Trace:** For a 2x5 embedding with G=4, scales are indexed as a 2x2 array.
Looking up token1, coordinate4 reads `values[1*5+4]` and
`scales[1*2+4/4]`, i.e. value9 and scale3. If they hold q=-2 and s=.25,
the activation written to coordinate4 is -.5. Only that token's five weights
are reconstructed. During the tied head, these same compressed rows become
output-class weight rows; the layout does not change.

The experiment freezes its rule in `docs/experiments/16-quality-rule.json`
before model measurements. Three public sentences provide a short teacher-forced
smoke test. Mean next-token loss, KL and greedy agreement use positions with a
known next token; maximum logit change covers every input position. This does
not establish broad language quality. Logical payload includes scales and retained
FP32 tensors. Point RSS and whole-process peak are separate physical measurements;
the original BF16 checkpoint file is unchanged, and loading still expands FP32
before conversion. Short timings cannot establish a reproducible speedup or
justify switching the default precision. See the T16 plan record for actual
results and the representation decision.

```sh
cargo test --offline --features int8-experiment --test quantized_model --test workspace
cargo build --offline --release --features int8-experiment --example int8_qualification
/usr/bin/time -l target/release/examples/int8_qualification \
  models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775 \
  docs/evidence/t16-new-run
```

The last command is an opt-in checkpoint/hardware experiment, using a fresh result
directory. It records actual identity and raw observations. Repeated whole-model
performance and sustained thermal/energy qualification remain separate opt-ins.

**Exercise:** For a 3x33 matrix with G=32, find the value and scale offsets for
row2, coordinate32. How many scale bytes does the entire matrix require? Explain
why sharing one tail group across adjacent rows would give the wrong answer.
