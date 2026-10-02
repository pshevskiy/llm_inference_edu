# 06 — From Qwen text to our Rust logits

The checkpoint is connected to the existing decoder. `Decoder::from_qwen`
consumes its named buffers and removes each owned vector from the map into a
concrete layer. It validates the final decoder and retains one embedding/output
matrix. This moves allocations; it does not duplicate the 1.98 GB FP32 weights.
Initial admission is at most 32 tokens, independently of the advertised
32768 model positions or tokenizer's 131072 metadata. Only short numerical
inputs are qualified here; larger-context qualification comes later.

Execution starts with byte-level BPE using Rust tokenizers 0.20.3, then embedding,
RMSNorm, Q/K/V projections plus their own bias vectors, split-half RoPE, causal
GQA, output projection/residual, RMSNorm/SwiGLU/residual, repeated for 24 layers,
final RMSNorm and the tied vocabulary projection. A bias feature is added to
every token row after the matrix multiplication and before rotation. Q heads
0–6 share KV head 0; heads 7–13 share KV head 1. K/V storage stays compact.
Normal forward returns one vocabulary row; inspection retains bounded traces.
Scratch allocation and repeated weight validation remain intentional baseline
costs; T06 demonstrates correctness, not speed.

The selected plain-message template emits
`<|im_start|>role\ncontent<|im_end|>\n` per message. Without an initial system
message it inserts the pinned Qwen default. An optional generation prompt adds
`<|im_start|>assistant\n` once. We call encode with special insertion disabled:
the template already owns these boundaries. Whitespace, Unicode and literal
special tokens retain the reference behavior. Tool calls/messages are outside
this branch. Hash-verified tokenizer and generation assets supply vocabulary
and stop IDs 151645/151643; no silent truncation or padding occurs.

A concrete teacher-forced trace: plain `Hello!` encodes to `[9707,0]`.
The first token's causal attention can see only itself. Its value, including
V bias, is shared within each KV group. After all layers, the reference and
Rust argmax IDs are 271 at the first position and 358 at the second. These
are identical-input numerical checks, not a generated-answer quality claim.

The independent [Transformers 4.46.3 implementation](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py)
runs PyTorch 2.6.0 CPU FP32 eager attention with one thread and no cache. It
consumes the same hash-verified checkpoint, never calls Rust, and captures first
and last block internals, all 24 block outputs, final normalization and every
logit for two/three-token inputs. Python and Rust loads run sequentially.
The isolated complete dependency freeze is `reference/real-lock.txt`.

The initial full-model `atol=rtol=1e-4` comparison failed. Compensated projection
sums worsened agreement, and a direct RMSNorm experiment also worsened it and
had an underflow edge case found by Astra. Both were removed; the original
scalar dot and scaled RMSNorm remain. Local negative evidence is preserved.
These observations do not isolate a single source of accumulated rounding.

We then calibrated against the reference itself: evaluating an identical first
token alone versus inside a full prefix changes reduction shapes. Reference
logits differed by up to 6.55651093e-4, final normalization by 3.66401672e-3,
and layer 21 output by 5.49316406e-3. Thus the initial threshold rejects its
own independent reference. Separate full-model rules were frozen before held-out
Rust evaluation: logits `atol=1e-3`, intermediates `atol=5e-3`, both
`rtol=1e-4`, with finite values and exact teacher-forced greedy choices.
The scale term matters for large intermediate values. Reference self-comparisons
consume at most 0.595481 of this budget. Tiny-fixture gates remain unchanged;
this calibrated rule does not qualify quantization, Core ML, or longer inputs.

All 49 arrays per prompt passed for `Hello!`, `Rust is` and the held-out Rust
prompt `Two plus two`: 147 array checks, 1215488 logits, eight matching greedy
IDs. Largest absolute intermediate error was 2.31933594e-3 in layer 21,
using 0.246822 of the allowed budget. Logit maxima were respectively
2.00748444e-4, 2.61306763e-4 and 2.30312347e-4. Large relative errors near
zero are disclosed in the logs rather than disguised by a relative-only test.

Run `reference/qwen.py` then `examples/qwen_parity.rs` as documented in
`reference/README.md`. Ordinary tests use tiny tensors and checked-in template
expectations; they require no downloaded model or Python runtime.

Exercise: for a user message followed by an assistant message, write the two
message boundaries and the optional generation header. Identify where an
extra automatic BOS insertion would change the token sequence. Do not solve
this by changing the template.
