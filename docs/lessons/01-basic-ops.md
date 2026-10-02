# 01 — From rows of numbers to normalized probabilities

T01 adds five mechanisms in `src/ops.rs`. They are ordinary functions over
slices, with caller-owned output buffers and no allocation inside the kernels.
They do not yet form a decoder. Run the small composition example:

```sh
cargo run --example basic_ops
```

## What each operation does

| Operation | Equation or action | Purpose |
|---|---|---|
| Embedding | `y[t,i] = table[token_ids[t],i]` | Replace each integer ID with a learned feature row. |
| Residual addition | `y[i] = x[i] + residual[i]` | Combine a branch's output with the incoming features. |
| RMSNorm | `y[t,i] = x[t,i] / sqrt(mean_j(x[t,j]^2) + epsilon) * weight[i]` | Control each row's magnitude, then scale its features. |
| SiLU | `y[i] = x[i] * sigmoid(x[i])` | Apply a smooth nonlinear gate to each value. |
| Softmax | `p[i] = exp(logit[i] - max) / sum_j(exp(logit[j] - max))` | Produce nonnegative probabilities summing approximately to one. |

Embedding, RMSNorm, and softmax use row-major matrices. For shape
`[rows,width]`, the flat offset of `[t,i]` is `t * width + i`.
An embedding table has shape `[vocab_size,width]`: token ID `k` selects the
contiguous interval `k * width .. (k + 1) * width`. Repeated IDs copy the same
row; token order is preserved. All IDs are validated before copying.

Residual addition and SiLU operate on flat slices with matching lengths. The
caller keeps the logical shape; neither function broadcasts or changes it.
RMSNorm's `weight[width]` is one scale per feature and is shared across rows.
Each softmax row is an independent distribution.

## Execution order and numerical details

Every function validates its dimensions and exact buffer lengths before
writing. Matrix element counts use checked multiplication. Errors carry
details such as the offending buffer, token position, or masked row. The
tests deliberately put invalid tokens/logits/masks in later positions to
confirm that validation finishes before earlier outputs can be modified.

RMSNorm validates positive finite epsilon and finite input/weights. It first
finds `scale = max(max_i(abs(x[i])), sqrt(epsilon))` for each row. It then sums
`(x[i] / scale)^2` and computes:

```text
denominator = sqrt(mean_i((x[i]/scale)^2) + (sqrt(epsilon)/scale)^2)
y[i] = (x[i]/scale) / denominator * weight[i]
```

Dividing numerator and denominator by the same positive scale preserves the
RMSNorm equation while keeping squares bounded. A zero row is safe because
positive epsilon makes the denominator nonzero. There is no mean subtraction
and no bias. FP32 rounding remains; a large final feature scale can still
overflow the final multiplication.

SiLU validates finite inputs. For nonnegative `x`, it computes
`x / (1 + exp(-x))`. For negative `x`, it computes
`x * exp(x) / (1 + exp(x))`. Both exponential arguments are nonpositive, so
large finite inputs cannot overflow the exponential.

Softmax validates each allowed logit and each row before writing. It finds
the maximum, writes the shifted exponentials, sums them, then divides those
same output values by the sum. The maximum's exponential is exactly one,
so the denominator is positive. Subtracting a common maximum leaves the
mathematical probabilities unchanged and avoids `exp(1000)` overflowing.
Very unlikely entries can underflow to zero, which is normal FP32 behavior.

`masked_softmax` takes a boolean slice with the same shape. **True means
allowed.** Masked entries do not contribute to the maximum or denominator,
and their output is exactly zero. Their logit values are ignored, so a masked
`-infinity` is allowed. A row with every mask entry false returns `AllMasked`
instead of trying to normalize an empty distribution. Active NaN or infinity
is an error.

Embedding copies values unchanged. Residual addition follows ordinary FP32
semantics, including overflow or non-finite values; it does not validate
finiteness. Finiteness checks in normalization/activation functions describe
their input contracts, rather than a complete model-wide validation policy.

## A concrete softmax trace

Take the first softmax fixture row: logits `[1000, 1001, 1002]`.

1. The row maximum is `1002`.
2. Subtract it: `[-2, -1, 0]`.
3. Exponentiate: approximately `[0.13533528, 0.36787944, 1]`.
4. The sum is approximately `1.50321472`.
5. Divide: approximately `[0.09003057, 0.24472847, 0.66524096]`.

These probabilities sum approximately to one. Using large original scores
does not require a large exponential after subtracting the maximum. The
second fixture row has three identical negative scores and gives a uniform
distribution.

The example combines all five functions using separate small buffers. Token
ID `1` selects `[3,4]`; adding residual `[1,-2]` gives `[4,2]`. With unit
weights and epsilon `1e-5`, RMSNorm divides by approximately `sqrt(10.00001)`.
SiLU gates the normalized values, then softmax produces a two-entry
distribution. This is a teaching composition, not a language model.

## Independent evidence

`fixtures/t01-inputs.json` contains the shared inputs. The NumPy equations in
`reference/basic_ops.py` generate `fixtures/t01-expected.json` independently of
Rust. They use FP64 arithmetic on FP32-rounded inputs; Rust uses FP32 loops.
RMSNorm's oracle uses the direct sum-of-squares equation, providing an
independent check on the scaled implementation.

Ordinary tests consume the saved fixtures without launching Python. They
check finiteness and the existing `atol=1e-5`, `rtol=1e-4` error rule, and report
maximum absolute and scale-aware relative errors. Embedding and the fixture's
exact residual sums also require exact equality. Other tests cover invalid
IDs/shapes, unchanged output on errors, extreme finite scores and SiLU input,
zero/large RMSNorm rows, invalid epsilon, and all-masked rows. See
`reference/README.md` for optional regeneration commands and the error metrics.

## One exercise

For logits `[8,9,10]` and mask `[true,false,true]`, write the maximum used,
the shifted allowed logits, their exponentials, and the normalized output.
Explain why the middle probability is zero. Work on paper; no code change
is needed.
