# 00 — A scalar linear operation

Run the example from the package root:

```sh
cargo run --example linear
```

It prints `Y = [[-2.0, 4.0], [-2.0, 13.0]]`.

## Storage and the weight convention

The function in `src/lib.rs` takes immutable input/weight slices, a mutable
caller-owned output slice, and three dimensions. It allocates no buffers and
adds no bias. Every element and the accumulator use `f32`.

Row-major storage puts each complete row before the next row:

```text
X [tokens, input_dim]  = [[1, 2, 3], [4, 5, 6]]
flat X                = [ 1, 2, 3,   4, 5, 6 ]
W [output_dim, input_dim] = [[1, 0, -1], [2, 1, 0]]
flat W                   = [ 1, 0, -1,   2, 1, 0 ]
Y [tokens, output_dim] = [[-2, 4], [-2, 13]]
flat Y                = [ -2, 4,   -2, 13 ]
```

For zero-based indices `t`, `o`, and `i`:

| Element | Flat offset |
|---|---|
| `X[t, i]` | `t * input_dim + i` |
| `W[o, i]` | `o * input_dim + i` |
| `Y[t, o]` | `t * output_dim + o` |

Each weight row describes one output feature. The equation is
`Y[t, o] = sum_i X[t, i] * W[o, i]`, or `Y = X * W^T` in matrix notation.
The transpose describes the mathematical multiplication; the function reads
the stored rows directly without creating a transposed copy.

## Execution order

1. Reject any zero dimension.
2. Use `checked_mul` for all three required element counts. Return an error
   if any product cannot fit in `usize`.
3. Require each slice length to equal its count, including the output length.
   All error returns occur before any output write.
4. For each token `t`, visit each output feature `o` and start a fresh zero sum.
5. In the innermost loop, advance `i` from zero through `input_dim - 1`.
   Read consecutive elements of the input row and the weight row, multiply
   them, and add to the FP32 sum.
6. Store the completed sum at `y[t * output_dim + o]`.

The checked products and exact lengths also bound every loop offset. For
example, the largest input offset is `tokens * input_dim - 1`, which is inside
the validated input slice. Rust retains safe bounds checks.

The arithmetic uses ordinary floating-point semantics, including rounding,
NaNs, infinities, and arithmetic overflow. There is no finiteness rejection.
Zero weights produce zero for the finite test inputs; `0 * infinity` can
produce NaN. Integer dimension overflow is a separate validation error.

## Trace: Y[0, 1]

Here `input_dim = 3`, `output_dim = 2`, `t = 0`, and `o = 1`.

| `i` | Input offset/value | Weight offset/value | Product | Running sum |
|---|---|---|---|---|
| 0 | `0`: 1 | `3`: 2 | 2 | 2 |
| 1 | `1`: 2 | `4`: 1 | 2 | 4 |
| 2 | `2`: 3 | `5`: 0 | 0 | 4 |

Write `4` to output offset `0 * 2 + 1 = 1`.

The tests in `tests/linear.rs` use independent hand-calculated expected arrays.
Their small integer results are exactly representable in FP32, so equality
needs no tolerance. Invalid-length tests cover short and long slices for
each buffer; zero and overflow tests check every dimension/product and assert
that sentinel output values remain unchanged.

## One exercise

For `tokens = 2`, `input_dim = 4`, and `output_dim = 3`, list the four flat
input offsets and four flat weight offsets read while computing `Y[1, 2]`.
Then give the output offset written. Explain which accesses are contiguous.
Do this on paper; no new code is needed.
