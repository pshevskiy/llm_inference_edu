# 04 — Two complete decoder blocks

`src/model.rs` joins the primitives into the synthetic architecture specified
by the plan: vocabulary 64, two layers, hidden width 32, four query heads,
two KV heads, head width 8, feed-forward width 64, and context capacity 32.
Weights are fixed nontrivial dyadic numbers in `fixtures/tiny-model.json`.
`reference/make_tiny_fixture.py` documents their formula; both languages read
the saved values, rather than hoping that random seeds produce equal weights.

Execution order for every block:

1. RMSNorm the incoming hidden rows.
2. Project Q/K/V, rotate Q/K, perform causal GQA, and project its output.
3. Add that attention output to the incoming hidden rows.
4. RMSNorm the result and compute gate/up projections.
5. Multiply `SiLU(gate)` by `up` elementwise: this is SwiGLU.
6. Project down to hidden width and add the second residual.

After two blocks, final RMSNorm and the shared embedding/output table produce
logits. Softmax is unnecessary for greedy argmax. Output weights are tied:
the embedding table is read as `[vocab,hidden]`, with no second copy.
There are no biases in this synthetic decoder; that is not a claim of exact
Qwen checkpoint compatibility.

`Decoder::new` checks configurations, exact shapes, and finite weights before
making them immutable inside the decoder. Forward errors return no valid state.
All storage is row-major. Normal `forward(ids,false)` returns only the final
input position's logits and drops each block's intermediates. Inspection mode
`forward(ids,true)` retains every named activation and teacher-forced logits
at all input positions; it is limited to 32 tokens. It is for understanding,
not performance measurement. Scratch allocation remains a later measured task.

For token zero, embedding picks row `token_id`. Its normalized row produces
Q/K/V. Each query head has just its group's position-zero V to attend, so its
context equals that value row. The output projection mixes those heads;
residual addition retains the embedding information before the feed-forward
branch. Token one can then attend to positions zero and one, but not token two.

The NumPy oracle saves 35 named arrays: embedding, 16 arrays per block, final
normalization, and all teacher-forced logits. Tests compare every array under
the existing tolerance, compare normal versus inspection mode, and confirm
prefix and future-token invariants. Regenerate with
`python3 reference/decoder.py --write`; verify without `--write`.

Run `cargo run --example tiny_decoder` for four greedy token-ID steps using
full-prefix recomputation. These synthetic weights are not expected to produce
language. There is no tokenizer, pretrained model or cache yet.

Exercise: draw the two residual paths for one block and label the three
different widths encountered: hidden, KV projection, and feed-forward.
Explain which branch expands then contracts the feature width.
