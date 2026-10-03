# 26 — Learn LLM inference by tracing memory

This small console application is for the owner's education in LLM inference.
Its first exercise type is an element offset in a row-major X, W or Y matrix.
It reuses the existing engine for a separate model probe; it implements no new
forward pass. The interface stays small enough to read and maintain.

## Use it

```sh
cargo run --offline --example learn_inference -- list
cargo run --offline --example learn_inference -- show w-offset
```

Then submit your own decimal answer with `answer w-offset YOUR_DECIMAL`.
To retain counters, append `--progress docs/evidence/my-learning-progress.json`.
Without that option, the application writes no progress file. The recommended
directory is ignored by Git. Progress stores problem IDs/counters only, never
your answer, prompts, model text or KV cache.

An incorrect answer gets a hint and leaves the solution locked. A correct answer
unlocks a curated explanation. The model never determines correctness. A completed
exercise means at least one correct submission, not verified understanding.

## Execution order and trace

`EXERCISES` holds the three curated nonsquare matrix problems. `exercise(id)`
selects one. `Progress::answer` validates the entire loaded record, validates the
decimal answer, computes checked `row*width+column`, and checks the global attempt
budget before changing counters. Invalid text, unknown IDs, invalid loaded state
or exhausted budget leaves the existing record unchanged.

For the public synthetic X case `[3,4]` at `[2,1]`, eight elements precede row2;
one more element selects its second column, giving offset9. The smoke runner's
answer8 increments attempts to1 and returns only a hint. Its answer9 reaches
attempts2/correct1/completedtrue and unlocks the explanation. This is a test
trace, not the owner's attempt or a claim that the owner understands it.

`storage::load` admits at most4,096bytes of strict versioned JSON. Known unique
IDs, positive attempts, bounded totals, correct≤attempts, and
`completed == (correct>0)` must all agree. `storage::save` serializes validated
state, writes a fresh sibling file, syncs it and renames it over the explicit
destination. Failed writes before rename preserve the previous file. Atomic
replacement does not promise durable parent-directory metadata across a crash.
On Unix, the new file is readable/writable only by its owner.

## A numerically correct model can teach incorrectly

`examples/education_probe.rs` is a separate opt-in experiment. It loads the pinned
Qwen through existing Rust weights/tokenizer/GenerationSession APIs. Each fixed
public question admits the whole80-token prompt plus96-output reserve into
capacity512. A pinned independent CPU FP32 Transformers reference exports every
sampled vocabulary row; Rust checks all finite values and exact greedy IDs using
the existing logit gate. EOS remains in history but may be unevaluated.

The probe asks for strict JSON numeric facts and one short explanation. Facts
must match the curated problem, including the canonical formula. Unknown or
duplicate fields, wrong facts, excessive size, incomplete/EOS-less output and
wrong types fail. Even correct fields would not establish prose truth; the test
suite deliberately demonstrates that distinction. Model output cannot alter
grading or progress.

The independent reference failed all three factual/schema probes: it reversed
dimensions/indices and added Markdown while omitting the required explanation.
Normal generated teaching stays disabled. Curated explanations remain usable.
Numerical parity and language quality are separate recorded results; see the
current experiment26 record for the completed Rust comparison.

Reproduce the opt-in comparison with fresh directories:

```sh
reference/.venv/bin/python reference/education.py --model MODEL \
  --output docs/evidence/my-learning-reference
cargo build --offline --release --example education_probe
target/release/examples/education_probe MODEL \
  docs/evidence/my-learning-reference/manifest.json docs/evidence/my-learning-rust
```

`MODEL` is the already installed pinned Qwen directory; no network access or model
download occurs. Missing tools are not replaced by another model/backend.

Exercise: in the linear convention `W[output_dim,input_dim]` and
`Y[tokens,output_dim]`, explain which dimension defines the row stride of each
buffer and why the two strides can differ. Use a nonsquare example you choose.
