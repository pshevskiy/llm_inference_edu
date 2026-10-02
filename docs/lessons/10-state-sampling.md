# 10 — Token history, sampling and text are different state

`GenerationSession` owns token history and one model-bound cache. It admits the
entire prompt before evaluation. Initial serial prefill processes that history
in order. `next_token` first evaluates any pending suffix, then delegates finite
last-row logits to `Sampler`, appends the sampled ID and checks stop/output
limits. Sampling does not evaluate the new token. At a completed emission,
history contains exactly one more token than the processed cache prefix.
The final stopped token is retained, including EOS, without an extra model call.

Tiny trace: start `[1,5,7]` leaves history=3/processed=3. Emit 41 leaves 4/3.
The next call evaluates 41 at absolute position 3, then emits 53, leaving 5/4.
Four greedy emissions are `[41,53,41,50]`, ending 7/6. Calling again returns
None and does not accidentally evaluate the stopped token.

Cancellation is cooperative at token boundaries, including prefill. It is
checked before stepping and before sampling. An already running token finishes;
its complete prefix remains valid. The admitted history and RNG stream survive.
Resume evaluates every missing prompt token before sampling, or the pending
emitted token before the next sample. An execution failure preserves IDs but
invalidates the cache and clears usable logits. Reset clears history/cache state
and restarts the configured seed. No threads or async framework are added.

Sampling policy is separate from model execution. Greedy ties choose the lowest
ID. Temperature sampling keeps the top-k logits using partial index selection,
orders them by descending value then ID for reproducibility, and samples mass
`exp((logit-max)/temperature)`. Probability accounting uses FP64 to avoid
extreme temperature/difference overflow; model weights and calculations remain
FP32. ChaCha8 uses a pinned seed and library version; invalid inputs do not
advance its stream. Only top-k filtering is included, not top-p or penalties.

Pinned RNGs are rand 0.9.5/rand_chacha 0.9.0, shared with tokenizer dependencies.
Rust tokenizers moves from 0.20.3 to 0.21.4 for its established
[contextual decode stream](https://github.com/huggingface/tokenizers/blob/v0.21.4/tokenizers/src/tokenizer/mod.rs).
The Python reference stays 0.20.3; every prior token/activation/logit check was
rerun successfully. Ordinary tests construct a tiny byte decoder: token bytes
C3 then A9 produce nothing after C3, then `é` after A9. IDs remain in generation
history even when no text chunk is available. Known special tokens are hidden
without creating false pending-byte warnings.

At an output limit or cancellation, unfinished decoder tokens are reported and
withheld, not force-flushed as corrupted partial Unicode. The upstream decoder
can also defer a literal trailing replacement character; this presentation
limitation never changes model token IDs. Intrinsically malformed byte output
still follows tokenizer replacement behavior. Text is a view, not cache identity.

The CLI now defaults to cached greedy execution and prints complete text chunks.
An explicit `uncached` final argument retains the baseline. Its public inputs,
Qwen stop IDs and 32-token total admission are unchanged. No shell signal handler
or parallel worker is installed; embedding callers control the atomic cancel.

```sh
cargo run --release -- models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775 'Say hi.' 4
cargo run --release --example cached_generation
```

Actual Qwen demonstration produced `Hello! How can`, history=23/evaluated=22,
load 4.556 s and generation 8.961 s. Earlier uncached demonstration produced the
same four tokens in 25.138 s. These are single runs including different work
paths, not a benchmark distribution. T11 will control workload and timing.
Seed-42 tiny temperature 0.7/top-k 8 demo emits `[5,3,34,57]`; no quality claim.
Tests also check an independent two-class probability of 3/4, seed/reset,
non-finite/extreme parameters, exact capacity, cancellation/recovery and partial
prefill failure cleanup. All real parity gates remain unchanged.

Exercise: after a five-token prompt and three emissions, write history length,
processed length and the absolute position evaluated by the next call. Explain
what cancellation before that call preserves. Leave the answer for review.
