# 07 — The uncached greedy loop

`uncached_greedy` first checks nonempty prompt, positive output limit, valid
prompt/stop IDs and checked `prompt + max_new <= capacity`. Admission counts
emitted tokens even though the final token may never be evaluated. Requests
that cannot fit fail before forwarding; history is never silently truncated.

The function copies the prompt into one history buffer. Each step forwards
that complete history, scans the last vocabulary row, chooses the largest logit
(lowest ID on ties), appends the token, then checks configured stops. A stop
ID is part of history and generated IDs but is not forwarded after stopping.
Otherwise the next iteration recomputes all positions. A forward failure returns
no completion; the immutable model has no cache to invalidate.

Concrete trace: tiny prompt `[1,5,7]` forwards three positions and emits 41.
History is now `[1,5,7,41]`; the next full-prefix pass emits 53. Four steps
produce `[41,53,41,50]`. If 41 is a stop ID, only the first step runs. The
fourth output token is recorded but not evaluated at an output-limit stop.

The CLI explicitly uses system `Be brief.` by default (override the last
argument), configured Qwen stop IDs and greedy policy rather than the source
checkpoint's sampling defaults. It renders/tokenizes and admits the prompt
before loading large weights. It decodes the whole generated token sequence
once at the end, avoiding partial-byte token printing. Generated text is stdout;
load/generation durations and stop reason are stderr. No streaming yet.

```sh
cargo run --release -- models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775 'Say hi.' 4 'Be brief.' uncached
```

T07 recorded the following bounded run on the M1 Pro (T10 adds the explicit
`uncached` selection shown above): prompt 19 tokens, emitted 4,
`Hello! How can`, OutputLimit; preparation/load 5.081 s and generation 25.138 s.
This is a single demonstration, not a benchmark distribution or language quality
qualification. Output is incomplete because of the explicitly requested limit.

Weights remain tied/immutable and per-dot features contiguous. Recomputing the
prefix rereads weight matrices and repeats earlier token work. This is the
correctness baseline; it is intentionally inefficient. T08/T09 will establish
cache reuse; measured allocation/kernel changes follow later.

Exercise: after emitting the second token in the tiny trace, write the history
length and the number of positions processed by that most recent forward.
Explain why they differ. Leave the answer for owner review.
