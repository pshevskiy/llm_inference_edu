# 11 — Measure scopes before changing performance

`run_sample` admits the whole workload before resetting cache state. Construction,
loading and tokenization stay outside a warm request timer. Serial prefill returns
the last prompt logits. Greedy sampling emits the first token; its timestamp is
TTFT. Teacher forcing picks a prescribed token and has no sampling-based TTFT.
Both retain first-logits-ready separately. Later emissions first evaluate the
previous emitted token at its absolute position. Decode measures N-1 evaluations
between the first and last emission. Text decoding has its own scope.

For a 16-token prompt and four emitted tokens, cache length progresses 16,17,18,19.
History length is 20 after the fourth emission: one token is pending. Appending
two continuation IDs evaluates that pending token followed by the two IDs, bringing
processed length to 22. Sampling the continuation response creates history length
23 and again leaves one pending token. Admission therefore reserves 23 slots.

`step_profiled` uses the same arithmetic and cache rules as `step`, with clocks
around operator groups. It runs outside benchmark repetitions: clocks and trace
allocation must not influence the baseline being compared. Contiguous immutable
weight rows and compact KV are retained. Timing indicates what to investigate;
it does not establish why a stage is expensive. Measure allocations in T12 before
assuming that scratch reuse will dominate arithmetic or weight reads.

A single Qwen diagnostic on this M1 Pro spent roughly 108 ms of 406 ms in the
vocabulary projection; gate/up and down projections account for much of the rest.
This is a scoped observation, not a speedup or steady thermal result. Keep a
callable scalar reference and use the numerical gates before accepting any kernel
change. See `benchmarks/README.md` for commands and limitations.

**Exercise:** A prompt has 7 tokens, generation emits 3 tokens, and a continuation
adds 4 token IDs then emits one response token. Write the processed and history
lengths immediately before and after continuation. How many slots must admission
reserve? Do not evaluate the final emitted token just to make the counts equal.
