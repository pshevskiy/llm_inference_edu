# 25 — A bounded calculator loop and exact continuation

Run `cargo run --offline --example agent`. The protocol decisions are explicitly
mock fixtures. Actual token segments use our tiny Rust model; its emitted `41`
does not encode the calculator JSON. A mocked decision is not model tool skill.

`AgentSession` starts one greedy generation segment. At its output boundary,
`prepare_reply` parses one JSON object, rejects unknown tools/fields, checks the
per-call calculator permission and step budget, then computes finite bounded
arithmetic. Nothing evaluates shell commands or instruction-like strings.
`append_result` accepts the caller's encoding of the trusted numeric envelope,
then admits the whole suffix and next output budget before evaluation. Callers
must encode that exact envelope with their explicitly selected framing; the tiny
example uses fixture IDs `[2,3]` and claims no real tokenization.

The loop permits at most eight calculator calls, 4 KiB per model reply, 256 bytes
per numeric result, and at most 2,048 total token positions (also subject to the
model's smaller capacity). A separate 64 KiB serialization limit bounds exported
transcripts. These are admission limits, not a measured process-memory ceiling.
The next generation segment stops at its configured output limit; syntactically
incomplete JSON is rejected, never repaired by arbitrary execution.

## Execution trace

1. Prompt `[1,5,7]` evaluates three positions.
2. The first segment emits `41`: history `[1,5,7,41]`, processed prefix length 3.
3. The mock requests calculator addition `2+3`. With permission, the result is
   `{"calculator_result":5.0}`.
4. Admitting fixture result `[2,3]` retains the existing three cached positions.
   Evaluation consumes pending `41` at position 3, then `2` at 4 and `3` at 5.
   History and processed lengths are now both 6.
5. The next segment emits `14`: history length 7, processed length 6. Independent
   NumPy equations provide the entire 64-logit expected vector, not just this ID.

The cache remains contiguous and one existing workspace is reused. Continuation
performs only the missing token steps; it neither recopies weights nor recomputes
the valid prefix. No speed improvement is claimed without a benchmark.

Pre-admission failure/cancellation preserves the previous segment and prepared
result. Post-admission cancellation retains the entire suffix and exact evaluated
prefix for resume. Numerical failure after KV writes invalidates the cache;
resumption is refused until reset/replay. Editing earlier history cannot use the
old KV values: `GenerationSession::replay_history` validates first, resets cache
and sampler, admits the complete replacement, and evaluates from position zero.
An agent transcript edit creates a fresh `AgentSession` instead.

The caller owns transcript JSON and may persist it separately from volatile KV.
It includes admitted IDs, emitted IDs, typed decisions and results. There is no
automatic file write or prompt logging. A deserialized transcript is data, not an
execution permission or proof of valid cache, and no blind restore API is exposed.

Real Qwen tool use is **not yet qualified**. The explicit plain protocol probe
uses 71 prompt IDs plus a 64-token output reserve (135 positions); current native
admission evidence permits only eight. No unsupported tools template is inferred
from ordinary chat tokenization. Run the admission-only probe with the local model:

```sh
reference/.venv/bin/python reference/agent_admission.py \
  models/installed-qwen/7ae557604adf67be50417f59c2c2f167def9a775
```

T28 must earn a larger CPU context before that model→tool→model experiment. Passing
numerical gates will still not prove reliable tool selection or argument quality.

Exercise: after step 5, suppose the calculator returns two more IDs `[4,6]`.
Write the exact history, evaluated-prefix length, and next three token/position
pairs that continuation must evaluate. Then explain what changes if position 1
is edited instead. Do not run code to obtain the answer.
