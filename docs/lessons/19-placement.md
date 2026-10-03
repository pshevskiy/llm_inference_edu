# 19 — Separate execution policy from observed work

T19 compares the same short public workload on our Rust CPU implementation,
Core ML CPU_ONLY and CPU_AND_NE. GPU inference is excluded. The experiment is
complete with a negative ANE qualification: observed CPU graph execution and no
useful policy improvement, while framework ANE tracing remains inconclusive.
Our Rust CPU engine stays selected. This is not an ANE-accelerated engine.

`reference/coreml_benchmark.py` executes these stages in order:

1. Verify the T18 artifact. Reduce only public output lists from49arrays to
   logits; restoring those lists must recover the identical protobuf, and weight
   bytes must match. Compile the new interface without changing precision/math.
2. Verify the pinned independent reference versions and assets. Build HF CPU
   FP32 full-prefix logits for the original cases and actual benchmark histories.
   Generate greedy expectations independently of our engine and native artifact.
3. Run Rust qualification and its existing release benchmark in separate
   processes. Run each native policy separately with its own native state.
4. Before timing, qualify every history under the unchanged reference gates.
   Reset outside request timing; evaluate prompt IDs, sample, prepare UTF-8 chunks,
   and evaluate each previously emitted token before sampling the next.
5. Preserve the last emitted token as pending. Continuation first evaluates it,
   then appended IDs, then samples another pending token. Check counters and IDs
   for every warmup and repetition. Save raw observations and robust summaries.

The model/session always owns its cache. The reference native wrapper never
mirrors it into Rust. Native state is approximate paired FP16; Rust state is
FP32. Both cache payloads are786,432bytes at capacity32; paired half saves no KV
memory. Rust reuses explicit workspace/caller logits. A logits-only native
interface avoids unnecessary diagnostic output marshalling; internal layout and
workers remain Core ML's responsibility. This does not explain the whole timing
advantage or demonstrate that every operation follows mechanical sympathy.

The comparison uses equivalent development greedy/text contracts, not identical
code: Rust sampler/tokenizers0.21.4 versus NumPy argmax/Python tokenizers0.20.3.
Lowest ID wins ties, nonfinite values fail, EOS is preserved in history, unfinished
UTF-8 stays pending, special tokens are hidden. Portable tests check these rules.
Python bridge/policy overhead is included in native timings. Initial tokenization,
loading, cache reset and UI delivery are excluded from the measured request.

**Concrete trace:** prompt `[9707,0]` creates2cached positions. Sampling358does
not yet cache it. Evaluate358, sample2776; evaluate2776, sample264; evaluate264,
sample48948. Now processed=5, history length=6 and48948is pending. Continuation
`[264,1172]` evaluates `[48948,264,1172]`, advancing processed to8. Sample614:
history length9, processed8, one pending token. All three paths match this
independent trace. Sampling another token never automatically adds its KV.

With3warmups and10repetitions, median warm greedy TTFT was651.074/73.371/73.895ms
and decode3.067/27.521/27.589tokens/s for Rust/CPU_ONLY/CPU_AND_NE. The native
CPU path is faster on this specific short workload; allowing NE adds no useful
measured improvement. Fixed process order, differing precision/thread/policy
implementations and uncontrolled filesystem caches are disclosed. Ten samples
are not robust p95 or product-wide speed evidence. Full scopes, memory accounting
and min/max/MAD are in [experiment19](../experiments/19-policy-audit.md).

`reference/coreml_trace.swift` is a public tracing control, not a production
native boundary. Its signposts bracket synchronous model load/predictions.
`reference/analyze_coreml_trace.py` resolves the exported XML's references,
filters our PID and deduplicates alternate signpost views. In the matched Qwen
control,325sampled BNNS CPU stacks fall within eight marked prediction intervals.
This is actual CPU involvement. Sample counts are not operator timing or wall-time
percentages, and do not prove exclusive CPU placement.

Core ML/ANE framework tables remain empty for both Python and native Swift;
recordings report a Time Mapping issue. Our custom intervals work. The exact
framework issue remains unresolved; empty ANE tables do not prove zero activity.
Anticipated plans favor CPU, but supported/preferred devices and estimated costs
are metadata. To qualify an accelerator, obtain healthy device attribution for a
numerically qualified artifact and demonstrate useful end-to-end improvement.

Reproduce with fresh output paths from the repository root. This is an opt-in
hardware benchmark (the owner authorized this run), bounded180s/path and240s for
preparation, sequential with one model process at a time:

```sh
cargo build --offline --release --example policy_trace --example benchmark
reference/.venv/bin/python reference/coreml_benchmark.py run \
  docs/evidence/t18-qwen docs/evidence/NEW-t19
mkdir -p docs/evidence/NEW-control
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swiftc \
  -O -g reference/coreml_trace.swift -o docs/evidence/NEW-control/coreml_trace
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace record \
  --template 'Time Profiler' --instrument 'Core ML' --instrument 'Neural Engine' \
  --instrument 'Points of Interest' --output "$PWD/docs/evidence/NEW-control/qwen.trace" \
  --time-limit 60s --target-stdout "$PWD/docs/evidence/NEW-control/stdout.txt" \
  --launch -- "$PWD/docs/evidence/NEW-control/coreml_trace" qwen \
  "$PWD/docs/evidence/NEW-t19/artifact/qwen.mlmodelc"
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace export \
  --input docs/evidence/NEW-control/qwen.trace --toc \
  --output docs/evidence/NEW-control/toc.xml
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace export \
  --input docs/evidence/NEW-control/qwen.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile" or @schema="os-signpost" or @schema="coreml-os-signpost" or @schema="ane-hw-intervals"]' \
  --output docs/evidence/NEW-control/events.xml
reference/.venv/bin/python reference/analyze_coreml_trace.py \
  docs/evidence/NEW-control/toc.xml docs/evidence/NEW-control/events.xml \
  docs/evidence/NEW-control/cpu-analysis.json
reference/.venv/bin/python -m unittest discover -s reference -p 'test_*.py'
```

Per-command Xcode selection leaves the global developer setting unchanged.
Instruments needs its ordinary user cache outside a restrictive sandbox. Trace
bundles and incidental process/device metadata stay in ignored evidence.

Public sources: [compute-plan API](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.compute_plan.MLComputePlan),
[Core ML performance reports](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode),
[public signposts](https://developer.apple.com/documentation/os/OSSignposter),
and [Apple's signpost diagnostics](https://developer.apple.com/forums/thread/713781).

**Exercise:** Reduce the output budget from4to2, leaving the2-token prompt and
2-token continuation unchanged. Calculate processed/history counts immediately
before continuation and immediately after sampling its next token. Identify the
pending token at each boundary; do not count emitted tokens as already cached.

Next proposed ticket: T20's CPU native boundary, with explicit ownership and a
single execution worker. ANE acceleration remains unqualified; owner review pending.
