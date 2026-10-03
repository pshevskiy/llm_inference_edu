# T26 — Owner-focused memory-layout learning

Three curated row-major offset exercises, deterministic grading and explicit
bounded progress persistence work in the Rust console application. No additional
UI, server, exercise type or numerical engine. Synthetic smoke answers8then9
exercise wrong-hint/correct-explanation/retry behavior; they are not owner work.

Ordinary commands: fmt check, offline Clippy with warnings denied, offline tests
all exit0:117unit/integration tests,0doctests. Five new tests cover nonsquare/final
indices, invalid indices/overflow, decimal parsing, retry/invalid-state
preservation, progress version/unique IDs/counter consistency/budget, strict model
JSON and explicit atomic saves. Deliberately plausible facts with false prose
demonstrate that field validation cannot qualify a teacher.

Independent real-model reference `t26-reference-final` completes/exits0, pinned
Qwen revision `7ae557604adf67be50417f59c2c2f167def9a775`, CPU FP32 eager,
torch2.6.0/Transformers4.46.3/tokenizers0.20.3/NumPy2.3.2/safetensors0.6.2,
one thread. Actual M1 Pro,16GiB,macOS26.6.2/build25G83. Three questions each
have80prompt IDs and96output reserve atcapacity512, outputs62/60/53IDs including
EOS. Independent factual/schema quality is0/3. All emit Markdown and omit the
required explanation field; numeric dimensions/indices/offsets are wrong.

| Problem | Correct row,width,column,offset | Model row,width,column,offset |
|---|---|---|
| X[2,1] in[3,4] | 2,4,1,9 | 2,3,4,1 |
| W[1,2] in[2,4] | 1,4,2,6 | 1,2,4,2 |
| Y[1,1] in[3,2] | 1,2,1,3 | 1,3,2,1 |

No prompt repair, output repair, model/precision switch or tolerance widening.
Generated explanations are rejected and ordinary generated mode remains disabled;
the application uses curated explanations. This is a teaching-quality rejection,
independent of decoder correctness. Three probes would not demonstrate general
reliability even if they had passed.

The initial reference `t26-reference` is preserved. Astra caught a mismatch in
Python/Rust size bounds: original UTF-8 explanation≤512bytes, full response≤4096,
separate nonempty trim. The final reference aligns those bounds and rejects
duplicate fields; it still fails all quality probes. Identity records both Python
script/helper hashes, model pin/versions/actual host before model execution.
Reference peakRSS2,722,889,728bytes; peak process footprint2,236,189,000bytes.
Physical measures are separate from weights, file sizes and KV estimates.

Rust current-source comparison `t26-rust` and process wrapper exit0. All175sampled
full-vocabulary rows agree with original atol0.001/rtol0.0001 and exact greedy IDs;
worst normalized budgets0.911655/0.857191/0.971271. Output text exactly matches
the independent reference and all factual/schema checks reject. Cache/history
lengths141/142,139/140,132/133 preserve pending EOS. Matching bad text establishes
decoder parity, not a teacher. Ordinary model wording stays disabled.

Rust maximum RSS2,473,279,488bytes, peak process footprint3,036,941,048bytes;
whole-process143.19s includes comparison work and is not a request-latency
benchmark. Per-emission records preserve partial observations; completion/failure
separate from before-run identity. No performance/energy/sustained thermal claim.
Model/state resources are separate from learning progress. Astra final review
approved, no remaining actionable findings; audited saved evidence without
running more model workloads. Owner review remains pending.

Reproduction: lesson26. Evidence remains ignored under `docs/evidence/t26-*`.
