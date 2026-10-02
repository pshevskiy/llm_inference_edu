# 18 — Persistent native cache with a qualified representation

The public stateful ML Program APIs require macOS15/iOS18 or later. The installed
macOS26.6.2 runtime and macOS26.5 Command Line Tools SDK meet that floor; this is
not iPhone validation. Core ML Tools9.0 accepted FP16 state and rejected FP32
state in a minimal public accumulator conversion. The exact rejection is saved,
rather than assuming that ordinary FP32 compute implies FP32 state support.

The first checkpoint candidate kept FP32 weights/activations/accumulation but
rounded KV entries to FP16. A same-source FP32-cache control passed all frozen
reference checks; single FP16 state failed. T18 then explicitly tested a different
internal cache representation against the same gate: `hi=FP16(x)`,
`lo=FP16(x-FP32(hi))`, `reconstructed=FP32(hi)+FP32(lo)`. This is approximate and
does not cover the full FP32 range. In particular, an overflowing FP16 leading
value cannot be recovered by the residual. It is not a general lossless format.
No numerical tolerance was widened and no model/weight precision silently changed.

There are four state planes per layer: K-leading, K-residual, V-leading and
V-residual, each [KVheads=2,capacity=32,D=64]. Across24layers,96FP16 planes use
786,432logical bytes per session, equal to the original two FP32 cache planes.
This improves precision over a single half plane; it saves no cache memory.
Core ML owns those planes and compiled weights. Python here is qualification/
export tooling, not another production inference engine. Rust caches are not
mirrored, and one compiled model serves independent native state objects.

Execution validates token ID, capacity and owned position before calling
`predict`. The graph embeds one token, normalizes, forms Q/K/V, rotates Q/K at
the absolute position, writes rotated K and unrotated V to that slot, reconstructs
the active cache for attention, and masks future capacity slots. Attention/FFN
residuals and the tied vocabulary projection remain FP32. State persists inside
one artifact for both sequential prefill and decode; there are no separately
loaded weight copies whose state is assumed to be shared. A prediction failure
or nonfinite output invalidates the session even if some planes were written;
reset creates fresh native state. Ordinary admission errors preserve the state.

The development graph expands GQA heads and reconstructs cached planes during
attention. Additional casts, residual writes and full-capacity access may cost
bandwidth; no zero-copy or optimized ANE claim follows. The export returns all49
reference-hook outputs for diagnosis. Correctness prediction timing includes
that diagnostic interface, and is not the generation benchmark contract.

**Trace:** FP32 value1.0001 rounds to an FP16 leading value1. Its residual is
computed in FP32, stored as a second FP16 value, then added back in FP32. This
particular value reconstructs exactly; that is not true of every FP32 value.
At position2, layer0 K head1 coordinate3 is stored in both `k0` and `kl0` at
`(1*32+2)*64+3`. Attention reconstructs it before its dot with a query; future
positions3..31 are masked even if storage contains stale values. A new independent
state object starts with zero planes and its own position0.

The paired source and traced graph passed the original reference checks. Actual
CPU_ONLY compiled Core ML outputs passed all392arrays (49x8positions), all8greedy
IDs, three interleaved independent states, invalid-admission state hashes and
reset replay. This qualifies only the tested short inputs at capacity32; longer
contexts, wider activation ranges, mobile resources and ANE placement remain
unqualified. The default Rust/CLI path stays FP32 CPU. See experiment18 for actual
limits and phase measurements.

```sh
reference/.venv/bin/python reference/coreml_qwen.py \
  models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775 \
  docs/evidence/t18-new-preflight
reference/.venv/bin/python reference/coreml_qwen_export.py run \
  models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775 \
  docs/evidence/t18-new-preflight docs/evidence/t18-new-export
reference/.venv/bin/python reference/test_coreml_session.py
```

Export/compile/predict are opt-in, phase-separated, require a fresh directory and
local verified checkpoint/T06 references. The last command tests failure/admission
contracts with a fake predictor and needs no Apple hardware. Public sources:
[stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
and [pinned backend state-type restriction](https://github.com/apple/coremltools/blob/9.0/coremltools/converters/mil/backend/mil/load.py).

**Exercise:** Each K/V pair uses two FP16 planes. For one layer at capacity64,
KVheads2 and D64, compute the total logical cache bytes. Explain why this doubles
when capacity doubles even though only one token is processed in each call.
