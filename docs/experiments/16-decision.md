# T16 — Reject groupwise INT8 for the pinned pretrained model

The quality rule was frozen in `16-quality-rule.json` before the first candidate
model run. Its SHA-256 is recorded with actual host, model, public token IDs,
source snapshot and executable hashes in ignored
`docs/evidence/t16-qwen/identity.json`. The rule and thresholds were not changed
after results. Actual host: MacBookPro18,3, M1 Pro, 16 GB, macOS build25G83.

| Metric | FP32 / INT8 result | Frozen gate | Decision |
|---|---|---|---|
| Mean next-token loss | 3.008327665 / 3.002956854 nats | increase <= .05 | pass |
| Mean KL FP32→INT8 | .0143751891 nats | <= .01 | fail |
| Teacher-forced greedy agreement | 15/17 = .882352941 | >= .9 | fail |
| Maximum absolute logit change, all rows | 2.541073799 | <= .5 | fail |
| Logical weights, including scales/FP32 small tensors | 1,976,131,072 / 555,992,576 bytes | ratio <= .4 | pass (.281354098) |
| Fused versus once-reconstructed reference logits | exactly equal for all three cases | original gates or better | pass |

Three held-out public sentences contain 17 target-bearing positions. This is a
short smoke qualification, not broad language/task quality. Group size32;
byte values/scales per row, FP32 activations/accumulation/norms/biases. Tied
embedding/head and all large attention/FFN matrices were quantized. No attribution
to an individual matrix is established. A future selective-quantization or other
precision experiment needs its own frozen rule and task authorization. No INT4
work is included.

Physical point RSS was 2,963,292,160 bytes after FP32 quality/diagnostics,
1,372,356,608 after conversion, and 1,402,126,336 after INT8 quality/diagnostics.
These include live result logits, tokenizer, allocator/page retention and other
process storage; they are not isolated weight measurements. Whole-process peak
RSS was 3,298,279,424 bytes; peak memory footprint3,392,572,512 bytes. The process
includes original FP32 loading and later reconstruction for the numerical oracle,
so this is not an INT8-only startup peak. The BF16 checkpoint file is still
988,097,824 bytes; no quantized file was exported. Conversion consumes the model
one matrix at a time; no second complete FP32 decoder is retained. Expansion
before conversion remains a load cost, not a solved compressed-loader problem.

One TeacherForced and one Greedy sample ran per precision after held-out
correctness, with no warmups or repeated requests. In the fixed-token sample,
prefill was649.569375ms FP32 and726.607833ms INT8; decode was3.07200013 and
2.75928546tokens/s. These are raw diagnostics, not a statistical speed comparison.
Astra ran a brief portable test process during the surrounding execution window;
any timing overlap is uncontrolled, so these observations do not support a speed
claim. Warm-state/order/allocator/thermal effects are also uncontrolled. Raw mode
samples and token/continuation counts are in `results.json`. Repeated whole-model
performance is not run: sustained hardware opt-in remains pending. No energy or
sustained thermal result is claimed. Whole qualification wall time39.68seconds,
verified load5.201896s, conversion1.306452708s, reference expansion.638591167s;
one process, not distributions.

**Decision:** no pretrained INT8 representation was selected. FP32 is the sole
default model path and the CLI has no INT8 precision switch. Preserve the T15
standalone teaching primitive and T16 reproduction behind a nondefault
`int8-experiment` development feature. This feature preserves the rejected
experiment's evidence; it is not a retained/qualified optional model precision.
It uses the same decoder/cache implementation, not another production engine.
The feature was added after the negative measurement; arithmetic and the frozen
rule are unchanged. The recorded executable/source hashes identify the measured
binary before feature gating. Final checks validate both build configurations.

Reproduction commands are in lesson16. Raw local evidence remains ignored:
`t16-qwen/{identity,results,completion}.json`, `t16-qwen-process.txt`, ordinary
check logs and `t16-fp32-parity.txt`. Owner review pending.
