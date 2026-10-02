# 17 — Export a verified graph before asking where it runs

`reference/coreml_tiny_graph.py` is a development export graph for the tiny T04
fixture. Rust remains our production engine. The graph reads the same FP32
weights, performs embedding, prenorm attention, split-half RoPE, causal GQA,
SwiGLU/residuals, final RMSNorm and the tied head. It returns35named arrays so
conversion can be checked at each stage. Core ML is a second public execution
experiment, not a replacement for the Rust reference.

`reference/coreml_tiny.py` first checks four lengths1..4 against independent
NumPy equations, then traces all four shapes. The converter receives int32 token
IDs with enumerated lengths and explicit FP32 output/compute types. Compilation
is deferred with `skip_model_load=True`; CPU_ONLY is explicit. A second process
compiles the saved package through `coremltools.utils.compile_model`. A third
loads the compiled model with CPU_ONLY and compares every output to NumPy,
using the original atol1e-5/rtol1e-4. Greedy IDs must agree. A future token must
preserve earlier causal logits within that same gate.

Phase timers separate conversion/save, compilation, compiled-model load and
prediction. Three process peaks include Python imports and phase storage; they
do not attribute memory used by separate system compiler services. File bytes,
logical Rust weight bytes and physical process peaks describe different things.
No repeated benchmark, energy, sustained thermal or ANE placement is established.
The actual M1 Pro identity, pinned tools, fixture/source/artifact hashes and raw
phase logs are local ignored evidence. Fresh directories preserve failed runs.

Exported inputs/outputs are token-major; Core ML may choose other internal
layouts. The development graph expands compact KV heads by indexing, whereas
our Rust implementation preserves compact KV storage. This is an explicitly
small correctness graph; neither its layout nor its timing establishes an
optimized ANE transformer implementation. The selected deployment floor is
macOS13 (iOS16 equivalent); no state API is involved in this stateless experiment.

**Trace:** At token position0, every RoPE angle is0, so Q/K are unchanged. Query
head2 maps to KVhead1 (`2/(4/2)=1`). The causal mask allows just position0; softmax
of one score is1, so head2's context equals V[0,1,:]. The output projection mixes
all four context heads before the residual. Returning Q, K, rotated Q/K and
context separately lets us distinguish rotation, mapping and masking errors.

```sh
reference/.venv/bin/python -m pip install -r reference/coreml-lock.txt
reference/.venv/bin/python reference/decoder.py
reference/.venv/bin/python reference/coreml_tiny.py run docs/evidence/t17-new-run
```

The last command requires actual macOS and is an opt-in export/hardware
correctness check. It creates a fresh directory, records failures, and uses
separate `/usr/bin/time -l` child logs for process peaks. Ordinary Rust tests
need no Python, model downloads or Apple hardware.

Public sources: [conversion API](https://apple.github.io/coremltools/docs-guides/source/new-conversion-options.html),
[typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html),
[flexible shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html),
and [pinned Core ML Tools9.0](https://github.com/apple/coremltools/releases/tag/9.0).
CPU_ONLY correctness permits proceeding to T18; it says nothing about ANE
placement or a pretrained cache. Those need separate evidence.

**Exercise:** The exported Q tensor has shape[tokens,4,8]. For token2, head3,
coordinate5, compute the token-major offset. Which compact KVhead supplies that
query head's values, and why does copying four KVheads waste storage?
