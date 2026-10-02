# T01 numerical reference

`basic_ops.py` is an independent NumPy oracle for the explicit values in
`fixtures/t01-inputs.json`. It evaluates equations in FP64 after rounding
inputs to FP32, then rounds outputs to FP32 and saves
`fixtures/t01-expected.json`. It does not call Rust or reproduce the production
loops. RMSNorm uses the direct equation; SiLU uses the direct sigmoid equation.
The fixtures use small deterministic values, with no random seed or model files.

The recorded generation used Python 3.12.8 and NumPy 2.3.2 already installed on
the host. NumPy is pinned in `requirements.txt`; the script checks its version.

For an isolated environment, run these optional setup commands from the root
(Python 3.12 or later must already be installed):

```sh
python3 -m venv reference/.venv
reference/.venv/bin/python -m pip install -r reference/requirements.txt
reference/.venv/bin/python reference/basic_ops.py
```

The `.venv` is ignored by Git. Setup may need network access to install NumPy.
No ML runtime, Transformers, or model download is needed. When the required
NumPy is already available, `python3 reference/basic_ops.py` suffices.

The default command recomputes the oracle and verifies the saved JSON byte for
byte. To deliberately regenerate it after changing inputs or reference math:

```sh
reference/.venv/bin/python reference/basic_ops.py --write
```

Inspect any numerical difference before accepting a regenerated fixture. Both
JSON files are checked in. Ordinary Rust tests read those files through
`serde_json` and require neither Python nor network access
once Cargo's locked test dependencies are available.

T01 introduced JSON parsing only for tests. T04 also uses Serde/JSON to read
the synthetic model fixture in its runnable example, so those small parsers
are now runtime dependencies. This is not a pretrained-checkpoint reader.

The comparison uses the plan's provisional `atol=1e-5`, `rtol=1e-4`:

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

Every comparison checks finiteness. It also reports maximum absolute error,
maximum scale-aware relative error (`error / max(abs(expected), atol)`), and
maximum fraction of the allowed error budget. Exact row copies and exactly
representable residual sums additionally require exact equality.

```sh
cargo test --test basic_ops -- --nocapture --test-threads=1
```

The single test thread keeps the printed numerical evidence readable. These
operator fixtures do not qualify a full model or a backend.


## Opt-in Qwen reference (T06)

The owner approved this isolated setup and the pinned model download. Ordinary
Rust tests require neither these Python packages nor pretrained weights.
`real-requirements.txt` selects core versions; `real-lock.txt` captures the
complete installed environment. A fresh setup uses:

```sh
python3 -m venv reference/.venv
reference/.venv/bin/python -m pip install -r reference/real-lock.txt
reference/.venv/bin/python reference/qwen.py --model models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775
cargo run --release --example qwen_parity -- models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775
```

The Python program verifies all asset hashes and uses only local files,
`trust_remote_code=False`, CPU FP32 eager attention, one PyTorch thread, and
no cache. Hooks capture first/last block internals and every layer's output.
It saves bounded 2–3 token teacher-forced references under ignored
`docs/evidence/t06-reference/`; tokenizer expectations are a small Git fixture.
Run Python and Rust models sequentially. No MPS/GPU or runtime substitution.

The selected reference is [Transformers 4.46.3 Qwen2](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py).
The plain-message branch of the pinned checkpoint's chat template is supported;
tool calls/messages are not implemented here. Special tokens in content retain
the tokenizer's reference behavior. This does not make untrusted content safe
as a system instruction.

Full-model gates were calibrated independently using the reference's own
prefix-versus-full numerical variation. Logits use `atol=1e-3`, intermediate
arrays `atol=5e-3`, both `rtol=1e-4`; all values must be finite and all eight
teacher-forced greedy IDs must agree. See lesson 06 for the measured reason and
initial failed experiments. These gates apply only to this FP32 checkpoint on
short inputs; original synthetic/operator tolerances remain unchanged.

## Opt-in tiny Core ML export (T17)

The owner approved isolated export setup. `coreml-requirements.txt` pins the
selected tools; `coreml-lock.txt` pins the complete export environment. It shares
the ignored `reference/.venv` reference-only environment; Rust has no Core ML or
Python dependency. Use the commands and public API links in
[lesson17](../docs/lessons/17-coreml-tiny.md). The export graph uses only the tiny
fixture and checks35named outputs at lengths1..4. CPU_ONLY and FP32 are explicit;
no GPU inference, pretrained state or ANE placement is claimed. Results/artifacts
remain under ignored `docs/evidence/`. Export, compile and CPU check are separate
processes with individual load/timing/memory scopes.
