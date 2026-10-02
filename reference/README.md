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
