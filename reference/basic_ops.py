"""Independent NumPy equations for the fixed T01 inputs; not an inference engine."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "fixtures" / "t01-inputs.json"
EXPECTED = ROOT / "fixtures" / "t01-expected.json"


def fp32_values(case, field):
    # Rust reads the same FP32-rounded inputs. Evaluate the oracle equations in
    # FP64 to separate numerical reference accuracy from scalar FP32 loop order.
    return np.asarray(case[field], dtype=np.float32).astype(np.float64)


def probabilities(logits):
    exponentials = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def generate():
    if np.__version__ != "2.3.2":
        raise RuntimeError("Use the pinned NumPy version in reference/requirements.txt")
    cases = json.loads(INPUTS.read_text())
    outputs = {}

    case = cases["embedding"]
    table = fp32_values(case, "table").reshape(case["vocab_size"], case["width"])
    outputs["embedding"] = table[np.asarray(case["token_ids"], dtype=np.int64)]

    case = cases["residual_add"]
    outputs["residual_add"] = fp32_values(case, "x") + fp32_values(case, "residual")

    case = cases["rmsnorm"]
    x = fp32_values(case, "x").reshape(case["rows"], case["width"])
    epsilon = float(np.float32(case["epsilon"]))
    outputs["rmsnorm"] = (
        x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + epsilon)
        * fp32_values(case, "weight")
    )

    case = cases["silu"]
    x = fp32_values(case, "x")
    outputs["silu"] = x / (1.0 + np.exp(-x))

    case = cases["softmax"]
    logits = fp32_values(case, "logits").reshape(case["rows"], case["width"])
    outputs["softmax"] = probabilities(logits)

    case = cases["masked_softmax"]
    logits = fp32_values(case, "logits").reshape(case["rows"], case["width"])
    mask = np.asarray(case["mask"], dtype=bool).reshape(logits.shape)
    if not np.all(np.any(mask, axis=1)):
        raise ValueError("All-masked rows have no probability distribution")
    outputs["masked_softmax"] = probabilities(np.where(mask, logits, -np.inf))

    if not all(np.all(np.isfinite(values)) for values in outputs.values()):
        raise ValueError("Non-finite oracle output")
    result = {
        "reference": {
            "numpy": np.__version__,
            "arithmetic": "FP64 equations from FP32-rounded inputs; outputs rounded to FP32",
        },
        "outputs": {
            name: values.astype(np.float32).reshape(-1).tolist()
            for name, values in outputs.items()
        },
    }
    return json.dumps(result, indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate expected fixtures")
    args = parser.parse_args()
    expected = generate()
    if args.write:
        EXPECTED.write_text(expected)
        print(f"Wrote {EXPECTED.name} using NumPy {np.__version__}")
    else:
        if EXPECTED.read_text() != expected:
            raise SystemExit("Oracle fixture differs; investigate before regenerating")
        print(f"Fixture verified using NumPy {np.__version__}")


if __name__ == "__main__":
    main()
