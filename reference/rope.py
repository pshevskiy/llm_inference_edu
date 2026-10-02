"""Independent NumPy split-half rotary-position equations for T02."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ROOT / "fixtures" / "t02-expected.json"


def generate():
    if np.__version__ != "2.3.2":
        raise RuntimeError("Use reference/requirements.txt")
    inputs = json.loads((ROOT / "fixtures" / "t02-inputs.json").read_text())
    outputs = {}
    for case in inputs["cases"]:
        dim = case["head_dim"]
        x = np.asarray(case["x"], dtype=np.float32).astype(np.float64)
        x = x.reshape(case["tokens"], case["heads"], dim)
        frequencies = float(np.float32(case["base"])) ** (-2 * np.arange(dim // 2) / dim)
        positions = case["start_position"] + np.arange(case["tokens"], dtype=np.float64)
        angles = positions[:, None, None] * frequencies[None, None, :]
        a, b = x[..., : dim // 2], x[..., dim // 2 :]
        y = np.concatenate(
            [a * np.cos(angles) - b * np.sin(angles),
             b * np.cos(angles) + a * np.sin(angles)], axis=-1,
        )
        if not np.all(np.isfinite(y)):
            raise ValueError("Non-finite rotary reference")
        outputs[case["name"]] = y.astype(np.float32).reshape(-1).tolist()
    return json.dumps({"numpy": np.__version__, "outputs": outputs}, indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = generate()
    if args.write:
        EXPECTED.write_text(result)
        print("Wrote t02-expected.json")
    elif EXPECTED.read_text() != result:
        raise SystemExit("Rotary fixture differs; investigate before regenerating")
    else:
        print("Rotary fixture verified")


if __name__ == "__main__":
    main()
