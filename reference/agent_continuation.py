"""Independent tiny NumPy continuation logits; no production Rust calls."""
import argparse
import json
from pathlib import Path
from decoder import outputs_for
import numpy as np

PATH = Path(__file__).resolve().parent.parent / "fixtures/t25-continuation.json"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    cases = []
    for ids in ([1, 5, 7, 41, 2, 3], [2, 5, 7, 41, 2, 3]):
        logits = outputs_for(ids)["logits"][-1]
        cases.append({"ids": ids, "logits": logits.tolist(), "greedy": int(np.argmax(logits))})
    result = json.dumps({"numpy": np.__version__, "cases": cases}, indent=2, allow_nan=False) + "\n"
    if args.write:
        PATH.write_text(result)
        print("Wrote independent continuation fixture")
    elif PATH.read_text() != result:
        raise SystemExit("Continuation fixture differs; investigate")
    else:
        print("Independent continuation fixture verified")

if __name__ == "__main__":
    main()
