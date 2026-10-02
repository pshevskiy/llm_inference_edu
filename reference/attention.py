"""Independent NumPy causal GQA reference for T03, with named intermediates."""

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ROOT / "fixtures" / "t03-expected.json"


def rotate(x, start, base):
    dim = x.shape[-1]
    frequency = base ** (-2 * np.arange(dim // 2) / dim)
    angle = (start + np.arange(x.shape[0]))[:, None, None] * frequency[None, None, :]
    a, b = x[..., :dim//2], x[..., dim//2:]
    return np.concatenate([a*np.cos(angle)-b*np.sin(angle), b*np.cos(angle)+a*np.sin(angle)], axis=-1)


def generate():
    if np.__version__ != "2.3.2":
        raise RuntimeError("Use reference/requirements.txt")
    case = json.loads((ROOT / "fixtures" / "t03-inputs.json").read_text())
    t, h, qh, kvh, d = (case[k] for k in ["tokens", "hidden_dim", "query_heads", "kv_heads", "head_dim"])
    values = lambda name: np.asarray(case[name], dtype=np.float32).astype(np.float64)
    x = values("x").reshape(t, h)
    q = (x @ values("query").reshape(qh*d, h).T).reshape(t,qh,d)
    k = (x @ values("key").reshape(kvh*d, h).T).reshape(t,kvh,d)
    v = (x @ values("value").reshape(kvh*d, h).T).reshape(t,kvh,d)
    qr = rotate(q, case["start_position"], float(np.float32(case["rope_base"])))
    kr = rotate(k, case["start_position"], float(np.float32(case["rope_base"])))
    # Expand by indexing for the oracle only; Rust keeps KV heads compact.
    groups = np.arange(qh) // (qh // kvh)
    scores = np.einsum("thd,shd->hts", qr, kr[:,groups,:]) / np.sqrt(d)
    allowed = np.tril(np.ones((t,t), dtype=bool))
    scores = np.where(allowed[None,:,:], scores, -np.inf)
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities = exp / exp.sum(axis=-1, keepdims=True)
    context = np.einsum("hts,shd->thd", probabilities, v[:,groups,:])
    output = context.reshape(t,h) @ values("output").reshape(h,h).T
    named = dict(query=q, key=k, value=v, rotated_query=qr, rotated_key=kr, context=context, output=output)
    if not all(np.all(np.isfinite(value)) for value in named.values()):
        raise ValueError("Non-finite attention oracle")
    return json.dumps({"numpy":np.__version__, "outputs":{k:v.astype(np.float32).reshape(-1).tolist() for k,v in named.items()}}, indent=2, allow_nan=False)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = generate()
    if args.write:
        EXPECTED.write_text(result)
        print("Wrote t03-expected.json")
    elif EXPECTED.read_text() != result:
        raise SystemExit("Attention fixture differs; investigate")
    else:
        print("Attention fixture verified")


if __name__ == "__main__":
    main()
