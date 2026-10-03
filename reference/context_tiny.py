"""Portable larger-context golden logits from independent NumPy equations."""
import argparse
import json
from pathlib import Path
from decoder import outputs_for
import numpy as np

PATH = Path(__file__).resolve().parent.parent / "fixtures/t28-tiny-context.json"

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args()
    ids=([1,5,7,2,9,4,0,63]*16)
    outputs=outputs_for(ids)["logits"]
    result=json.dumps({"numpy":np.__version__,"ids":ids,
        "positions":[{"position":p,"logits":outputs[p].tolist(),"greedy":int(np.argmax(outputs[p]))}
        for p in (0,31,63,127)]},indent=2,allow_nan=False)+"\n"
    if args.write: PATH.write_text(result);print("Wrote independent 128-position tiny fixture")
    elif PATH.read_text()!=result: raise SystemExit("Larger-context fixture differs; investigate")
    else: print("Independent 128-position tiny fixture verified")

if __name__=="__main__":main()
