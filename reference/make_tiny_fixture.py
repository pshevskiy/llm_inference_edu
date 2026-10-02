"""Explicit deterministic synthetic weights for T04; no shared RNG assumptions."""

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def values(count, seed):
    return [((i*17+seed)%97-48)/256 for i in range(count)]


def main():
    config=dict(vocab_size=64,layers=2,hidden_dim=32,query_heads=4,kv_heads=2,
                head_dim=8,ff_dim=64,max_tokens=32,rope_base=10000,norm_epsilon=1e-5)
    layers=[]
    for i in range(2):
        seed=13+i*29
        layers.append(dict(
            attention_norm=[1+v for v in values(32,seed)],
            query=values(1024,seed+1), key=values(512,seed+2), value=values(512,seed+3),
            attention_output=values(1024,seed+4), ffn_norm=[1+v for v in values(32,seed+5)],
            gate=values(2048,seed+6),up=values(2048,seed+7),down=values(2048,seed+8),
        ))
    weights=dict(embedding=values(2048,7),layers=layers,final_norm=[1+v for v in values(32,43)])
    (ROOT/"fixtures"/"tiny-model.json").write_text(json.dumps(dict(config=config,weights=weights),indent=2)+"\n")
    (ROOT/"fixtures"/"t04-inputs.json").write_text(json.dumps(dict(token_ids=[1,5,7]),indent=2)+"\n")
    print("Wrote explicit tiny-model weights and token IDs")


if __name__=="__main__": main()
