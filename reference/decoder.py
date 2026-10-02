"""Independent NumPy two-layer decoder equations with named activations for T04."""

import argparse
import json
from pathlib import Path
import numpy as np
from attention import rotate

ROOT=Path(__file__).resolve().parent.parent
EXPECTED=ROOT/"fixtures"/"t04-expected.json"


def rms(x,w,eps): return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*w


def outputs_for(ids):
    if np.__version__!="2.3.2": raise RuntimeError("Use reference/requirements.txt")
    file=json.loads((ROOT/"fixtures"/"tiny-model.json").read_text()); c=file["config"]; w=file["weights"]
    h,d,qh,kvh,ff=(c[k] for k in ["hidden_dim","head_dim","query_heads","kv_heads","ff_dim"])
    eps=float(np.float32(c["norm_epsilon"]))
    fp=lambda v:np.asarray(v,dtype=np.float32).astype(np.float64)
    table=fp(w["embedding"]).reshape(c["vocab_size"],h)
    x=table[ids]; outputs={"embedding":x}; groups=np.arange(qh)//(qh//kvh)
    t=len(ids); allowed=np.tril(np.ones((t,t),dtype=bool))
    for i,layer in enumerate(w["layers"]):
        prefix=f"layer_{i}."; l={k:fp(v) for k,v in layer.items()}; an=rms(x,l["attention_norm"],eps)
        q=(an@l["query"].reshape(h,h).T).reshape(t,qh,d)
        k=(an@l["key"].reshape(kvh*d,h).T).reshape(t,kvh,d)
        v=(an@l["value"].reshape(kvh*d,h).T).reshape(t,kvh,d)
        qr=rotate(q,0,c["rope_base"]); kr=rotate(k,0,c["rope_base"])
        scores=np.einsum("thd,shd->hts",qr,kr[:,groups,:])/np.sqrt(d)
        scores=np.where(allowed[None,:,:],scores,-np.inf)
        exp=np.exp(scores-scores.max(axis=-1,keepdims=True)); p=exp/exp.sum(axis=-1,keepdims=True)
        context=np.einsum("hts,shd->thd",p,v[:,groups,:]).reshape(t,h)
        ao=context@l["attention_output"].reshape(h,h).T
        after=x+ao; fn=rms(after,l["ffn_norm"],eps)
        gate=fn@l["gate"].reshape(ff,h).T; up=fn@l["up"].reshape(ff,h).T
        activation=gate/(1+np.exp(-gate)); product=activation*up
        fo=product@l["down"].reshape(h,ff).T; x=after+fo
        named=dict(attention_norm=an,query=q,key=k,value=v,rotated_query=qr,rotated_key=kr,
                   context=context,attention_output=ao,after_attention=after,ffn_norm=fn,
                   gate=gate,up=up,activated_gate=activation,product=product,ffn_output=fo,output=x)
        outputs.update({prefix+k:v for k,v in named.items()})
    final=rms(x,fp(w["final_norm"]),eps); outputs["final_norm"]=final; outputs["logits"]=final@table.T
    if not all(np.all(np.isfinite(v)) for v in outputs.values()): raise ValueError("Non-finite decoder reference")
    return {k:v.astype(np.float32) for k,v in outputs.items()}


def generate():
    ids=json.loads((ROOT/"fixtures"/"t04-inputs.json").read_text())["token_ids"]
    outputs=outputs_for(ids)
    return json.dumps({"numpy":np.__version__,"outputs":{k:v.reshape(-1).tolist() for k,v in outputs.items()}},indent=2,allow_nan=False)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--write",action="store_true"); args=parser.parse_args()
    result=generate()
    if args.write: EXPECTED.write_text(result); print("Wrote t04-expected.json")
    elif EXPECTED.read_text()!=result: raise SystemExit("Decoder fixture differs; investigate")
    else: print("Decoder fixture verified")


if __name__=="__main__": main()
