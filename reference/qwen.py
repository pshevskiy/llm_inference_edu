"""Opt-in, CPU FP32 pinned Transformers parity; never calls the Rust engine."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models/.hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer, AutoModelForCausalLM

REV = "7ae557604adf67be50417f59c2c2f167def9a775"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    for name, version in [("torch", "2.6.0"), ("transformers", "4.46.3"), ("tokenizers", "0.20.3"), ("numpy", "2.3.2"), ("safetensors", "0.6.2")]:
        assert importlib.metadata.version(name) == version
    inventory = json.loads((ROOT / "fixtures/qwen-inventory.json").read_text())
    assert inventory["revision"] == REV
    for name, asset in inventory["assets"].items():
        path = args.model / name
        assert path.stat().st_size == asset["bytes"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(chunk)
        assert digest.hexdigest() == asset["sha256"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    cases = []
    for text in ["Hello!", "Rust is", "  a\t b\n", "Привет 🌍 café", "<|im_start|>user\nx<|im_end|>\n", "", "世界🙂"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        cases.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=False)})
    chats = []
    for messages in [
        [{"role":"user","content":"Hello!"}],
        [{"role":"system","content":"Be brief."},{"role":"user","content":"世界🙂"},{"role":"assistant","content":"Sure."},{"role":"user","content":"  next\n"}],
        [{"role":"user","content":""},{"role":"system","content":"Late system."}],
    ]:
        for generation in [False, True]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation)
            chats.append({"messages":messages,"generation_prompt":generation,"text":text,"ids":ids})
    (ROOT / "fixtures/t06-tokenizer.json").write_text(json.dumps({"revision":REV,"cases":cases,"chats":chats}, ensure_ascii=False, indent=2)+"\n")
    print("Verified assets, saved independent tokenizer cases", flush=True)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True,
        trust_remote_code=False, attn_implementation="eager", low_cpu_mem_usage=False).eval().to("cpu")
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    destination = ROOT / "docs/evidence/t06-reference"
    destination.mkdir(parents=True, exist_ok=True)
    # Separate full-model gates calibrated from reference prefix/full variability.
    # Frozen before evaluating held-out Rust; initial strict failures are retained.
    manifest = {"revision":REV,"device":"cpu","dtype":"float32","attention":"eager","threads":1,
        "logit_atol":0.001,"intermediate_atol":0.005,"rtol":0.0001,"versions":{n:importlib.metadata.version(n) for n in ["torch","transformers","tokenizers","numpy","safetensors"]},"cases":[]}
    for case_index, text in enumerate(["Hello!", "Rust is", "Two plus two"]):
        ids = tokenizer.encode(text, add_special_tokens=False)
        captured = {}
        handles = []
        def save(name, input_value=False):
            def hook(module, inputs, output):
                value = inputs[0] if input_value else output
                if isinstance(value, tuple): value = value[0]
                captured[name] = value.detach().contiguous().cpu().numpy().astype("<f4", copy=True).reshape(-1)
            return hook
        handles.append(model.model.embed_tokens.register_forward_hook(save("embedding")))
        handles.append(model.model.norm.register_forward_hook(save("final_norm")))
        for layer_index in [0,23]:
            layer = model.model.layers[layer_index]
            prefix = f"layer_{layer_index}."
            for name, module, input_value in [
                ("attention_norm",layer.input_layernorm,False),
                ("query",layer.self_attn.q_proj,False),("key",layer.self_attn.k_proj,False),("value",layer.self_attn.v_proj,False),
                ("context",layer.self_attn.o_proj,True),("attention_output",layer.self_attn.o_proj,False),
                ("ffn_norm",layer.post_attention_layernorm,False),
                ("gate",layer.mlp.gate_proj,False),("up",layer.mlp.up_proj,False),
                ("product",layer.mlp.down_proj,True),("ffn_output",layer.mlp.down_proj,False),("output",layer,False),
            ]:
                handles.append(module.register_forward_hook(save(prefix+name,input_value)))
        for layer_index in range(1,23):
            handles.append(model.model.layers[layer_index].register_forward_hook(save(f"layer_{layer_index}.output")))
        output = model(input_ids=torch.tensor([ids]), use_cache=False)
        captured["logits"] = output.logits.contiguous().numpy().astype("<f4").reshape(-1)
        records = []
        for name, values in captured.items():
            assert np.isfinite(values).all()
            filename = f"{case_index}-{name}.bin"
            payload = values.tobytes()
            (destination / filename).write_bytes(payload)
            records.append({"name":name,"file":filename,"elements":len(values),"sha256":hashlib.sha256(payload).hexdigest()})
        manifest["cases"].append({"text":text,"ids":ids,"arrays":records})
        print(f"case {text!r}: {len(ids)} tokens; {len(records)} arrays; logits argmax {output.logits[0].argmax(-1).tolist()}", flush=True)
        # Independent calibration: unchanged implementation, different linear
        # shapes. It tests the gate against reference-vs-reference variability,
        # without consulting any Rust outputs.
        full = {name: values.copy() for name, values in captured.items()}
        for position in range(len(ids)):
            prefix_logits = model(input_ids=torch.tensor([ids[:position+1]]), use_cache=False).logits[0,-1].numpy()
            peak = (0.0, "", 0.0)
            for name, values in full.items():
                expected = values.reshape(len(ids),-1)[position]
                actual = prefix_logits if name == "logits" else captured[name].reshape(position+1,-1)[-1]
                assert np.isfinite(actual).all() and np.isfinite(expected).all()
                error = np.abs(actual-expected)
                atol = manifest["logit_atol"] if name == "logits" else manifest["intermediate_atol"]
                budget = error / (atol + manifest["rtol"]*np.abs(expected))
                if budget.max() > peak[0]: peak = (float(budget.max()), name, float(error.max()))
            print(f"reference prefix {position+1} vs full: max_budget={peak[0]:.6f} array={peak[1]} max_abs={peak[2]:.8e}",flush=True)

            assert peak[0] <= 1.0, "reference exceeds calibrated stability gate"
        for handle in handles: handle.remove()
        del output, captured
    (destination / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Saved bounded teacher-forced CPU reference", flush=True)

if __name__ == "__main__": main()
