"""Opt-in streamed CPU FP32 context reference; never calls production Rust.

One vocabulary row at a time in cached qualification. Prefix/full calibration
keeps only the final logits row via pinned HF API, but eager attention uses a
quadratic score matrix, explicitly estimated and bounded before admission.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(ROOT / "models/.hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ACTIVE_OUTPUT = None
STAGES = (32, 128, 256, 512, 1024, 2048)
TEXT = "Rust inference uses a contiguous key and value cache. A pending token must be evaluated exactly once before appending a calculator result. Two plus three equals five. Memory layout matters. 世界 café.\n"

def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--capacity", required=True, type=int, choices=STAGES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    global ACTIVE_OUTPUT
    if args.output.exists():
        parser.error("choose a fresh result directory; preceding evidence is preserved")
    versions = {n:importlib.metadata.version(n) for n in ("torch","transformers","numpy","tokenizers","safetensors")}
    assert versions == {"torch":"2.6.0","transformers":"4.46.3","numpy":"2.3.2","tokenizers":"0.20.3","safetensors":"0.6.2"}
    inventory = json.loads((ROOT / "fixtures/qwen-inventory.json").read_text())
    for name, entry in inventory["assets"].items():
        path = args.model / name
        assert path.stat().st_size == entry["bytes"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b""): digest.update(chunk)
        assert digest.hexdigest() == entry["sha256"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    base = tokenizer.encode(TEXT, add_special_tokens=False)
    ids = (base * ((args.capacity+len(base)-1)//len(base)))[:args.capacity]
    args.output.mkdir(parents=True)
    ACTIVE_OUTPUT = args.output
    identity = {"status":"running","source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inventory_sha256":hashlib.sha256((ROOT / "fixtures/qwen-inventory.json").read_bytes()).hexdigest(),
        "repository":inventory["repository"],"revision":inventory["revision"],"capacity":args.capacity,
        "ids":ids,"workload_sha256":hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest(),
        "weight_sha256":inventory["assets"]["model.safetensors"]["sha256"],"versions":versions,
        "device":"cpu","compute_dtype":"float32","attention":"eager","threads":1,
        "logit_atol":0.001,"rtol":0.0001,"host":{"system":platform.system(),"machine":platform.machine(),
        "os_version":command(["sw_vers","-productVersion"]),"os_build":command(["sw_vers","-buildVersion"]),
        "chip":command(["sysctl","-n","machdep.cpu.brand_string"]),"memory_bytes":command(["sysctl","-n","hw.memsize"]),
        "pressure":command(["memory_pressure","-Q"])}}
    (args.output / "identity.json").write_text(json.dumps(identity,indent=2)+"\n")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    config = json.loads((args.model / "config.json").read_text())
    calibration_estimate = (1976131072 + 2*24*args.capacity*2*64*4
        + 2*config["num_attention_heads"]*args.capacity**2*4
        + 4*config["intermediate_size"]*args.capacity*4 + 1024**3)
    assert calibration_estimate <= 6*1024**3, "reference calibration estimate exceeds6GiB"
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
        local_files_only=True, trust_remote_code=False, attn_implementation="eager",
        low_cpu_mem_usage=False).eval().to("cpu")
    load_seconds = time.perf_counter()-start
    assert args.capacity <= model.config.max_position_embeddings
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    digest = hashlib.sha256()
    selected = {}
    checkpoints = {0, args.capacity//2-1, args.capacity-1}
    cache = None
    timings = []
    with (args.output / "logits.bin").open("wb") as stream:
        for position, token in enumerate(ids):
            start = time.perf_counter()
            output = model(input_ids=torch.tensor([[token]]), past_key_values=cache,
                use_cache=True, num_logits_to_keep=1)
            timings.append(time.perf_counter()-start)
            cache = output.past_key_values
            logits = output.logits[0,-1].numpy().astype("<f4",copy=True)
            assert logits.shape == (model.config.vocab_size,) and np.isfinite(logits).all()
            payload = logits.tobytes()
            stream.write(payload)
            digest.update(payload)
            if position in checkpoints: selected[position] = logits
            if (position+1)%32 == 0:
                print(f"reference {position+1}/{args.capacity}",flush=True)
    del cache, output
    calibration = []
    for position, expected in selected.items():
        actual = model(input_ids=torch.tensor([ids[:position+1]]),use_cache=False,
            num_logits_to_keep=1).logits[0,-1].numpy()
        assert np.isfinite(actual).all()
        error = np.abs(actual-expected)
        budget = error / (0.001+0.0001*np.abs(expected))
        record = {"position":position,"max_abs":float(error.max()),"max_budget":float(budget.max()),
            "greedy_equal":int(np.argmax(actual))==int(np.argmax(expected))}
        calibration.append(record)
        print("reference cached/prefix", record, flush=True)
        assert record["max_budget"] <= 1 and record["greedy_equal"], "frozen reference stability gate failed"
    # Independent expected rows for the actual sampled request, not the original
    # teacher-forced stream after it diverges at the first emitted token.
    request_prompt = ids[:args.capacity-12]
    tool_ids = ids[args.capacity-6:args.capacity-2]
    request_rows = []
    def save_row(name, logits):
        values = logits.numpy().astype("<f4",copy=True)
        assert np.isfinite(values).all()
        payload = values.tobytes()
        filename = name+".bin"
        (args.output / filename).write_bytes(payload)
        record = {"name":name,"file":filename,"elements":len(values),
            "sha256":hashlib.sha256(payload).hexdigest(),"greedy":int(np.argmax(values))}
        request_rows.append(record)
        return record["greedy"]
    output = model(input_ids=torch.tensor([request_prompt]),use_cache=True,num_logits_to_keep=1)
    first = save_row("request-prefill",output.logits[0,-1])
    cache = output.past_key_values
    output = model(input_ids=torch.tensor([[first]]),past_key_values=cache,use_cache=True,num_logits_to_keep=1)
    second = save_row("request-decode",output.logits[0,-1])
    cache = output.past_key_values
    for token in [second]+tool_ids:
        output = model(input_ids=torch.tensor([[token]]),past_key_values=cache,use_cache=True,num_logits_to_keep=1)
        cache = output.past_key_values
    next_id = save_row("request-continuation",output.logits[0,-1])
    request = {"prompt_ids":request_prompt,"tool_fixture_ids":tool_ids,
        "generated_ids":[first,second],"continuation_id":next_id,"rows":request_rows}
    del output,cache
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == "Darwin" else peak*1024
    assert peak_bytes <= 6*1024**3, "reference exceeds planning RSS budget"
    manifest = {"status":"passed","source_sha256":identity["source_sha256"],"request":request,"repository":inventory["repository"],"revision":inventory["revision"],
        "weight_sha256":inventory["assets"]["model.safetensors"]["sha256"],
        "capacity":args.capacity,"ids":ids,"vocab_size":model.config.vocab_size,
        "workload_sha256":hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest(),
        "logits_file":"logits.bin","logits_bytes":args.capacity*model.config.vocab_size*4,
        "logits_sha256":digest.hexdigest(),"logit_atol":0.001,"rtol":0.0001,
        "device":"cpu","compute_dtype":"float32","storage_dtype":"bfloat16",
        "attention":"eager","threads":1,"versions":versions,"reference_load_seconds":load_seconds,
        "reference_peak_rss_bytes":peak_bytes,"reference_calibration_estimate_bytes":calibration_estimate,"reference_step_seconds":timings,
        "calibration":calibration,"host":{"system":platform.system(),"machine":platform.machine(),
        "os_version":command(["sw_vers","-productVersion"]),"os_build":command(["sw_vers","-buildVersion"]),
        "chip":command(["sysctl","-n","machdep.cpu.brand_string"]),
        "memory_bytes":command(["sysctl","-n","hw.memsize"]),
        "pressure":command(["memory_pressure","-Q"])}}
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.output / "completion.json").write_text(json.dumps({"status":"passed","peak_rss_bytes":peak_bytes})+"\n")
    print(f"reference complete capacity={args.capacity}, peak RSS={peak_bytes}",flush=True)

if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ACTIVE_OUTPUT is not None:
            (ACTIVE_OUTPUT / "completion.json").write_text(json.dumps({"status":"failed","error":str(error)})+"\n")
        raise
