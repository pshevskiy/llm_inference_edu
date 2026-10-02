# 05 — Loading one pinned checkpoint

The selected artifact is [Qwen2.5-0.5B-Instruct at revision
7ae557604adf67be50417f59c2c2f167def9a775](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775),
licensed Apache-2.0. `fixtures/qwen-inventory.json` records every selected asset's
size and SHA256. `fixtures/qwen-tensors.json` records all 290 tensor names,
shapes and BF16 storage. We retain tokenizer/config/generation files as assets,
without executing repository code. Full assets live under ignored `models/`.

`load_qwen` checks the exact repository/revision and single-shard layout, verifies
config size/hash, checks supported dimensions and checked payload arithmetic,
then verifies the checkpoint size/hash. The established Safetensors parser
validates file bounds and coverage. We require every expected tensor with its
exact shape and BF16 type; missing or extra names fail. Only then do we decode.
All intermediate ownership stays local; a failure returns no partial model.

BF16 stores the high 16 bits of an FP32 value. Two little-endian bytes `80 3f`
form `0x3f80`; shifting left gives FP32 bits `0x3f800000`, exactly `1.0`.
Expansion adds zero low bits, so there is no extra rounding. Non-finite weights
are rejected. Matrix weights remain row-major `[output,input]`; no transpose.

Qwen has hidden width 896, 14 query heads, two KV heads, head width 64,
feed-forward width 4864 and 24 layers. Q/K/V biases are real tensors and are
retained. The output head shares `model.embed_tokens.weight`; no second
151936-by-896 matrix exists. T06 will connect these buffers to our decoder and
qualify its arithmetic; successful loading alone does not establish parity.

The raw file is 988097824 bytes. Expanded FP32 payload is 1976131072 bytes.
A single release-mode load on this MacBookPro18,3 (M1 Pro, 16 GB, macOS 26.6.2)
took 4.12 seconds, with maximum resident set size 2813886464 bytes and macOS
peak memory footprint 2969914008 bytes. These are different OS metrics,
separate from logical payload and file size. Raw bytes plus expanded weights
coexist during loading and raw bytes drop on return. The planning load payload
budget is 6 GiB, not an OS reservation or an inference admission guarantee.
No Python model was resident during this measurement. Sequential byte decoding
and moving owned buffers avoid a second full weight copy; loading still allocates.

Runtime parser pins: safetensors 0.6.2, sha2 0.10.9, Serde 1.0.228,
serde_json 1.0.149; locked Rust 1.90.0. Downloader: Python 3.12.8 and host curl.
The next reference setup selects Transformers 4.46.3, PyTorch 2.6.0 and
Python tokenizers 0.20.3; they are not installed or run in T05.

Reproduce the explicitly opted-in download and bounded load from the root:

```sh
python3 reference/download_qwen.py --download
cargo build --release --example load_qwen
/usr/bin/time -l target/release/examples/load_qwen models/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775
```

Ordinary tests use tiny locally serialized tensors, without network or weights.
They cover missing/extra tensors, shape/dtype mismatch, truncated/extra bytes,
NaN, overflow, budget rejection, corrupted hashes and unsupported shard lists.
The example writes only small validated metadata after successful loading.

Exercise: calculate the logical FP32 byte count of one K-projection weight
matrix, using two KV heads and head width 64. Identify the contiguous input
row for output feature 3. Leave the answer for owner review.
