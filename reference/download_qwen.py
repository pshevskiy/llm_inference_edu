"""Explicit opt-in pinned public checkpoint download with hashes and atomic files."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent.parent
REPO="Qwen/Qwen2.5-0.5B-Instruct"
REVISION="7ae557604adf67be50417f59c2c2f167def9a775"
FILES=["LICENSE","README.md","config.json","generation_config.json","tokenizer.json",
       "tokenizer_config.json","merges.txt","vocab.json","model.safetensors"]


def digest(path):
    result=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(4*1024*1024),b""):result.update(block)
    return result.hexdigest()


def matches(path, entry):
    if not path.exists() or path.stat().st_size != entry["size"]:
        return False
    if "lfs" in entry:
        return digest(path) == entry["lfs"]["sha256"]
    result=hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open("rb") as file:
        for block in iter(lambda:file.read(4*1024*1024),b""):result.update(block)
    return result.hexdigest() == entry["blobId"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download",action="store_true",help="Explicitly download roughly 1 GB")
    args=parser.parse_args()
    if not args.download:raise SystemExit("Download requires --download; no files fetched")
    directory=ROOT/"models"/"qwen2.5-0.5b-instruct"/REVISION
    directory.mkdir(parents=True,exist_ok=True)
    url=f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    # Use the installed curl's verified system TLS configuration; never disable
    # certificate verification to work around a Python certificate-store issue.
    response=subprocess.run(["curl","--fail","--location","--silent","--show-error",url],check=True,capture_output=True,text=True)
    metadata=json.loads(response.stdout)
    if metadata["sha"]!=REVISION:raise ValueError("Unexpected repository revision")
    entries={entry["rfilename"]:entry for entry in metadata["siblings"]}
    required=sum(entries[name]["size"] for name in FILES)
    if shutil.disk_usage(directory).free<required:raise ValueError("Insufficient disk space")
    assets={}
    for name in FILES:
        entry=entries[name];target=directory/name
        if matches(target,entry):
            print(f"Verified existing {name}",flush=True)
        else:
            temporary=directory/(name+".part")
            download=f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
            print(f"Fetching {name}: {entry['size']} bytes",flush=True)
            subprocess.run(["curl","--fail","--location","--silent","--show-error","--retry","2","--output",str(temporary),download],check=True)
            if not matches(temporary,entry):
                temporary.unlink(missing_ok=True);raise ValueError(f"Integrity failure: {name}")
            temporary.replace(target)
            print(f"Downloaded {name}: {target.stat().st_size} bytes",flush=True)
        assets[name]={"bytes":target.stat().st_size,"sha256":digest(target)}
    inventory={"repository":REPO,"revision":REVISION,"license":metadata.get("cardData",{}).get("license"),
               "format":"safetensors","shards":["model.safetensors"],"assets":assets}
    (ROOT/"fixtures"/"qwen-inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")
    print(f"Saved pinned inventory; local directory: {directory}",flush=True)


if __name__=="__main__":main()
