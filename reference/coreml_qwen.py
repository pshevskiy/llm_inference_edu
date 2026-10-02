"""T18 qualification preflight: public state dtype audit and independent numerical gate."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT=Path(__file__).resolve().parent.parent
REV='7ae557604adf67be50417f59c2c2f167def9a775'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def write(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False); f.write('\n')

def audit_state(ct,np,torch):
    class Accumulator(torch.nn.Module):
        def __init__(self,dtype):
            super().__init__(); self.register_buffer('cache',torch.zeros(1,dtype=dtype))
        def forward(self,x):
            self.cache.add_(x); return self.cache.float()
    records=[]
    for label,dtype,npdtype in [('FP32',torch.float32,np.float32),('FP16',torch.float16,np.float16)]:
        graph=torch.jit.trace(Accumulator(dtype).eval(),torch.tensor([1.]),check_trace=False)
        try:
            model=ct.convert(graph,inputs=[ct.TensorType(name='x',shape=(1,),dtype=np.float32)],
                states=[ct.StateType(name='cache',wrapped_type=ct.TensorType(shape=(1,),dtype=npdtype))],
                outputs=[ct.TensorType(name='y',dtype=np.float32)],compute_precision=ct.precision.FLOAT32,
                compute_units=ct.ComputeUnit.CPU_ONLY,minimum_deployment_target=ct.target.macOS15,
                skip_model_load=True)
            records.append({'dtype':label,'conversion':'accepted','model_type':model.get_spec().WhichOneof('Type')})
        except Exception as e:
            records.append({'dtype':label,'conversion':'rejected','error':str(e)})
    return records

def check_graph(model,manifest,reference,np,torch,dtype):
    results=[]; passed=True
    for case in manifest['cases']:
        model.reset(dtype)
        expected={}
        for array in case['arrays']:
            file=reference/array['file']
            if sha(file)!=array['sha256']: raise ValueError('reference array hash mismatch')
            expected[array['name']]=np.fromfile(file,dtype='<f4').reshape(len(case['ids']),-1)
        for p,token in enumerate(case['ids']):
            actual=model(torch.tensor([token],dtype=torch.int32),torch.tensor([p],dtype=torch.int32))
            arrays={}
            for name,value in actual.items():
                a=value.numpy().reshape(-1); b=expected[name][p]
                if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('invalid reference comparison')
                error=np.abs(a.astype(np.float64)-b.astype(np.float64))
                atol=manifest['logit_atol'] if name=='logits' else manifest['intermediate_atol']
                fraction=float((error/(atol+manifest['rtol']*np.abs(b))).max())
                good=fraction<=1
                arrays[name]={'max_absolute_error':float(error.max()),'max_budget_fraction':fraction,'passed':good}
                passed &= good
            greedy=int(actual['logits'].argmax(-1)[0])==int(expected['logits'][p].argmax())
            passed &= greedy
            results.append({'text':case['text'],'position':p,'token':token,'arrays':arrays,'greedy_exact':greedy})
    return {'passed':bool(passed),'positions':results}

def run(directory,output):
    output.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HOME']=str(ROOT/'models/.reference-cache')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    import coremltools as ct
    from coreml_qwen_graph import QwenStateExport
    from coreml_tiny import versions
    versions(); torch.set_num_threads(1); torch.set_grad_enabled(False)
    inventory=json.loads((ROOT/'fixtures/qwen-inventory.json').read_text())
    if inventory['revision']!=REV: raise ValueError('wrong model revision')
    for name,asset in inventory['assets'].items():
        file=directory/name
        if file.stat().st_size!=asset['bytes'] or sha(file)!=asset['sha256']:
            raise ValueError(f'checkpoint asset mismatch: {name}')
    reference=ROOT/'docs/evidence/t06-reference'
    manifest=json.loads((reference/'manifest.json').read_text())
    if manifest['revision']!=REV or (manifest['logit_atol'],manifest['intermediate_atol'],manifest['rtol'])!=(.001,.005,.0001):
        raise ValueError('reference identity/gates changed')
    write(output/'identity.json',{'revision':REV,'model_sha256':inventory['assets']['model.safetensors']['sha256'],
        'manifest_sha256':sha(reference/'manifest.json'),'tool_sha256':sha(__file__),
        'graph_sha256':sha(ROOT/'reference/coreml_qwen_graph.py'),'lock_sha256':sha(ROOT/'reference/coreml-lock.txt'),
        'coremltools':ct.__version__,'torch':torch.__version__,'numpy':np.__version__,
        'architecture':platform.machine(),'macos':platform.mac_ver()[0],
        'host':{k:v for k,v in json.loads(subprocess.run(['system_profiler','-json','SPHardwareDataType'],capture_output=True,text=True,check=True).stdout)['SPHardwareDataType'][0].items() if k in ('machine_model','chip_type','physical_memory')},
        'capacity':32,'weights_activations_accumulation':'FP32','state_candidates':['FP16 KV','FP16 leading + FP16 residual KV reconstructed in FP32; logical bytes equal FP32'],
        'gate':{'logit_atol':.001,'intermediate_atol':.005,'rtol':.0001,'all_greedy_ids':'exact'},
        'deployment_floor':'macOS15 / iOS18 stateful ML Program APIs',
        'policy':'CPU_ONLY public APIs; no GPU; fail source numerical gate before expensive checkpoint conversion'})
    audit=audit_state(ct,np,torch); write(output/'state-audit.json',audit)
    if audit[0]['conversion']!='rejected' or audit[1]['conversion']!='accepted':
        raise ValueError('unexpected state capability; inspect before selecting graph')
    start=time.perf_counter_ns(); model=QwenStateExport(directory,torch.float32).eval(); load_ns=time.perf_counter_ns()-start
    control=check_graph(model,manifest,reference,np,torch,torch.float32)
    write(output/'fp32-cache-control.json',control)
    if not control['passed']: raise ValueError('FP32 source graph fails reference; do not attribute failure to cache precision')
    candidate=check_graph(model,manifest,reference,np,torch,torch.float16)
    write(output/'fp16-cache-candidate.json',candidate)
    model.enable_split_state()
    split=check_graph(model,manifest,reference,np,torch,torch.float16)
    write(output/'split-cache-candidate.json',split)
    write(output/'preflight.json',{'load_ns':load_ns,'fp32_source_pass':control['passed'],'fp16_cache_pass':candidate['passed'],
        'split_cache_pass':split['passed'],
        'decision':'split-state source precision gate passed; conversion may proceed' if split['passed'] else 'stateful pretrained path blocked: supported cache representations fail frozen reference gate',
        'pretrained_coreml_conversion':'not run in preflight','gpu_inference':False,
        'owner_review':'pending'})
    print(f"FP32 source={control['passed']}, FP16 cache={candidate['passed']}, split cache={split['passed']}")

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('model',type=Path); parser.add_argument('output',type=Path)
    args=parser.parse_args(); run(args.model.resolve(),args.output.resolve())
