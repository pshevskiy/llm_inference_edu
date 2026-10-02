"""T18 opt-in pretrained split-state Core ML conversion and CPU qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from coreml_qwen import ROOT, REV, sha, write, check_graph

STATES=[f'{kind}{layer}' for layer in range(24) for kind in ('k','v','kl','vl')]

def tree_hash(path):
    h=hashlib.sha256()
    for file in sorted(Path(path).rglob('*')):
        if file.is_file():
            name=str(file.relative_to(path)).encode()
            h.update(len(name).to_bytes(8,'little')); h.update(name)
            h.update(file.stat().st_size.to_bytes(8,'little'))
            with file.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def imports():
    os.environ['HF_HOME']=str(ROOT/'models/.reference-cache')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    import coremltools as ct
    from coreml_tiny import versions
    versions(); torch.set_num_threads(1); torch.set_grad_enabled(False)
    return np,torch,ct

def prerequisites(preflight):
    identity=json.loads((preflight/'identity.json').read_text())
    result=json.loads((preflight/'preflight.json').read_text())
    if identity['revision']!=REV or not result['fp32_source_pass'] or not result['split_cache_pass']:
        raise ValueError('passed source/control preflight required')
    if identity['graph_sha256']!=sha(ROOT/'reference/coreml_qwen_graph.py'):
        raise ValueError('source graph changed after numerical preflight')
    return identity,result

def export(directory,preflight,output):
    np,torch,ct=imports()
    from coreml_qwen_graph import QwenStateExport
    identity,_=prerequisites(preflight)
    inventory=json.loads((ROOT/'fixtures/qwen-inventory.json').read_text())
    for name,asset in inventory['assets'].items():
        if (directory/name).stat().st_size!=asset['bytes'] or sha(directory/name)!=asset['sha256']:
            raise ValueError('verified checkpoint changed')
    model=QwenStateExport(directory).eval(); model.enable_split_state(); model.tuple_outputs=True
    model.reset()
    names=list(json.loads((preflight/'fp32-cache-control.json').read_text())['positions'][0]['arrays'])
    traced=torch.jit.trace(model,(torch.tensor([9707],dtype=torch.int32),torch.tensor([0],dtype=torch.int32)),check_trace=False)
    class TracedCheck:
        def reset(self,dtype):
            for name in STATES: getattr(traced,name).zero_()
        def __call__(self,*args): return dict(zip(names,traced(*args)))
    reference=ROOT/'docs/evidence/t06-reference'
    manifest=json.loads((reference/'manifest.json').read_text())
    if sha(reference/'manifest.json')!=identity['manifest_sha256']:
        raise ValueError('reference manifest changed')
    result=check_graph(TracedCheck(),manifest,reference,np,torch,torch.float16)
    write(output/'traced-check.json',result)
    if not result['passed']: raise ValueError('traced source fails frozen numerical gate')
    for name in STATES: getattr(traced,name).zero_()
    start=time.perf_counter_ns()
    converted=ct.convert(traced,convert_to='mlprogram',minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,compute_units=ct.ComputeUnit.CPU_ONLY,skip_model_load=True,
        inputs=[ct.TensorType(name='token_ids',shape=(1,),dtype=np.int32),ct.TensorType(name='position',shape=(1,),dtype=np.int32)],
        outputs=[ct.TensorType(name=name.replace('.','_'),dtype=np.float32) for name in names],
        states=[ct.StateType(name=name,wrapped_type=ct.TensorType(shape=(2,32,64),dtype=np.float16)) for name in STATES])
    conversion_ns=time.perf_counter_ns()-start
    start=time.perf_counter_ns(); converted.save(str(output/'qwen.mlpackage')); save_ns=time.perf_counter_ns()-start
    spec=converted.get_spec()
    state_specs=[{'name':s.name,'shape':list(s.type.stateType.arrayType.shape),'dtype':s.type.stateType.arrayType.dataType} for s in spec.description.state]
    write(output/'export.json',{'conversion_ns':conversion_ns,'save_ns':save_ns,'output_names':names,
        'states':state_specs,'state_logical_bytes':24*4*2*32*64*2,
        'artifact_sha256':tree_hash(output/'qwen.mlpackage'),
        'artifact_file_bytes':sum(p.stat().st_size for p in (output/'qwen.mlpackage').rglob('*') if p.is_file()),
        'precision':'FP32 compute/weights/outputs; persistent FP16 high+residual pairs',
        'policy':'CPU_ONLY; no GPU; compilation deferred'})

def compile_model(output):
    _,_,ct=imports()
    saved=json.loads((output/'export.json').read_text())
    if tree_hash(output/'qwen.mlpackage')!=saved['artifact_sha256']: raise ValueError('export changed')
    start=time.perf_counter_ns()
    ct.utils.compile_model(str(output/'qwen.mlpackage'),str(output/'qwen.mlmodelc'))
    elapsed=time.perf_counter_ns()-start
    write(output/'compile.json',{'compile_ns':elapsed,'artifact_sha256':tree_hash(output/'qwen.mlmodelc'),
        'memory_scope':'child process only; separate compiler services not attributed'})

class CoreMLCheckSession:
    """Development qualification wrapper; native state is never mirrored in Rust."""
    def __init__(self,model,np):
        self.model,self.np=model,np; self.reset()
    def reset(self):
        self.state=self.model.make_state(); self.processed=0; self.valid=True
    def step(self,token,position=None):
        # Admission cannot mutate the state. Position is owned by this wrapper.
        if not self.valid: raise ValueError('invalid state; reset required')
        if type(token) is not int or not 0<=token<151936: raise ValueError('invalid token')
        if self.processed>=32: raise ValueError('capacity full')
        if position is not None and position!=self.processed: raise ValueError('position mismatch')
        try:
            result=self.model.predict({'token_ids':self.np.array([token],dtype=self.np.int32),
                'position':self.np.array([self.processed],dtype=self.np.int32)},state=self.state)
            if not all(self.np.isfinite(v).all() for v in result.values()): raise ValueError('nonfinite result')
        except Exception:
            self.valid=False; self.processed=0; raise
        self.processed+=1
        return result

def check(preflight,output):
    np,_,ct=imports()
    identity,_=prerequisites(preflight)
    saved=json.loads((output/'compile.json').read_text())
    if tree_hash(output/'qwen.mlmodelc')!=saved['artifact_sha256']: raise ValueError('compiled artifact changed')
    reference=ROOT/'docs/evidence/t06-reference'
    if sha(reference/'manifest.json')!=identity['manifest_sha256']: raise ValueError('reference changed')
    manifest=json.loads((reference/'manifest.json').read_text())
    expected=[]
    for case in manifest['cases']:
        values={}
        for a in case['arrays']:
            if sha(reference/a['file'])!=a['sha256']: raise ValueError('reference array changed')
            values[a['name']]=np.fromfile(reference/a['file'],dtype='<f4').reshape(len(case['ids']),-1)
        expected.append(values)
    start=time.perf_counter_ns()
    model=ct.models.CompiledMLModel(str(output/'qwen.mlmodelc'),compute_units=ct.ComputeUnit.CPU_ONLY)
    load_ns=time.perf_counter_ns()-start
    sessions=[CoreMLCheckSession(model,np) for _ in manifest['cases']]
    results=[]; passed=True
    # Interleave independent native states against full-prefix references.
    for position in range(3):
        for case,values,session in zip(manifest['cases'],expected,sessions):
            if position>=len(case['ids']): continue
            before=time.perf_counter_ns(); actual=session.step(case['ids'][position]); predict_ns=time.perf_counter_ns()-before
            arrays={}
            for name,b in values.items():
                a=np.asarray(actual[name.replace('.','_')]).reshape(-1); b=b[position]
                if a.shape!=b.shape or not np.isfinite(a).all(): raise ValueError('invalid comparison shape/values')
                error=np.abs(a.astype(np.float64)-b.astype(np.float64))
                atol=manifest['logit_atol'] if name=='logits' else manifest['intermediate_atol']
                fraction=float((error/(atol+manifest['rtol']*np.abs(b))).max()); passed &= fraction<=1
                arrays[name]={'max_absolute_error':float(error.max()),'max_budget_fraction':fraction,'passed':fraction<=1}
            greedy=int(actual['logits'].argmax())==int(values['logits'][position].argmax()); passed &= greedy
            results.append({'text':case['text'],'position':position,'predict_ns':predict_ns,'arrays':arrays,'greedy_exact':greedy})
    # Bad admissions leave the complete native state unchanged.
    session=sessions[0]
    digest=lambda:hashlib.sha256(b''.join(session.state.read_state(name).tobytes() for name in STATES)).hexdigest()
    old=digest(); processed=session.processed
    for token,position in [(151936,None),(0,processed+1)]:
        try: session.step(token,position); raise AssertionError('invalid admission accepted')
        except ValueError: pass
        assert session.processed==processed and session.valid and digest()==old
    session.reset(); repeated=session.step(manifest['cases'][0]['ids'][0])
    # Fresh-state replay must reproduce the corresponding first-step logit row.
    row=expected[0]['logits'][0]; error=np.abs(repeated['logits'].reshape(-1)-row)
    reset_fraction=float((error/(manifest['logit_atol']+manifest['rtol']*np.abs(row))).max()); passed &= reset_fraction<=1
    write(output/'cpu-check.json',{'passed':bool(passed),'positions':results,'load_compiled_ns':load_ns,
        'interleaved_independent_native_states':True,'invalid_admission_preserved_all_state':True,
        'reset_reference_budget_fraction':reset_fraction,'state_logical_bytes_per_session':786432,
        'precision':'FP32 weights/activations/accumulation; FP16 high+residual cache pairs',
        'policy':'CPU_ONLY','gpu_inference':False,'ane_placement':'not investigated',
        'scope':'eight short teacher-forced positions at capacity32; no long-context/mobile/performance qualification'})
    if not passed: raise ValueError('converted checkpoint fails frozen reference gate')

def run(directory,preflight,output):
    prerequisites(preflight)
    if platform.system()!='Darwin': raise ValueError('actual macOS required')
    output.mkdir(parents=True,exist_ok=False)
    # Fixed verified checkpoint, conservative conversion planning rather than RSS.
    expanded_weights=1_976_131_072
    planned=3*expanded_weights+1024**3
    budget=8*1024**3
    free=shutil.disk_usage(output).free
    if planned>budget or free<planned:
        raise ValueError('conversion planning/disk admission failed')
    write(output/'identity.json',{'preflight':str(preflight),'preflight_identity_sha256':sha(preflight/'identity.json'),
        'preflight_result_sha256':sha(preflight/'preflight.json'),'tool_sha256':sha(__file__),
        'source_identity':json.loads((preflight/'identity.json').read_text()),
        'policy':'one shared compiled checkpoint; native persistent state; CPU_ONLY',
        'planning':'fixed 1.976GB FP32 weights; tiny C32 state; conversion may hold multiple weight views; no model download',
        'planning_bytes':planned,'planning_budget_bytes':budget,'disk_free_before_bytes':free,
        'resource_scope':'phase child peaks, not separate compiler services; correctness run, no sustained benchmark'})
    for stage in ('export','compile','check'):
        with (output/f'{stage}-process.txt').open('x') as log:
            status=subprocess.run(['/usr/bin/time','-l',sys.executable,str(Path(__file__).resolve()),stage,str(directory),str(preflight),str(output)],stdout=log,stderr=subprocess.STDOUT).returncode
        if status:
            write(output/'failure.json',{'stage':stage,'exit_code':status,'passed':False})
            raise RuntimeError(f'{stage} failed; preserve {output}/{stage}-process.txt')
    write(output/'completion.json',{'passed':True,'owner_review':'pending'})
    print(f'Pretrained split-state Core ML CPU qualification passed: {output}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['run','export','compile','check'])
    parser.add_argument('model',type=Path); parser.add_argument('preflight',type=Path); parser.add_argument('output',type=Path)
    a=parser.parse_args(); directory,preflight,output=a.model.resolve(),a.preflight.resolve(),a.output.resolve()
    if a.stage=='run': run(directory,preflight,output)
    elif a.stage=='export': export(directory,preflight,output)
    elif a.stage=='compile': compile_model(output)
    else: check(preflight,output)
