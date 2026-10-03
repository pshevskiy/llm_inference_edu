"""T19 one-shot policy correctness and anticipated compute-plan metadata, not placement proof."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from coreml_qwen import ROOT, REV, sha, write
from coreml_qwen_export import imports, tree_hash, CoreMLCheckSession

POLICIES=('CPU_ONLY','CPU_AND_NE')

def current_host():
    info=json.loads(subprocess.run(['system_profiler','-json','SPHardwareDataType'],capture_output=True,text=True,check=True).stdout)['SPHardwareDataType'][0]
    return {**{k:info.get(k) for k in ('machine_model','chip_type','physical_memory')},
        'os_version':subprocess.run(['sw_vers','-productVersion'],capture_output=True,text=True,check=True).stdout.strip(),
        'os_build':subprocess.run(['sw_vers','-buildVersion'],capture_output=True,text=True,check=True).stdout.strip()}

def plan_metadata(ct,path,policy):
    from coremltools.models.compute_plan import MLComputePlan
    try:
        before=time.perf_counter_ns(); plan=MLComputePlan.load_from_path(str(path),compute_units=policy)
        load_ns=time.perf_counter_ns()-before
        program=plan.model_structure.program
        if program is None: raise ValueError('not an ML Program')
        records=[]
        def block(values,prefix):
            for index,op in enumerate(values.operations):
                name=f'{prefix}/{index}'
                usage=plan.get_compute_device_usage_for_mlprogram_operation(op)
                cost=plan.get_estimated_cost_for_mlprogram_operation(op)
                preferred=type(usage.preferred_compute_device).__name__ if usage else None
                if preferred=='MLGPUComputeDevice': raise ValueError('GPU preferred under excluded policy')
                records.append({'path':name,'operator':op.operator_name,'outputs':[v.name for v in op.outputs],
                    'anticipated_preferred_device':preferred,
                    'supported_devices':[type(v).__name__ for v in usage.supported_compute_devices] if usage else None,
                    'estimated_relative_cost':cost.weight if cost else None})
                for i,child in enumerate(op.blocks): block(child,f'{name}/block{i}')
        for name,function in program.functions.items(): block(function.block,name)
        return {'available':True,'load_plan_ns':load_ns,'operations':records,
            'anticipated_counts':dict(Counter(r['anticipated_preferred_device'] or 'unknown' for r in records)),
            'operator_counts':dict(Counter(r['operator'] for r in records)),
            'scope':'anticipated supported/preferred devices and estimated costs; not observed execution or ANE placement proof'}
    except Exception as error:
        return {'available':False,'reason':str(error),'scope':'not actual execution'}

def native(artifact,policy_name,output):
    np,_,ct=imports(); policy=getattr(ct.ComputeUnit,policy_name)
    if policy_name not in POLICIES: raise ValueError('GPU policies forbidden')
    output.mkdir(parents=True,exist_ok=False)
    saved=json.loads((artifact/'compile.json').read_text())
    if tree_hash(artifact/'qwen.mlmodelc')!=saved['artifact_sha256']: raise ValueError('compiled artifact changed')
    model_identity=json.loads((artifact/'identity.json').read_text())
    reference=ROOT/'docs/evidence/t06-reference'; manifest_bytes=(reference/'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=model_identity['source_identity']['manifest_sha256']:
        raise ValueError('reference manifest changed')
    manifest=json.loads(manifest_bytes)
    if manifest['revision']!=REV: raise ValueError('wrong reference checkpoint')
    write(output/'identity.json',{'policy':policy_name,'gpu_inference':False,'artifact_sha256':saved['artifact_sha256'],
        'reference_manifest_sha256':sha(reference/'manifest.json'),'tool_sha256':sha(__file__),
        'session_wrapper_sha256':sha(ROOT/'reference/coreml_qwen_export.py'),
        't18_identity_sha256':sha(artifact/'identity.json'),'host':current_host(),
        'artifact_source_host':model_identity['source_identity']['host'],
        'precision':'FP32 weights/compute, paired FP16 cache; same artifact as T18',
        'scope':'one diagnostic prediction per position,49outputs; no repeated benchmark or sampling/text'})
    write(output/'plan.json',plan_metadata(ct,artifact/'qwen.mlmodelc',policy))
    before=time.perf_counter_ns(); model=ct.models.CompiledMLModel(str(artifact/'qwen.mlmodelc'),compute_units=policy)
    load_ns=time.perf_counter_ns()-before
    records=[]; passed=True
    for case_index,case in enumerate(manifest['cases']):
        expected={}
        for entry in case['arrays']:
            file=reference/entry['file']
            if sha(file)!=entry['sha256']: raise ValueError('reference changed')
            expected[entry['name']]=np.fromfile(file,dtype='<f4').reshape(len(case['ids']),-1)
        session=CoreMLCheckSession(model,np)
        for position,token in enumerate(case['ids']):
            before=time.perf_counter_ns(); actual=session.step(token); predict_ns=time.perf_counter_ns()-before
            arrays={}
            for name,rows in expected.items():
                a=np.asarray(actual[name.replace('.','_')]).reshape(-1); b=rows[position]
                if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('invalid shape/value')
                error=np.abs(a.astype(np.float64)-b.astype(np.float64))
                atol=manifest['logit_atol'] if name=='logits' else manifest['intermediate_atol']
                fraction=float((error/(atol+manifest['rtol']*np.abs(b))).max()); passed &= fraction<=1
                arrays[name]={'max_absolute_error':float(error.max()),'max_budget_fraction':fraction}
            row=np.asarray(actual['logits'],dtype='<f4').reshape(-1)
            greedy=int(row.argmax())==int(expected['logits'][position].argmax()); passed &= greedy
            filename=f'{case_index}-{position}.bin'; payload=row.tobytes(); (output/filename).write_bytes(payload)
            records.append({'case':case_index,'ids':case['ids'],'position':position,'file':filename,'sha256':hashlib.sha256(payload).hexdigest(),
                'predict_ns':predict_ns,'arrays':arrays,'greedy_exact':greedy,'processed':session.processed})
    write(output/'results.json',{'policy':policy_name,'load_compiled_ns':load_ns,'records':records,'passed':bool(passed),
        'scope':'one diagnostic prediction/position; fixed case order; compute-plan loading precedes model loading; not cold qualification'})
    if not passed: raise ValueError('policy fails frozen numerical gate')

def compare_runs(output):
    import numpy as np
    reference=ROOT/'docs/evidence/t06-reference'; manifest=json.loads((reference/'manifest.json').read_text())
    runs={name:json.loads((output/name/'results.json').read_text()) for name in ('rust',*POLICIES)}
    identities={name:json.loads((output/name/'identity.json').read_text()) for name in ('rust',*POLICIES)}
    expected_hash=sha(reference/'manifest.json')
    if identities['rust']['manifest_sha256']!=expected_hash or any(identities[p]['reference_manifest_sha256']!=expected_hash for p in POLICIES):
        raise ValueError('mismatched reference identity')
    reference_logits={}
    for index,case in enumerate(manifest['cases']):
        entry=next(a for a in case['arrays'] if a['name']=='logits')
        if sha(reference/entry['file'])!=entry['sha256']: raise ValueError('reference logits changed')
        reference_logits[index]=np.fromfile(reference/entry['file'],dtype='<f4').reshape(len(case['ids']),-1)
    rows={}; comparison=[]; passed=True
    for name,run in runs.items():
        if not run['passed']: raise ValueError('incomplete/failed run')
        seen=set()
        for row in run['records']:
            key=(row['case'],row['position']); case=manifest['cases'][row['case']]
            if key in seen or row['ids']!=case['ids'] or row['processed']!=row['position']+1: raise ValueError('prefix identity mismatch')
            seen.add(key); file=output/name/row['file']
            if sha(file)!=row['sha256']: raise ValueError('record changed')
            a=np.fromfile(file,dtype='<f4'); b=reference_logits[row['case']][row['position']]
            if a.shape!=b.shape or not np.isfinite(a).all(): raise ValueError('invalid logits')
            fraction=float((np.abs(a.astype(np.float64)-b)/(manifest['logit_atol']+manifest['rtol']*np.abs(b))).max())
            greedy=int(a.argmax())==int(b.argmax()); passed &= fraction<=1 and greedy
            rows[(name,key)]=a
            comparison.append({'path':name,'case':key[0],'position':key[1],'reference_budget_fraction':fraction,'greedy_exact':greedy})
        if seen!={(i,p) for i,c in enumerate(manifest['cases']) for p in range(len(c['ids']))}: raise ValueError('missing/extra position')
    direct=[]
    for key in sorted(k for name,k in rows if name=='rust'):
        for left,right in [('rust','CPU_ONLY'),('rust','CPU_AND_NE'),('CPU_ONLY','CPU_AND_NE')]:
            a,b=rows[(left,key)],rows[(right,key)]
            error=np.abs(a.astype(np.float64)-b.astype(np.float64))
            direct.append({'left':left,'right':right,'case':key[0],'position':key[1],
                'max_absolute_logit_difference':float(error.max()),'greedy_exact':int(a.argmax())==int(b.argmax())})
    write(output/'comparison.json',{'all_reference_gates_passed':bool(passed),'reference_checks':comparison,'direct_pairs':direct,
        'precision_difference':'Rust FP32 KV versus approximate paired half state; declared, not silently matched',
        'timing_comparability':'Rust buffered logits-only step versus Core ML49-output diagnostic predict; no shared sampler/text/TTFT contract',
        'actual_ane_placement':'blocked: requires measured report/trace; compute-plan metadata is insufficient',
        'repeated_performance':'not run; sustained hardware opt-in pending','energy_thermal':'not qualified',
        'decision':'CPU engine retained; no ANE acceleration claimed or selected','owner_review':'pending'})
    if not passed: raise ValueError('one-shot path fails reference gate')

def run(artifact,output):
    output.mkdir(parents=True,exist_ok=False)
    trace=subprocess.run(['xcrun','--find','xctrace'],capture_output=True,text=True)
    selection=subprocess.run(['xcode-select','-p'],capture_output=True,text=True)
    write(output/'identity.json',{'tool_sha256':sha(__file__),'artifact_identity_sha256':sha(artifact/'identity.json'),
        'host':current_host(),
        'policies':POLICIES,'gpu_inference':False,'xctrace_available':trace.returncode==0,
        'xctrace_result':trace.stdout.strip() if trace.returncode==0 else trace.stderr.strip(),
        'developer_directory':selection.stdout.strip(),'scope':'short correctness and anticipated-plan audit only',
        'performance_opt_in':'sustained owner approval pending'})
    # Each native policy and Rust execution has its own process/model allocation.
    for name in ('rust',*POLICIES):
        if name=='rust':
            source=json.loads((artifact/'identity.json').read_text())['source_identity']
            directory=ROOT/'models/qwen2.5-0.5b-instruct'/source['revision']
            command=[str(ROOT/'target/release/examples/policy_trace'),str(directory),str(ROOT/'docs/evidence/t06-reference/manifest.json'),str(output/name)]
        else:
            command=[sys.executable,str(Path(__file__).resolve()),name,str(artifact),str(output/name)]
        with (output/f'{name}-process.txt').open('x') as log:
            status=subprocess.run(['/usr/bin/time','-l',*command],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT).returncode
        if status:
            write(output/'failure.json',{'path':name,'exit_code':status}); raise RuntimeError(f'{name} failed; preserve evidence')
    compare_runs(output)
    write(output/'completion.json',{'one_shot_correctness':'passed','T19':'partial; measured placement/repeated performance blocked'})
    print('One-shot policy correctness complete; measured ANE placement/performance remains blocked')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=['run',*POLICIES])
    parser.add_argument('artifact',type=Path); parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.mode=='run': run(args.artifact.resolve(),args.output.resolve())
    else: native(args.artifact.resolve(),args.mode,args.output.resolve())
