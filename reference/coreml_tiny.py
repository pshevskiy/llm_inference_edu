"""Opt-in T17 export/compile/CPU correctness experiment; never production math."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
CASES = [[1], [1,5], [1,5,7], [1,5,7,2]]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tree_hash(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob('*')):
        if file.is_file():
            name = str(file.relative_to(path)).encode()
            data = file.read_bytes()
            digest.update(len(name).to_bytes(8,'little')); digest.update(name)
            digest.update(len(data).to_bytes(8,'little')); digest.update(data)
    return digest.hexdigest()

def write_new(path, value):
    with path.open('x') as output:
        json.dump(value,output,indent=2,allow_nan=False); output.write('\n')

def versions():
    result = {p:importlib.metadata.version(p) for p in ('numpy','torch','coremltools')}
    if result != {'numpy':'2.3.2','torch':'2.6.0','coremltools':'9.0'}:
        raise RuntimeError(f'use pinned reference/coreml-lock.txt: {result}')
    return result

def compare(actual, expected):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'invalid numerical comparison {a.shape}/{b.shape}')
    error = np.abs(a.astype(np.float64)-b.astype(np.float64))
    budget = 1e-5 + 1e-4*np.abs(b.astype(np.float64))
    fraction = float((error/budget).max())
    if fraction > 1:
        raise ValueError(f'original tiny gate failed: max_abs={error.max()}, budget_fraction={fraction}')
    return {'elements':a.size,'max_absolute_error':float(error.max()),'max_budget_fraction':fraction}

def heavy_imports():
    # Conversion may discover Transformers; keep any library cache isolated/offline.
    os.environ['HF_HOME'] = str(ROOT/'models/.reference-cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    import coremltools as ct
    torch.set_num_threads(1)
    return np, torch, ct

def stage_export(output):
    np, torch, ct = heavy_imports()
    from coreml_tiny_graph import TinyExport
    from decoder import outputs_for
    model = TinyExport().eval()
    names = list(outputs_for(CASES[2]))
    comparisons = []
    with torch.inference_mode():
        for ids in CASES:
            reference = outputs_for(ids)
            result = model(torch.tensor(ids,dtype=torch.int32))
            comparisons.append({'ids':ids,'arrays':{name:compare(value.numpy(),reference[name])
                                for name,value in zip(names,result)}})
        traced = torch.jit.trace(model,torch.tensor(CASES[2],dtype=torch.int32),check_inputs=[
            (torch.tensor(ids,dtype=torch.int32),) for ids in CASES])
    before = time.perf_counter_ns()
    converted = ct.convert(traced, convert_to='mlprogram',
        inputs=[ct.TensorType(name='token_ids',dtype=np.int32,
            shape=ct.EnumeratedShapes(shapes=[(len(ids),) for ids in CASES],default=(3,)))],
        outputs=[ct.TensorType(name=name.replace('.','_'),dtype=np.float32) for name in names],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True)
    conversion_ns = time.perf_counter_ns()-before
    before = time.perf_counter_ns()
    converted.save(str(output/'tiny.mlpackage'))
    save_ns = time.perf_counter_ns()-before
    write_new(output/'export.json',{'conversion_ns':conversion_ns,'save_ns':save_ns,
        'torch_vs_independent_numpy':comparisons,'output_names':names,
        'artifact_sha256':tree_hash(output/'tiny.mlpackage'),
        'artifact_file_bytes':sum(p.stat().st_size for p in (output/'tiny.mlpackage').rglob('*') if p.is_file()),
        'precision':'FP32 typed graph/inputs except token IDs int32; no FP16 conversion',
        'compute_units':'CPU_ONLY','minimum_deployment':'macOS13 / iOS16 equivalent',
        'compilation':'deferred; skip_model_load=True'})

def stage_compile(output):
    _, _, ct = heavy_imports()
    export = json.loads((output/'export.json').read_text())
    if tree_hash(output/'tiny.mlpackage') != export['artifact_sha256']:
        raise ValueError('export artifact changed')
    before = time.perf_counter_ns()
    compiled = ct.utils.compile_model(str(output/'tiny.mlpackage'),str(output/'tiny.mlmodelc'))
    elapsed = time.perf_counter_ns()-before
    write_new(output/'compile.json',{'compile_ns':elapsed,'compiled_path':str(Path(compiled).relative_to(output)),
        'compiled_sha256':tree_hash(output/'tiny.mlmodelc'),
        'compiler':'public coremltools.utils.compile_model, system Core ML compiler',
        'memory_scope':'child process peak including Python imports; separate compiler service memory not attributed'})

def stage_check(output):
    np, _, ct = heavy_imports()
    from decoder import outputs_for
    compilation = json.loads((output/'compile.json').read_text())
    if tree_hash(output/'tiny.mlmodelc') != compilation['compiled_sha256']:
        raise ValueError('compiled artifact changed')
    before = time.perf_counter_ns()
    model = ct.models.CompiledMLModel(str(output/'tiny.mlmodelc'),compute_units=ct.ComputeUnit.CPU_ONLY)
    load_ns = time.perf_counter_ns()-before
    results = []
    actual_cases = []
    for ids in CASES:
        reference = outputs_for(ids)
        before = time.perf_counter_ns()
        actual = model.predict({'token_ids':np.array(ids,dtype=np.int32)})
        predict_ns = time.perf_counter_ns()-before
        arrays = {name:compare(actual[name.replace('.','_')],value) for name,value in reference.items()}
        for position in range(len(ids)):
            if int(np.argmax(actual['logits'][position])) != int(np.argmax(reference['logits'][position])):
                raise ValueError('teacher-forced greedy token mismatch')
        results.append({'ids':ids,'predict_ns':predict_ns,'arrays':arrays,'greedy_exact':True})
        actual_cases.append(actual)
    # All outputs are token-major. Causal prefixes must survive a future token.
    prefix = actual_cases[2]['logits']
    future = actual_cases[3]['logits'][:3]
    causal = compare(future,prefix)
    write_new(output/'cpu-check.json',{'load_compiled_ns':load_ns,'cases':results,
        'causal_prefix_check':causal,'compute_units':'CPU_ONLY','gpu_inference':False,
        'ane_execution':'not investigated or claimed','state':'stateless full-prefix tiny experiment',
        'timing_scope':'one prediction per shape; correctness diagnostics, not repeated benchmark',
        'atol':1e-5,'rtol':1e-4,'passed':True})

def run(output):
    if platform.system() != 'Darwin':
        raise RuntimeError('Core ML compilation/prediction requires actual macOS; not run elsewhere')
    output.mkdir(parents=True,exist_ok=False)
    hardware = subprocess.run(['system_profiler','-json','SPHardwareDataType'],capture_output=True,text=True,check=True)
    info = json.loads(hardware.stdout)['SPHardwareDataType'][0]
    write_new(output/'identity.json',{'versions':versions(),'python':platform.python_version(),
        'host':{k:info.get(k) for k in ('machine_model','chip_type','physical_memory')},
        'macos':platform.mac_ver()[0],'architecture':platform.machine(),'cases':CASES,
        'fixture_sha256':sha(ROOT/'fixtures/tiny-model.json'),
        'reference_fixture_sha256':sha(ROOT/'fixtures/t04-expected.json'),
        'tool_sha256':sha(__file__),'export_graph_sha256':sha(ROOT/'reference/coreml_tiny_graph.py'),
        'oracle_sha256':sha(ROOT/'reference/decoder.py'),
        'oracle_rope_sha256':sha(ROOT/'reference/attention.py'),
        'lock_sha256':sha(ROOT/'reference/coreml-lock.txt'),
        'policy':'CPU_ONLY everywhere; existing tiny numerical gates; no pretrained weights',
        'memory':'three separate process peaks in phase logs; includes imports, excludes separately attributed compiler services'})
    for stage in ('export','compile','check'):
        with (output/f'{stage}-process.txt').open('x') as log:
            status = subprocess.run(['/usr/bin/time','-l',sys.executable,str(Path(__file__).resolve()),
                stage,str(output)],stdout=log,stderr=subprocess.STDOUT).returncode
        if status:
            write_new(output/'failure.json',{'stage':stage,'exit_code':status,'passed':False})
            raise RuntimeError(f'{stage} failed; inspect {output}/{stage}-process.txt; preserve evidence')
    write_new(output/'completion.json',{'passed':True,'owner_review':'pending'})
    print(f'Tiny Core ML CPU qualification passed: {output}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['run','export','compile','check'])
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    versions()
    if args.stage == 'run':
        run(output)
    else:
        {'export':stage_export,'compile':stage_compile,'check':stage_check}[args.stage](output)

if __name__ == '__main__':
    main()
