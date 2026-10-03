"""T19 opt-in development comparison; never a production generation backend.

One model per child process. Fixed public workload, unchanged frozen gates,
logits-only native outputs, native cache, equivalent greedy/UTF-8 contracts.
Python/Core ML bridge and Python policy overhead remain explicitly included.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from coreml_qwen import ROOT, REV, sha, write
from coreml_qwen_export import imports, tree_hash, CoreMLCheckSession
from coreml_policy import POLICIES, current_host, plan_metadata

MODEL = ROOT / 'models/qwen2.5-0.5b-instruct' / REV
WORKLOAD = ROOT / 'benchmarks/qwen-short.json'
REFERENCE = ROOT / 'docs/evidence/t06-reference'
STOPS = (151645, 151643)


def validate_workload(w):
    if set(w) != {'name', 'prompt', 'fixed_output', 'continuation', 'capacity', 'warmups', 'repetitions'}:
        raise ValueError('unknown/missing workload fields')
    for key in ('prompt', 'fixed_output', 'continuation'):
        if not isinstance(w[key], list) or not w[key] or any(type(t) is not int or not 0 <= t < 151936 for t in w[key]):
            raise ValueError('invalid token IDs')
    if any(type(w[k]) is not int for k in ('capacity', 'warmups', 'repetitions')):
        raise ValueError('integer bounds required')
    if w['capacity'] != 32 or not 0 <= w['warmups'] <= 10 or not 1 <= w['repetitions'] <= 100:
        raise ValueError('unsupported capacity/repetition bounds')
    if sum(len(w[k]) for k in ('prompt', 'fixed_output', 'continuation')) + 1 > w['capacity']:
        raise ValueError('whole workload exceeds capacity')


def prepare(source, output):
    _, _, ct = imports()
    if tree_hash(source / 'qwen.mlmodelc') != json.loads((source / 'compile.json').read_text())['artifact_sha256']:
        raise ValueError('T18 compiled artifact changed')
    if tree_hash(source / 'qwen.mlpackage') != json.loads((source / 'export.json').read_text())['artifact_sha256']:
        raise ValueError('T18 source artifact changed')
    output.mkdir(parents=True, exist_ok=False)
    model = ct.models.MLModel(str(source / 'qwen.mlpackage'), skip_model_load=True, compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    original = copy.deepcopy(spec)
    logits = copy.deepcopy(next(o for o in spec.description.output if o.name == 'logits'))
    del spec.description.output[:]
    spec.description.output.append(logits)
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            if 'logits' not in block.outputs:
                raise ValueError('unexpected program interface')
            del block.outputs[:]
            block.outputs.append('logits')
    # Restoring only the changed output lists must recover the identical spec.
    restored = copy.deepcopy(spec)
    del restored.description.output[:]
    restored.description.output.extend(original.description.output)
    for name, function in restored.mlProgram.functions.items():
        for key, block in function.block_specializations.items():
            del block.outputs[:]
            block.outputs.extend(original.mlProgram.functions[name].block_specializations[key].outputs)
    if restored != original:
        raise ValueError('unexpected arithmetic/state/spec change')
    before = time.perf_counter_ns()
    lean = ct.models.MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True, compute_units=ct.ComputeUnit.CPU_ONLY)
    lean.save(str(output / 'qwen.mlpackage'))
    save_ns = time.perf_counter_ns() - before
    if tree_hash(output / 'qwen.mlpackage/Data/com.apple.CoreML/weights') != tree_hash(Path(model.weights_dir)):
        raise ValueError('weights changed')
    before = time.perf_counter_ns()
    ct.utils.compile_model(str(output / 'qwen.mlpackage'), str(output / 'qwen.mlmodelc'))
    compile_ns = time.perf_counter_ns() - before
    write(output / 'identity.json', {'host': current_host(), 'source_identity': json.loads((source / 'identity.json').read_text()),
        't18_identity_sha256': sha(source / 'identity.json'), 't18_package_sha256': tree_hash(source / 'qwen.mlpackage'),
        'tool_sha256': sha(__file__), 'lock_sha256': sha(ROOT / 'reference/coreml-lock.txt'),
        'change': 'only public output lists reduced from49 to logits; restored spec equality and identical weight bytes checked',
        'precision': 'unchanged FP32 weights/compute, paired FP16 persistent state', 'capacity': 32})
    write(output / 'compile.json', {'save_ns': save_ns, 'compile_ns': compile_ns,
        'artifact_sha256': tree_hash(output / 'qwen.mlmodelc'), 'package_sha256': tree_hash(output / 'qwen.mlpackage')})


def oracle(output):
    np, torch, _ = imports()
    from transformers import AutoModelForCausalLM
    w = json.loads(WORKLOAD.read_text())
    validate_workload(w)
    inventory = json.loads((ROOT / 'fixtures/qwen-inventory.json').read_text())
    for name, asset in inventory['assets'].items():
        if (MODEL / name).stat().st_size != asset['bytes'] or sha(MODEL / name) != asset['sha256']:
            raise ValueError('checkpoint asset changed')
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((REFERENCE / 'manifest.json').read_text())
    if old['revision'] != REV:
        raise ValueError('wrong frozen reference')
    actual_versions = {name: importlib.metadata.version(name) for name in old['versions']}
    if actual_versions != old['versions']:
        raise ValueError(f'use pinned numerical reference versions: {actual_versions}')
    manifest = {k: old[k] for k in ('revision', 'logit_atol', 'intermediate_atol', 'rtol', 'versions')}
    manifest['versions'] = actual_versions
    manifest.update({'cases': [], 'expected_samples': {}, 'device': 'CPU', 'dtype': 'FP32', 'threads': 1,
        'source_manifest_sha256': sha(REFERENCE / 'manifest.json'), 'workload_sha256': sha(WORKLOAD),
        'tool_sha256': sha(__file__), 'host': current_host()})

    def save_case(text, ids, values):
        index = len(manifest['cases'])
        if np.asarray(values).size != len(ids) * 151936 or not np.isfinite(values).all():
            raise ValueError('invalid independent reference shape/values')
        payload = np.asarray(values, dtype='<f4').tobytes()
        filename = f'{index}-logits.bin'
        (output / filename).write_bytes(payload)
        manifest['cases'].append({'text': text, 'ids': ids, 'arrays': [{'name': 'logits', 'file': filename,
            'sha256': hashlib.sha256(payload).hexdigest(), 'elements': len(payload) // 4}]})

    for case in old['cases']:
        entry = next(a for a in case['arrays'] if a['name'] == 'logits')
        if sha(REFERENCE / entry['file']) != entry['sha256']:
            raise ValueError('original logits changed')
        save_case(case['text'], case['ids'], np.fromfile(REFERENCE / entry['file'], dtype='<f4'))
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, local_files_only=True,
        trust_remote_code=False, attn_implementation='eager', low_cpu_mem_usage=False).eval().to('cpu')

    def logits(ids):
        return model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].numpy()

    for mode in ('TeacherForced', 'Greedy'):
        prefix = list(w['prompt'])
        emitted = []
        for forced in w['fixed_output']:
            if emitted:
                prefix.append(emitted[-1])
            token = forced if mode == 'TeacherForced' else int(logits(prefix)[-1].argmax())
            emitted.append(token)
            if mode == 'Greedy' and token in STOPS:
                break
        before = len(prefix)
        prefix.extend([emitted[-1], *w['continuation']])
        rows = logits(prefix)
        save_case(mode, prefix, rows)
        manifest['expected_samples'][mode] = {'emitted': emitted, 'stopped_by_eos': mode == 'Greedy' and emitted[-1] in STOPS,
            'processed_before_continuation': before, 'pending_emitted_tokens': 1,
            'processed_after_continuation': len(prefix), 'continuation_emitted_token': int(rows[-1].argmax()),
            'pending_after_continuation': 1}
    write(output / 'manifest.json', manifest)


class TextStream:
    """Development equivalent of tokenizers0.21.4 step_decode_stream.

    The pinned Python tokenizers0.20.3 lacks that entry point. This retains
    UTF-8 pending bytes, checks prefix stability and hides special tokens.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special = {int(k) for k, v in tokenizer.get_added_tokens_decoder().items() if v.special}
        self.ids, self.prefix, self.index = [], '', 0

    def push(self, token):
        if self.tokenizer.id_to_token(token) is None:
            raise ValueError('invalid text token')
        if token in self.special:
            return None
        self.ids.append(token)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if len(text) > len(self.prefix) and not text.endswith('\ufffd'):
            if not text.startswith(self.prefix):
                raise ValueError('invalid decode prefix')
            new = text[len(self.prefix):]
            new_index = len(self.ids) - self.index
            self.ids = self.ids[self.index:]
            self.prefix = self.tokenizer.decode(self.ids, skip_special_tokens=True)
            self.index = new_index
            return new
        return None


def sample(session, w, mode, tokenizer, np):
    # Preparation/reset is outside request timing on both paths.
    validate_workload(w)
    stream = TextStream(tokenizer)
    session.reset()
    emitted, backend, sampling, text_ns, token_times = [], [], [], [], []
    visible, stopped = None, False
    request = time.perf_counter_ns()
    before = time.perf_counter_ns()
    for token in w['prompt']:
        row = session.step(token)['logits'].reshape(-1)
    ready = time.perf_counter_ns() - request
    prefill = time.perf_counter_ns() - before
    for forced in w['fixed_output']:
        if emitted:
            before = time.perf_counter_ns()
            row = session.step(emitted[-1])['logits'].reshape(-1)
            backend.append(time.perf_counter_ns() - before)
        before = time.perf_counter_ns()
        if mode == 'Greedy':
            if not np.isfinite(row).all():
                raise ValueError('nonfinite greedy logits')
            token = int(row.argmax())  # First index wins ties, matching Rust.
            sampling.append(time.perf_counter_ns() - before)
        else:
            token = forced
            sampling.append(0)
        emitted.append(token)
        token_times.append(time.perf_counter_ns() - request)
        before = time.perf_counter_ns()
        chunk = stream.push(token)
        if chunk is not None and visible is None:
            visible = time.perf_counter_ns() - request
        text_ns.append(time.perf_counter_ns() - before)
        if mode == 'Greedy' and token in STOPS:
            stopped = True
            break
    processed = session.processed
    if len(w['prompt']) + len(emitted) - processed != 1:
        raise ValueError('pending token invariant failed')
    append = [emitted[-1], *w['continuation']]
    before = time.perf_counter_ns()
    for token in append:
        row = session.step(token)['logits'].reshape(-1)
    continuation_backend = time.perf_counter_ns() - before
    next_token = int(row.argmax())
    continuation_next = time.perf_counter_ns() - before
    interval = token_times[-1] - token_times[0]
    throughput = lambda duration: (len(emitted) - 1) * 1e9 / duration if len(emitted) > 1 and duration else None
    return {'mode': mode, 'emitted': emitted, 'stopped_by_eos': stopped, 'prefill_ns': prefill,
        'backend_decode_ns': backend, 'sampling_ns': sampling, 'text_ns': text_ns,
        'ttft_ns': token_times[0] if mode == 'Greedy' else None, 'first_logits_ready_ns': ready,
        'first_visible_text_ns': visible, 'token_times_ns': token_times,
        'decode_tokens_per_second': throughput(interval), 'backend_decode_tokens_per_second': throughput(sum(backend)),
        'continuation_backend_ns': continuation_backend, 'continuation_next_token_ns': continuation_next,
        'continuation_emitted_token': next_token, 'pending_after_continuation': 1,
        'processed_before_continuation': processed, 'pending_emitted_tokens': 1,
        'processed_after_continuation': session.processed}


def qualification(model, manifest, reference, output, np):
    records = []
    for case_index, case in enumerate(manifest['cases']):
        entry = case['arrays'][0]
        if sha(reference / entry['file']) != entry['sha256']:
            raise ValueError('oracle row changed')
        expected = np.fromfile(reference / entry['file'], dtype='<f4').reshape(len(case['ids']), -1)
        session = CoreMLCheckSession(model, np)
        for position, token in enumerate(case['ids']):
            actual = session.step(token)
            if set(actual) != {'logits'}:
                raise ValueError('benchmark requires logits-only output')
            row = np.asarray(actual['logits'], dtype='<f4').reshape(-1)
            if row.shape != (151936,):
                raise ValueError('wrong output shape')
            error = np.abs(row.astype(np.float64) - expected[position].astype(np.float64))
            fraction = float((error / (manifest['logit_atol'] + manifest['rtol'] * np.abs(expected[position]))).max())
            greedy = int(row.argmax()) == int(expected[position].argmax())
            filename = f'{case_index}-{position}.bin'
            payload = row.tobytes()
            (output / filename).write_bytes(payload)
            records.append({'case': case_index, 'ids': case['ids'], 'position': position, 'processed': session.processed, 'file': filename,
                'sha256': hashlib.sha256(payload).hexdigest(), 'max_budget_fraction': fraction,
                'max_absolute_error': float(error.max()), 'greedy_exact': greedy, 'passed': fraction <= 1 and greedy})
    passed = all(r['passed'] for r in records)
    write(output / 'qualification.json', {'passed': passed, 'records': records,
        'scope': 'all teacher-forced prefixes for frozen original cases and both actual benchmark histories'})
    if not passed:
        raise ValueError('lean artifact fails unchanged reference gate')


def native(artifact, reference, policy_name, output):
    if policy_name not in POLICIES:
        raise ValueError('GPU policies forbidden')
    np, _, ct = imports()
    from tokenizers import Tokenizer
    compiled = json.loads((artifact / 'compile.json').read_text())
    if tree_hash(artifact / 'qwen.mlmodelc') != compiled['artifact_sha256']:
        raise ValueError('compiled artifact changed')
    w = json.loads(WORKLOAD.read_text())
    validate_workload(w)
    manifest = json.loads((reference / 'manifest.json').read_text())
    if manifest['revision'] != REV or manifest['workload_sha256'] != sha(WORKLOAD):
        raise ValueError('oracle/workload identity mismatch')
    inventory = json.loads((ROOT / 'fixtures/qwen-inventory.json').read_text())
    if sha(MODEL / 'tokenizer.json') != inventory['assets']['tokenizer.json']['sha256']:
        raise ValueError('tokenizer changed')
    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    output.mkdir(parents=True, exist_ok=False)
    policy = getattr(ct.ComputeUnit, policy_name)
    before = time.perf_counter_ns()
    model = ct.models.CompiledMLModel(str(artifact / 'qwen.mlmodelc'), compute_units=policy)
    load_ns = time.perf_counter_ns() - before
    write(output / 'metadata.json', {'host': current_host(), 'policy': policy_name, 'gpu_inference': False,
        'artifact_sha256': compiled['artifact_sha256'], 'reference_manifest_sha256': sha(reference / 'manifest.json'),
        'workload_file_sha256': sha(WORKLOAD), 'workload': w, 'tool_sha256': sha(__file__),
        'session_wrapper_sha256': sha(ROOT / 'reference/coreml_qwen_export.py'),
        'lock_sha256': sha(ROOT / 'reference/coreml-lock.txt'), 'load_compiled_ns': load_ns,
        'precision': 'FP32 weights/compute, approximate paired FP16 cache', 'logical_cache_bytes': 786432,
        'scope': 'warm logits-only predictions plus equivalent development greedy/text/continuation policy; Python bridge included',
        'threads': 'Core ML system-managed, Python reference torch single-thread; Rust numerical loop single-thread',
        'load_scope': 'compiled wrapper only; uncontrolled filesystem/runtime caches; not certified cold load',
        'energy': 'unavailable; no calibrated energy instrument', 'environment_before': environment(), 'run_status': 'running'})
    qualification(model, manifest, reference, output, np)
    session = CoreMLCheckSession(model, np)
    with (output / 'raw.jsonl').open('x') as stream:
        for mode in ('TeacherForced', 'Greedy'):
            for repetition in range(-w['warmups'], w['repetitions']):
                result = sample(session, w, mode, tokenizer, np)
                if any(result[key] != value for key, value in manifest['expected_samples'][mode].items()):
                    raise ValueError('sample history/cache/greedy differs from independent oracle')
                if repetition >= 0:
                    stream.write(json.dumps({'repetition': repetition, 'sample': result}, allow_nan=False) + '\n')
                    stream.flush()
    write(output / 'plan.json', plan_metadata(ct, artifact / 'qwen.mlmodelc', policy))
    write(output / 'completion.json', {'passed': True, 'owner_review': 'pending', 'environment_after': environment()})


def environment():
    snapshots = {}
    for name, command in [('power', ['pmset', '-g', 'batt']), ('thermal', ['pmset', '-g', 'therm']),
            ('memory_pressure', ['memory_pressure', '-Q'])]:
        result = subprocess.run(command, capture_output=True, text=True)
        snapshots[name] = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    return snapshots


def report(output):
    import numpy as np
    manifest = json.loads((output / 'oracle/manifest.json').read_text())
    manifest_hash = sha(output / 'oracle/manifest.json')
    for case in manifest['cases']:
        if sha(output / 'oracle' / case['arrays'][0]['file']) != case['arrays'][0]['sha256']:
            raise ValueError('oracle data changed')
    summaries, parity = {}, []
    for name in ('rust', *POLICIES):
        rows = [json.loads(line) for line in (output / name / 'raw.jsonl').read_text().splitlines()]
        meta = json.loads((output / name / 'metadata.json').read_text())
        if meta['workload_file_sha256'] != sha(WORKLOAD):
            raise ValueError('mismatched workload')
        if name == 'rust' and meta['run_status'] != 'complete':
            raise ValueError('incomplete Rust benchmark')
        if name != 'rust' and not json.loads((output / name / 'completion.json').read_text())['passed']:
            raise ValueError('incomplete native benchmark')
        if name != 'rust' and meta['reference_manifest_sha256'] != manifest_hash:
            raise ValueError('native reference mismatch')
        summaries[name] = {}
        for mode in ('TeacherForced', 'Greedy'):
            chosen = [r for r in rows if r['sample']['mode'] == mode]
            if sorted(r['repetition'] for r in chosen) != list(range(meta['workload']['repetitions'])):
                raise ValueError('missing/duplicate repetition')
            samples = [r['sample'] for r in chosen]
            for s in samples:
                if any(s[key] != value for key, value in manifest['expected_samples'][mode].items()):
                    raise ValueError('cross-path history/cache/greedy mismatch')
                if len(s['backend_decode_ns']) != len(s['emitted']) - 1:
                    raise ValueError('wrong decode evaluation count')
            summaries[name][mode] = {}
            for metric, label, convert in [('prefill_ns', 'prefill_ms', 1e-6), ('ttft_ns', 'ttft_ms', 1e-6),
                    ('first_visible_text_ns', 'first_visible_text_ms', 1e-6),
                    ('continuation_next_token_ns', 'continuation_ms', 1e-6), ('decode_tokens_per_second', 'decode_tokens_per_second', 1)]:
                values = [s[metric] * convert for s in samples if s[metric] is not None]
                if values:
                    center = statistics.median(values)
                    summaries[name][mode][label] = {'samples': len(values), 'median': center, 'min': min(values),
                        'max': max(values), 'MAD': statistics.median(abs(v - center) for v in values)}
        # Qualify actual Rust rows from the separate uninstrumented trace.
        if name == 'rust':
            result = json.loads((output / 'rust-qualified/results.json').read_text())
            qualified_identity = json.loads((output / 'rust-qualified/identity.json').read_text())
            if not result['passed'] or qualified_identity['manifest_sha256'] != manifest_hash:
                raise ValueError('Rust reference mismatch/incomplete qualification')
            seen = set()
            for r in result['records']:
                key = (r['case'], r['position'])
                if key in seen:
                    raise ValueError('duplicate Rust qualification position')
                seen.add(key)
                case = manifest['cases'][r['case']]
                expected = np.fromfile(output / 'oracle' / case['arrays'][0]['file'], dtype='<f4').reshape(len(case['ids']), -1)[r['position']]
                payload = output / 'rust-qualified' / r['file']
                if sha(payload) != r['sha256'] or r['ids'] != case['ids'] or r['processed'] != r['position'] + 1:
                    raise ValueError('Rust qualification record changed')
                actual = np.fromfile(payload, dtype='<f4')
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise ValueError('invalid Rust logits')
                fraction = float((np.abs(actual.astype(np.float64) - expected) /
                    (manifest['logit_atol'] + manifest['rtol'] * np.abs(expected))).max())
                passed = fraction <= 1 and int(actual.argmax()) == int(expected.argmax())
                parity.append({'case': r['case'], 'position': r['position'], 'max_budget_fraction': fraction, 'passed': passed})
            if {(r['case'], r['position']) for r in parity} != {(i, p) for i, c in enumerate(manifest['cases']) for p in range(len(c['ids']))}:
                raise ValueError('missing/extra Rust qualification positions')
            if not all(r['passed'] for r in parity):
                raise ValueError('Rust benchmark histories fail unchanged reference gate')
        else:
            q = json.loads((output / name / 'qualification.json').read_text())
            expected_keys = {(i, p) for i, c in enumerate(manifest['cases']) for p in range(len(c['ids']))}
            if not q['passed'] or len(q['records']) != len(expected_keys) or {(r['case'], r['position']) for r in q['records']} != expected_keys:
                raise ValueError('missing/duplicate/failed native qualification')
            for r in q['records']:
                case = manifest['cases'][r['case']]
                if r['ids'] != case['ids'] or r['processed'] != r['position'] + 1 or sha(output / name / r['file']) != r['sha256']:
                    raise ValueError('native qualification record changed')
    write(output / 'comparison.json', {'summaries': summaries, 'rust_qualification': parity,
        'identical_history_cache_and_greedy': True, 'owner_review': 'pending',
        'scope': 'same public token workload, logits-only output, greedy tie/EOS/text/pending/continuation contracts; implementations differ',
        'exceptions': ['Rust FP32 cache vs paired FP16 native cache', 'Rust single numerical thread vs system-managed Core ML workers',
            'Rust tokenizers0.21.4 vs Python tokenizers0.20.3 equivalent text stream', 'Python bridge/policy overhead included in native request timings'],
        'statistics': '10samples/mode/path, median/min/max/MAD; no robust p95 or product-wide speed claim',
        'energy': 'unavailable', 'placement': 'requires separately attributed trace; policy and plan insufficient'})


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'identity.json', {'host': current_host(), 'tool_sha256': sha(__file__),
        'workload_sha256': sha(WORKLOAD), 'source_artifact_identity_sha256': sha(source / 'identity.json'),
        'owner_authorization': '2026-10-03 do it autonomously after explicit bounded hardware-test opt-in request',
        'execution': 'sequential isolated children;180s cap/benchmark path,240s preparation cap; no simultaneous full models; no GPU'})
    commands = [
        ('prepare', [sys.executable, __file__, 'prepare', str(source), str(output)]),
        ('oracle', [sys.executable, __file__, 'oracle', str(source), str(output)]),
        ('rust-qualified', [str(ROOT / 'target/release/examples/policy_trace'), str(MODEL), str(output / 'oracle/manifest.json'), str(output / 'rust-qualified')]),
        ('rust', [str(ROOT / 'target/release/examples/benchmark'), 'qwen', str(MODEL), str(WORKLOAD), str(output / 'rust')]),
        *[(p, [sys.executable, __file__, p, str(source), str(output)]) for p in POLICIES],
    ]
    for name, command in commands:
        with (output / f'{name}-process.txt').open('x') as log:
            limit = 180 if name in ('rust', *POLICIES) else 240
            child = subprocess.Popen(['/usr/bin/time', '-l', *command], stdout=log, stderr=subprocess.STDOUT,
                cwd=ROOT, start_new_session=True)
            try:
                status = child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                # The group was created solely for this child and its descendants.
                # Kill the numerical process too, not only /usr/bin/time.
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                write(output / 'failure.json', {'phase': name, 'reason': f'{limit}s bound reached; not completed'})
                raise
        if status:
            write(output / 'failure.json', {'phase': name, 'exit_code': status})
            raise RuntimeError(f'{name} failed; preserve evidence')
    report(output)
    write(output / 'completion.json', {'passed': True, 'placement': 'separate trace required', 'owner_review': 'pending'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'prepare', 'oracle', 'report', *POLICIES])
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.stage == 'run': run(source, output)
    elif args.stage == 'prepare': prepare(source, output / 'artifact')
    elif args.stage == 'oracle': oracle(output / 'oracle')
    elif args.stage == 'report': report(output)
    else: native(output / 'artifact', output / 'oracle', args.stage, output / args.stage)
