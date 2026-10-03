"""T19 read-only analysis of public xctrace XML exports, not private ANE APIs.

CPU sampling proves observed CPU involvement; counts are not elapsed-time
percentages and cannot establish exclusive placement or zero ANE execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def analyze(toc_path, profile_path):
    toc = ET.parse(toc_path).getroot()
    target = toc.find('.//target/process')
    if target is None or target.get('return-exit-status') != '0':
        raise ValueError('successful launched target required')
    pid = int(target.get('pid'))
    tables = toc.findall('.//run/data/table')
    root = ET.parse(profile_path).getroot()
    ids = {e.get('id'): e for e in root.iter() if e.get('id') is not None}

    def resolve(element):
        if element is None:
            raise ValueError('missing expected XML field')
        if element.get('ref') is not None:
            return ids[element.get('ref')]
        return element

    # Points of Interest can be exported in duplicate views. Deduplicate exact
    # events before pairing our control's public, synchronous intervals.
    events = set()
    for node in root:
        index = re.search(r'table\[(\d+)\]', node.get('xpath', ''))
        if not index or tables[int(index.group(1)) - 1].get('schema') != 'os-signpost':
            continue
        for row in node.findall('row'):
            if int(resolve(resolve(row.find('process')).find('pid')).text) != pid:
                continue
            if resolve(row.find('subsystem')).text != 'org.llm-inference.t19':
                continue
            events.add((int(resolve(row.find('event-time')).text), resolve(row.find('signpost-name')).text,
                resolve(row.find('event-type')).text, resolve(row.find('os-signpost-identifier')).text))
    active, intervals = {}, []
    for timestamp, name, kind, identifier in sorted(events):
        key = (name, identifier)
        if kind == 'Begin':
            if key in active:
                raise ValueError('overlapping exclusive control intervals')
            active[key] = timestamp
        elif kind == 'End':
            if key not in active:
                raise ValueError('control interval ends without begin')
            start = active.pop(key)
            if name == 'Prediction':
                intervals.append([start, timestamp])
    if active:
        raise ValueError('incomplete control interval')

    counts = {'target_cpu_samples': 0, 'coreml_prediction_samples': 0,
        'bnns_cpu_prediction_samples': 0, 'target_samples_without_backtrace': 0}
    times = []
    for node in root:
        index = re.search(r'table\[(\d+)\]', node.get('xpath', ''))
        if not index or tables[int(index.group(1)) - 1].get('schema') != 'time-profile':
            continue
        for row in node.findall('row'):
            process = resolve(row.find('process'))
            if int(resolve(process.find('pid')).text) != pid:
                continue
            counts['target_cpu_samples'] += 1
            if row.find('tagged-backtrace') is None:
                counts['target_samples_without_backtrace'] += 1
                continue
            trace = resolve(row.find('tagged-backtrace'))
            names = [resolve(frame).get('name', '') for frame in trace.findall('frame')]
            if not any('predictionFromFeatures' in name for name in names):
                continue
            counts['coreml_prediction_samples'] += 1
            if any('BnnsCpuInferenceOperation::ExecuteSync' in name for name in names) and any('BNNSGraphContextExecute' in name for name in names):
                counts['bnns_cpu_prediction_samples'] += 1
                times.append(int(resolve(row.find('sample-time')).text))
    return {'target_pid': pid, **counts, 'bnns_sample_time_range_ns': [min(times), max(times)] if times else None,
        'unique_control_signpost_events': len(events), 'control_prediction_intervals_ns': intervals,
        'bnns_samples_within_control_prediction_intervals': sum(any(start <= t <= end for start, end in intervals) for t in times),
        'toc_sha256': hashlib.sha256(Path(toc_path).read_bytes()).hexdigest(),
        'profile_sha256': hashlib.sha256(Path(profile_path).read_bytes()).hexdigest(),
        'scope': 'target-only sampled CPU stacks; imports/loading and lazy initialization may be inside predictions; not operator timing or wall-time percentages',
        'placement': 'observed CPU/BNNS involvement only; no exclusive CPU placement or zero ANE inference'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('toc', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('new_output', type=Path)
    args = parser.parse_args()
    result = analyze(args.toc, args.profile)
    with args.new_output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
