#!/usr/bin/env python3
"""Check the named stop-bound qualification artifacts; never runs an engine.

The no-B run is the matched-prefix state reference: same binary, kernels, input
rows and sampling, but no speculative front-stage commit/restore. This is not a
general serial/pipeline numerical-equivalence claim or a model-quality test.
"""
import argparse
import json
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def fields(line):
    return dict(re.findall(r'(\w+)=([^ ]+)', line))


def state(record):
    lines = [s for s in record['trace'] if s.startswith('STOP_STATE ')]
    states = {int(fields(s)['stage']): fields(s) for s in lines}
    require(len(lines) == 4 and set(states) == {0, 1, 2, 3}, record['name'] + ': missing/duplicate stage')
    history = [s for s in record['trace'] if s.startswith('STOP_HISTORY ids=')]
    require(len(history) == 1, record['name'] + ': missing/duplicate consumed history')
    ids = [int(s) for s in history[0].split('ids=')[1].split(',') if s]
    require(ids == record['prompt_ids'] + record['ids'][:-1], record['name'] + ': wrong consumed input prefix')
    require(all(int(s['L']) == len(ids) for s in states.values()), record['name'] + ': stage length differs')
    require(states[0]['ple_prev'] == ','.join(map(str, ids[-2:])), record['name'] + ': PLE owner token history differs')
    return states


def same_request(a, b):
    for key in ('prompt_ids', 'ids', 'sampling'):
        require(a[key] == b[key], f"{a['name']}: different {key}")


def same_state(a, b):
    sa, sb = state(a), state(b)
    # idx_pooled's spare row and KV cells past L are not valid-prefix state.
    keys = ('L', 'gdn', 'ple', 'tail', 'dead', 'pooled', 'kv')
    for stage in range(4):
        for key in keys:
            require(sa[stage][key] == sb[stage][key], f"{a['name']}: stage {stage} differs in {key}")
    # In this four-stage profile only stage0 holds layer1 (Verifier::ple_stage).
    # Restore reconstructs ple_prev even on other stages; cold sessions leave it
    # at -1. Those non-owner fields are not consumed by PLE or advanced by commit.
    require(sa[0]['ple_prev'] == sb[0]['ple_prev'], a['name'] + ': PLE owner history differs')
    require(sa[0]['draft_upto'] == sb[0]['draft_upto'], a['name'] + ': draft interval differs')
    require(int(sa[0]['draft_upto']) == int(sa[0]['L']) - 1, a['name'] + ': fixture exceeds draft identity range')
    require(sa[0]['mtp'] == sb[0]['mtp'], a['name'] + ': valid draft KV differs')


def continuation(record, route):
    require(record['route_requested'] == route, record['name'] + ': wrong requested reuse route')
    restored = any('conversation cache: restored ' in s for s in record['trace'])
    require(restored == (route == 'park'), record['name'] + ': reuse route not proved')
    last = record['last']
    expected = len(record['prompt_ids']) - 1
    require((record['resume'], last.get('reused'), last.get('prompt_read')) == (expected, expected, 1),
            record['name'] + ': unexpected resume/reuse/read')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('artifacts', type=Path)
    ap.add_argument('--output', type=Path, required=True, help='new summary file; existing file refused')
    a = ap.parse_args()
    runs, manifests = {}, {}
    for name in ('baseline-sweep', 'candidate-sweep', 'candidate-no-b', 'candidate-serial-matched',
                 'candidate-eos-first', 'candidate-eos-last', 'release-smoke'):
        result = json.loads((a.artifacts / name / 'results.json').read_text())
        require(result['error'] is None, name + ': engine error')
        if name != 'baseline-sweep':
            require(not result['violations'] and not result['coverage_gaps'], name + ': incomplete/failed run')
        rows = result['requests']
        runs[name] = {r['name']: r for r in rows}
        require(len(runs[name]) == len(rows), name + ': duplicate request names')
        limit = 8 if name.startswith('candidate-eos-') else 2
        expected = {f'A-{limit}', f'live-{limit}', f'B-{limit}', f'park-{limit}'}
        if name in ('baseline-sweep', 'candidate-sweep', 'candidate-no-b'):
            expected |= {f'A-{n}' for n in (1, 2, 3, 4, 5, 7, 8)} | {'greedy-control'}
        require(set(runs[name]) == expected, name + ': missing or unexpected required requests')
        if name != 'baseline-sweep':
            continuation(runs[name][f'live-{limit}'], 'live')
            continuation(runs[name][f'park-{limit}'], 'park')
        manifests[name] = json.loads((a.artifacts / name / 'manifest.json').read_text())
        if name not in ('baseline-sweep', 'release-smoke'):
            for row in rows:
                state(row)
    baseline, candidate, reference = (runs[n] for n in ('baseline-sweep', 'candidate-sweep', 'candidate-no-b'))
    for name in ('candidate-no-b', 'candidate-serial-matched', 'candidate-eos-first', 'candidate-eos-last'):
        require(manifests[name]['engine_sha256'] == manifests['candidate-sweep']['engine_sha256'],
                name + ': diagnostic candidate binary differs')
    for name in ('candidate-no-b', 'candidate-serial-matched'):
        require(manifests[name]['args'] == manifests['candidate-sweep']['args'], name + ': startup layout arguments differ')
        info = json.loads((a.artifacts / name / 'engine-info.json').read_text())
        original = json.loads((a.artifacts / 'candidate-sweep/engine-info.json').read_text())
        for key in ('expert_slots', 'expert_slots_primary', 'expert_cache_mib', 'expert_cache_primary_mib'):
            require(info[key] == original[key], name + ': expert placement capacity differs')
    require(baseline['A-2']['state_lengths'] == [60] * 4, 'baseline did not reproduce overcommit')
    require(candidate['A-2']['state_lengths'] == [59] * 4, 'candidate did not correct the boundary')
    for name, row in candidate.items():
        same_request(baseline[name], row)
        require(baseline[name]['last']['drafts_accepted'] == row['last']['drafts_accepted'], name + ': raw verdict changed')
        same_request(row, reference[name])
        same_state(row, reference[name])
    require(not any(s.startswith('STOP_EARLY ') for r in reference.values() for s in r['trace']),
            'no-B reference speculated')
    boundary = candidate['A-2']['trace']
    require(any(all(f'{k}={v}' in s.split() for k, v in
                    {'path': 'pipe', 'seq': 1, 'accepted': 2, 'kept': 1, 'remaining': 1, 'B': 1, 'com': 7}.items())
                for s in boundary if s.startswith('STOP_BOUND ')), 'missing actual truncated B-launched window')
    for stage in (0, 1, 2):
        require(f'STOP_EARLY stage={stage} seq=1 rows=2' in boundary, 'missing early front-stage commit')
        require(f'STOP_UNDO stage={stage} seq=1 keep=1' in boundary, 'missing bounded front-stage replay')
    for name, limit, eos_id, kept in (('candidate-eos-first', 2, 11, 1), ('candidate-eos-last', 3, 220, 2)):
        row = runs[name]['A-8']
        require(row['last']['finish'] == 'stop' and row['ids'].index(eos_id) == len(row['ids']) - 1,
                name + ': first EOS position not established')
        same_request(row, candidate[f'A-{limit}'])
        same_state(row, candidate[f'A-{limit}'])
        require(any(all(f'{k}={v}' in s.split() for k, v in
                        {'path': 'pipe', 'seq': 1, 'T': 2, 'accepted': 2, 'kept': kept}.items())
                    for s in row['trace'] if s.startswith('STOP_BOUND ')), name + ': missing required EOS window')
    serial = runs['candidate-serial-matched']['A-2']
    require(any(all(x in s.split() for x in ('path=serial', 'T=2', 'accepted=2', 'kept=1', 'remaining=1'))
                for s in serial['trace'] if s.startswith('STOP_BOUND ')), 'serial truncation not exercised')
    same_request(serial, candidate['A-2'])
    same_state(serial, candidate['A-2'])
    for name, row in runs['release-smoke'].items():
        same_request(row, candidate[name])
    summary = {
        'passed': True, 'baseline_bad_L': 60, 'candidate_L': 59,
        'baseline_continuation_reused_read': [baseline['live-2']['last'][k] for k in ('reused', 'prompt_read')],
        'candidate_continuation_reused_read': [candidate['live-2']['last'][k] for k in ('reused', 'prompt_read')],
        'all_stage_bounded_replay': [0, 1, 2], 'state_reference': 'same-binary pipeline with theta=2 (no speculative B)',
        'state_keys': ['gdn', 'ple', 'tail', 'dead', 'pooled', 'kv', 'stage0 ple_prev', 'draft KV [0,L-1)'],
        'excluded_stale_state': ['KV beyond L', 'idx_pooled spare row', 'non-owner stage1/2/3 ple_prev'],
        'runs': {name: len(rows) for name, rows in runs.items()},
        'scope': 'short synthetic Q3/gfx906 native length/EOS and continuation; not a quality or performance benchmark',
    }
    with a.output.open('x') as out:
        json.dump(summary, out, indent=2)
        out.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
