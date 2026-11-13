#!/usr/bin/env python3
"""Controller-only WIDE51 QSTRAN01 fixtures and review plan; never runs a GPU."""
import argparse
import collections
import functools
import hashlib
import json
import pathlib
import struct
from analyze_qsa_dump import order_key, analyze_row

CAPACITIES = (52223, 52224, 52225)
NQS = (1, 4, 6, 8, 9)
OUT_CAP = 2080
GUARDS = 16
SOURCE_DIR = pathlib.Path('/home/opencode/ai/qsa-topk-wide51-resource-probe-20261009')
INPUTS = {
    str(SOURCE_DIR / 'qsa_select.fit17.cu'): '5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a',
    str(SOURCE_DIR / 'qsa_select.wide51.cu'): '98f30cd9fb2fd80619a8caf70c6c1d74bb5c7c14d50ca312b36158cd9518ae08',
    '/home/opencode/ai/qsa-topk-fit17-public-build-20261009/external-tests/qsa_reg17_transition.cpp': '65738c8661973c6c14c9a7910e29b28c3a75c4e49ac3fd4208be9e7903c17f35',
    '/home/opencode/ai/qsa-selection-capture-tools-20261009/reg17-transitions/analyze_qsa_dump.py': '53ad4c4d185d761aa35656e463d34b07a9c7bc91deeafb0e36e2b8bc3ea3daaa',
    '/home/opencode/ai/qsa-selection-capture-tools-20261009/reg17-transitions/generate_reg17_transitions.py': 'b3f3488b431da82fe73960558030c684a82814260c67f1ef1ed4af9b77699766',
}

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def validate_input_hashes():
    for path, expected in INPUTS.items():
        if sha(path) != expected:
            raise ValueError(f'input hash changed: {path}')

def exact_one(value):
    return value == '1'

def expected_route(nq, capacity, nkvs, fit17='1', wide51='1'):
    """Source model only. Never claims observation or replaces GPU correctness."""
    if capacity not in CAPACITIES or nq not in NQS or len(nkvs) != nq:
        raise ValueError('unsupported fixture geometry')
    if any(not 1 <= n <= 262144 or n // 4 + 1 > capacity for n in nkvs):
        raise ValueError('row violates n_bid + 1 <= max_blocks')
    # No active-count argument; reach is immutable score-row capacity.
    if nq > 8:
        return 'fallback_ref'
    if nkvs[-1] <= 24576:
        return 'guarded_ref'
    if not exact_one(fit17):
        return 'guarded_reg66_legacy'
    if max(n // 4 + 1 for n in nkvs) <= 17408:
        return 'guarded_reg17'
    return 'guarded_reg51' if exact_one(wide51) and capacity <= 52224 else 'guarded_reg66'

def oracle(nkv, bits):
    width = min(nkv, 2051)
    ids = []
    for b in sorted(range(len(bits)), key=lambda b: (-order_key(bits[b]), b)):
        weight = min(4, nkv - 4*b)
        ids.extend(range(4*b, 4*b + min(weight, width-len(ids))))
        if len(ids) == width:
            break
    return sorted(ids)

def scores(active, pattern, salt):
    if pattern == 0:
        pool = [0, 0x80000000, 0x3f800000, 0x3f800000, 0x7fc00001,
                0xffc00007, 0x7f800000, 0xff800000, 0xbf800000, 0x00000001]
        return [pool[(b*7+salt) % len(pool)] for b in range(active)]
    if pattern == 1:
        return [0x7fc00001 if b % 2 else 0xffc00007 for b in range(active)]
    if pattern == 2:
        return [0x80000000 if (b+salt) % 2 else 0 for b in range(active)]
    x = 0x9e3779b9 ^ salt
    out = []
    for _ in range(active):
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        x &= 0xffffffff
        out.append(x)
    return out

def cases(nq, capacity):
    if nq not in NQS or capacity not in CAPACITIES:
        raise ValueError('unsupported fixture geometry')
    # The final two uniform rows restore FIT17 then short; all four FIT17
    # threshold tails, exact requested contexts, and valid allocation edge tails.
    values = [1, 2051, 24575, 24576, 24577,
              69628, 69629, 69630, 69631, 69632, 69633, 69634, 69635,
              200703, 200704, 204799, 204800,
              4*capacity-4, 4*capacity-1, 69631, 24576, 17]
    out = [(f'uniform_{i:02d}_{n}', [n]*nq, i % 4) for i, n in enumerate(values)]
    if nq > 1:
        for boundary, label in ((24576, 'short'), (69632, 'fit17'),
                                (200704, '200704'), (204800, '204800')):
            out.append((f'straddle_{label}', [boundary-nq//2+i for i in range(nq)], 3))
        a = [65539]*nq
        a[0] = 204800
        out.append(('earlier_long_last_middle', a, 3))
        a = [65539]*nq
        a[-2] = 4*capacity-1
        out.append(('penultimate_capacity_last_middle', a, 0))
        a = [24576]*nq
        a[0] = 4*capacity-1
        out.append(('earlier_capacity_last_reference', a, 0))
        out.append(('shrink_mixed', [[1, 17, 2048, 2051, 24575, 24576][i % 6]
                                     for i in range(nq)], 2))
    for _, nkvs, _ in out:
        expected_route(nq, capacity, nkvs)
    return out

@functools.lru_cache(maxsize=None)
def packed_row(nkv, pattern, salt):
    step = (nkv-1, nkv, nkv//4, min(nkv, 2051))
    bits = scores(nkv//4+1, pattern, salt)
    ids = oracle(nkv, bits)
    # Vetted independent weighted-radix reconstruction, including tie emission.
    analyze_row({'step': step, 'bits': bits, 'ids': ids})
    payload = struct.pack('<2I', len(bits), len(ids))
    payload += struct.pack(f'<{len(bits)}I', *bits)
    payload += struct.pack(f'<{len(ids)}i', *ids)
    return step, payload

def generate_fixture(root, nq, capacity):
    cs = cases(nq, capacity)
    path = root / 'fixtures' / f'wide51-cap{capacity}-nq{nq}.bin'
    records = []
    with path.open('wb') as f:
        f.write(struct.pack('<8s8I', b'QSTRAN01', 1, 0x01020304, nq, capacity,
                            OUT_CAP, 4, len(cs), GUARDS))
        for ci, (name, nkvs, pattern) in enumerate(cs):
            bname = name.encode()
            f.write(struct.pack('<I', len(bname)))
            f.write(bname)
            rows = [packed_row(n, pattern, ci*17+row) for row, n in enumerate(nkvs)]
            for step, _ in rows:
                f.write(struct.pack('<4i', *step))
            for _, payload in rows:
                f.write(payload)
            records.append(dict(name=name, nkv=nkvs, pattern=pattern,
                expected_primary_route=expected_route(nq, capacity, nkvs),
                max_active_blocks=max(n//4+1 for n in nkvs), tails=[n % 4 for n in nkvs]))
    return dict(file=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size,
                nq=nq, capacity=capacity, cases=records)

def process_plan(fixtures):
    plan = []
    def append(fixture, label, fit17, wide51):
        routes = {c['name']: expected_route(fixture['nq'], fixture['capacity'],
                   c['nkv'], fit17, wide51) for c in fixture['cases']}
        plan.append(dict(process_id=f'{label}-cap{fixture["capacity"]}-nq{fixture["nq"]}',
            fixture=fixture['file'], fixture_sha256=fixture['sha256'],
            env={'STRATA_GFX906_TOPK_REG': '2', 'STRATA_GFX906_TOPK_FIT17': fit17,
                 'STRATA_GFX906_TOPK_WIDE51': wide51, 'STRATA_TOPK_TRACE': None,
                 'STRATA_GFX906_TOPK_FIT17_TRACE': None, 'STRATA_TOPK_OLD': None},
            expected_launch_count_per_selector_call=1 if fixture['nq'] > 8 else (3 if exact_one(fit17) else 2),
            expected_routes=routes, expected_route_counts=dict(collections.Counter(routes.values())),
            run_status='NOT_RUN', separate_os_process_required=True,
            constant_capacity=True, constant_addresses=True, graph_instantiations=1))
    for fixture in fixtures:
        append(fixture, 'primary', '1', '1')
    control = next(f for f in fixtures if f['capacity'] == 52224 and f['nq'] == 4)
    for label, fit17, wide51 in [('wide-unset', '1', None), ('wide-zero', '1', '0'),
                               ('wide-invalid-two', '1', '2'), ('wide-invalid-leading-zero', '1', '01'),
                               ('fit-unset', None, '1'), ('fit-zero', '0', '1')]:
        append(control, label, fit17, wide51)
    return plan

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent)
    args = p.parse_args()
    validate_input_hashes()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'fixtures').mkdir(exist_ok=True)
    fixtures = [generate_fixture(root, nq, cap) for cap in CAPACITIES for nq in NQS]
    plan = process_plan(fixtures)
    result = dict(format='QSTRAN01', experiment='isolated default-off WIDE51 selector',
        gpu_executed=False, gpu_binary_built=False, timing_claim=False, performance_evidence=False,
        source_provenance=INPUTS, fixtures=fixtures, future_process_plan=plan,
        fixture_count=len(fixtures), future_process_count=len(plan),
        future_process_count_per_device=len(plan), future_process_count_two_devices=2*len(plan),
        future_graph_replays=sum(len(p['expected_routes']) for p in plan),
        fixture_cases=sum(len(f['cases']) for f in fixtures),
        fixture_rows=sum(f['nq']*len(f['cases']) for f in fixtures),
        capacity_contract='Every row: n_bid = floor(n_kv/4); n_bid + 1 <= max_blocks',
        route_evidence='Source-derived expectation only: exact environment strings, immutable capacity, pinned host dispatch and current-step device guards. No fabricated device counter; no old trace proof.',
        oracle='Integer IEEE order keys + ascending block tie IDs, checked by unchanged weighted-radix reconstruction',
        resource_gate='Parent reported automated and independent manual ISA/resource gate PASS; gate artifacts remain in the separate resource-probe package',
        pending=[
                 'Isolated component-only compile/link validation of this adapted driver',
                 'Parent authorization and GPU correctness execution',
                 'Separate real 200K capture qualification; synthetic fixtures are not performance evidence'])
    (root / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['fixture_count', 'future_process_count',
        'future_graph_replays', 'fixture_cases', 'fixture_rows', 'gpu_executed', 'timing_claim']}, indent=2))

if __name__ == '__main__':
    main()
