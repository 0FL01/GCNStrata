#!/usr/bin/env python3
"""CPU-only source/model, binary-fixture, oracle and inherited-driver checks."""
import argparse
import hashlib
import json
import pathlib
import struct
import unittest
from analyze_qsa_dump import analyze_row, order_key
import generate_wide51_transitions as g

ROOT = pathlib.Path(__file__).resolve().parent

def read_fixture(path):
    data = path.read_bytes()
    offset = 0
    def read(fmt):
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset+size > len(data):
            raise ValueError('truncated fixture')
        result = struct.unpack_from(fmt, data, offset)
        offset += size
        return result
    magic, version, endian, nq, capacity, out_cap, count, ncases, guards = read('<8s8I')
    if (magic != b'QSTRAN01' or version != 1 or endian != 0x01020304 or nq not in g.NQS or
            capacity not in g.CAPACITIES or out_cap != 2080 or count != 4 or guards != 16 or
            not 1 <= ncases <= 100):
        raise ValueError('unsupported dimensions')
    cases = []
    for _ in range(ncases):
        namelen, = read('<I')
        if not 0 < namelen < 128:
            raise ValueError('bad name length')
        name, = read(f'<{namelen}s')
        steps = [read('<4i') for _ in range(nq)]
        rows = []
        for step in steps:
            pos, nkv, nbid, width = step
            if not (1 <= nkv <= 262144 and pos == nkv-1 and nbid == nkv//4 and
                    nbid+1 <= capacity and width == min(nkv, 2051)):
                raise ValueError('row violates geometry or capacity contract')
            active, length = read('<2I')
            if active != nbid+1 or length != width or length > out_cap:
                raise ValueError('packed lengths disagree')
            bits = read(f'<{active}I')
            ids = read(f'<{length}i')
            if any(not 0 <= cell < nkv or (i and cell <= ids[i-1]) for i, cell in enumerate(ids)):
                raise ValueError('invalid CPU IDs')
            rows.append({'step': step, 'bits': bits, 'ids': ids})
        cases.append((name.decode(), rows))
    if offset != len(data):
        raise ValueError('trailing bytes')
    return nq, capacity, cases

class ModelTests(unittest.TestCase):
    def test_exact_environment_switches(self):
        invalid = [None, '', '0', '2', '01', '1x', ' 1', '1 ', '+1', 'true']
        for value in invalid:
            self.assertFalse(g.exact_one(value))
            self.assertEqual(g.expected_route(4, 52224, [204800]*4, '1', value), 'guarded_reg66')
            self.assertEqual(g.expected_route(4, 52224, [204800]*4, value, '1'), 'guarded_reg66_legacy')
        self.assertEqual(g.expected_route(4, 52224, [204800]*4, '1', '1'), 'guarded_reg51')

    def test_capacity_is_immutable_host_bound(self):
        for nq in g.NQS:
            for cap in g.CAPACITIES:
                expected = 'fallback_ref' if nq == 9 else ('guarded_reg51' if cap <= 52224 else 'guarded_reg66')
                for nkv in [200703, 200704, 204799, 204800, 4*cap-4, 4*cap-1]:
                    self.assertEqual(g.expected_route(nq, cap, [nkv]*nq), expected)

    def test_short_middle_wide_growth_and_restoration(self):
        nkvs = [24576, 24577, 69631, 69632, 204800, 69631, 24576]
        expected = ['guarded_ref', 'guarded_reg17', 'guarded_reg17', 'guarded_reg51',
                    'guarded_reg51', 'guarded_reg17', 'guarded_ref']
        for nq in (1, 4, 6, 8):
            self.assertEqual([g.expected_route(nq, 52224, [n]*nq) for n in nkvs], expected)

    def test_nonmonotonic_rows_use_all_rows_for_fit17(self):
        for nq in (4, 6, 8):
            for index in range(nq-1):
                nkvs = [65539]*nq
                nkvs[index] = 204800
                self.assertEqual(g.expected_route(nq, 52224, nkvs), 'guarded_reg51')
                self.assertEqual(g.expected_route(nq, 52225, nkvs), 'guarded_reg66')
                nkvs[-1] = 24576
                self.assertEqual(g.expected_route(nq, 52224, nkvs), 'guarded_ref')

    def test_all_fixture_rows_are_in_contract(self):
        for cap in g.CAPACITIES:
            for nq in g.NQS:
                cs = g.cases(nq, cap)
                self.assertEqual(len(cs), 22 if nq == 1 else 30)
                self.assertEqual(len(set(name for name, _, _ in cs)), len(cs))
                observed = set()
                for _, nkvs, _ in cs:
                    self.assertEqual(len(nkvs), nq)
                    for n in nkvs:
                        self.assertLessEqual(n//4+1, cap)
                        observed.add(n)
                self.assertTrue({24575, 24576, 24577, 69628, 69629, 69630, 69631,
                    69632, 69633, 69634, 69635, 200703, 200704, 204799, 204800,
                    4*cap-4, 4*cap-1}.issubset(observed))

    def test_ieee_exception_and_tie_semantics(self):
        self.assertEqual(order_key(0x7fc00001), 0)
        self.assertEqual(order_key(0xffc00007), 0)
        self.assertEqual(order_key(0x00000000), order_key(0x80000000))
        self.assertLess(order_key(0xff800000), order_key(0xbf800000))
        self.assertLess(order_key(0xbf800000), order_key(0))
        self.assertLess(order_key(0), order_key(0x7f800000))
        for nkv in (1, 17, 2051, 69631, 204800):
            for pattern in (1, 2):
                bits = g.scores(nkv//4+1, pattern, 0)
                ids = g.oracle(nkv, bits)
                self.assertEqual(ids, list(range(min(nkv, 2051))))
                analyze_row({'step': (nkv-1, nkv, nkv//4, min(nkv, 2051)), 'bits': bits, 'ids': ids})

    def test_pinned_source_dispatch_and_guard_contract(self):
        g.validate_input_hashes()
        source = (g.SOURCE_DIR/'qsa_select.wide51.cu').read_text()
        for token in ['const bool counted = active_blocks > 0;',
                      'const int64_t reach = counted && active_blocks < max_blocks ? active_blocks : max_blocks;',
                      'const bool extended_decode = decode_reg66 && !counted && nq <= 8;',
                      'if (wide51 && reach <= (int64_t) TK_T * 51)',
                      'block_topk_reg_kernel<51, true, 2>',
                      'const bool fits17 = (int64_t) max_bid + 1 <= (int64_t) TK_T * 17;',
                      'for (int q = 0; q < 8; ++q)']:
            self.assertIn(token, source)
        self.assertIn('const char* v = std::getenv("STRATA_GFX906_TOPK_WIDE51");\n                return v && std::strcmp(v, "1") == 0;', source)
        self.assertIn('const char* v = std::getenv("STRATA_GFX906_TOPK_FIT17");\n            return v && std::strcmp(v, "1") == 0;', source)

    def test_vetted_gpu_check_logic_preserved(self):
        base = pathlib.Path('/home/opencode/ai/qsa-topk-fit17-public-build-20261009/external-tests/qsa_reg17_transition.cpp').read_text()
        driver = (ROOT/'qsa_wide51_transition.cpp').read_text()
        def region(text, start, end):
            return text.split(start, 1)[1].split(end, 1)[0]
        self.assertEqual(region(base, 'struct Resources {', '\nint main'),
                         region(driver, 'struct Resources {', '\nint main'))
        self.assertEqual(region(base, '        Resources r;', '            int32_t max_blocks=0;'),
                         region(driver, '        Resources r;', '            int32_t max_blocks=0;'))
        self.assertIn('source_expected_route=%s route_observed=0', driver)
        self.assertIn('require(!trace && !std::getenv("STRATA_GFX906_TOPK_FIT17_TRACE")', driver)
        self.assertNotIn('candidate_intended_route', driver)
        self.assertEqual(g.sha(ROOT/'analyze_qsa_dump.py'), '53ad4c4d185d761aa35656e463d34b07a9c7bc91deeafb0e36e2b8bc3ea3daaa')

def verify_generated_fixtures(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['gpu_executed'] or manifest['timing_claim']:
        raise AssertionError('offline scope mislabelled')
    if len(manifest['fixtures']) != 15 or len(manifest['future_process_plan']) != 21:
        raise AssertionError('unexpected test-plan expansion')
    seen = set()
    checked_rows = checked_cases = 0
    for fixture in manifest['fixtures']:
        path = root/fixture['file']
        if g.sha(path) != fixture['sha256']:
            raise AssertionError(f'fixture hash mismatch: {path}')
        nq, cap, cs = read_fixture(path)
        if (nq, cap) != (fixture['nq'], fixture['capacity']) or len(cs) != len(fixture['cases']):
            raise AssertionError('manifest/fixture geometry mismatch')
        for (name, rows), info in zip(cs, fixture['cases']):
            if name != info['name'] or [r['step'][1] for r in rows] != info['nkv']:
                raise AssertionError('manifest/fixture case mismatch')
            if g.expected_route(nq, cap, info['nkv']) != info['expected_primary_route']:
                raise AssertionError('route model mismatch')
            for row in rows:
                digest = hashlib.sha256(struct.pack('<4i', *row['step']) +
                    struct.pack(f'<{len(row["bits"])}I', *row['bits']) +
                    struct.pack(f'<{len(row["ids"])}i', *row['ids'])).digest()
                if digest not in seen:
                    analyze_row(row)
                    seen.add(digest)
                checked_rows += 1
            checked_cases += 1
        print(f'CPU_FIXTURE_PASS {fixture["file"]} cases={len(cs)} rows={nq*len(cs)}', flush=True)
    if manifest['future_process_plan'] != g.process_plan(manifest['fixtures']):
        raise AssertionError('future process plan mismatch')
    return {'fixture_count': len(manifest['fixtures']), 'cases_checked': checked_cases,
            'rows_checked': checked_rows, 'unique_rows_weighted_radix_checked': len(seen),
            'future_process_count': len(manifest['future_process_plan'])}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', action='store_true', help='also read every generated binary and verify each unique row')
    args = p.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    report = {'cpu_tests_passed': result.testsRun, 'gpu_executed': False,
              'gpu_binary_built': False, 'timing_claim': False,
              'verification': 'CPU/source/fixture validation only; adapted HIP driver compile and GPU checks are NOT_RUN'}
    if args.fixtures:
        report.update(verify_generated_fixtures(ROOT))
    (ROOT/'offline-test-results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
