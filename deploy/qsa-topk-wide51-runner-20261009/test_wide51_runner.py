#!/usr/bin/env python3
"""Small CPU-only parser/control-flow tests; every external operation is mocked."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, Mock, mock_open
import run_wide51 as R

HERE = Path(__file__).resolve().parent
M = json.loads((HERE/'test_inputs/manifest.json').read_text())
FILES = {f['file']: f for f in M['fixtures']}

def output(f, p):
    lines = [f"CASE_PASS nq={f['nq']} case={c['name']} last_nkv={c['nkv'][-1]} max_active_blocks={c['max_active_blocks']} source_expected_route={p['expected_routes'][c['name']]} route_observed=0 IDs=graph=reference=CPU padding=end_canaries=inputs_unchanged=PASS" for c in f['cases']]
    fit, wide = [p['env'][k] if p['env'][k] is not None else '<unset>' for k in ('STRATA_GFX906_TOPK_FIT17', 'STRATA_GFX906_TOPK_WIDE51')]
    lines.append(f"TRANSITION_PASS nq={f['nq']} capacity={f['capacity']} cases={len(f['cases'])} graph_instantiations=1 constant_addresses=1 constant_capacities=1 fit17={fit} wide51={wide} route_observed=0 timing_claim=0")
    return '\n'.join(lines)+'\n'

def args(root):
    return types.SimpleNamespace(binary=root/'transition.wide51', binary_sha='a'*64,
        manual_review=HERE/'test_inputs/manual-review.json', manual_sha=R.MANUAL,
        resource_gate=HERE/'test_inputs/resource-gate.json', resource_sha=R.RESOURCE,
        fixtures=HERE/'test_inputs', manifest_sha=R.MANIFEST, output=root/'fresh-output',
        leave_api_stopped=True, execute=True)

class RunnerTests(unittest.TestCase):
    def test_all_21_parser_configurations(self):
        R.validate_manifest(M)
        for p in M['future_process_plan']:
            f = FILES[p['fixture']]
            result = R.parse_transition(output(f, p), '', f, p)
            self.assertTrue(result['parity_passed'])
            self.assertFalse(result['route_observed'])
            self.assertFalse(result['timing_claim'])
            self.assertEqual(result['cases'], len(f['cases']))

    def test_parser_rejects_corruption_and_old_trace(self):
        p = M['future_process_plan'][5]
        f = FILES[p['fixture']]
        text = output(f, p)
        lines = text.splitlines()
        bad = [text.replace('route_observed=0', 'route_observed=1', 1),
               text.replace('IDs=graph=reference=CPU', 'IDs=graph=CPU', 1),
               text.replace('canaries=inputs_unchanged=PASS', 'canaries=inputs_unchanged=FAIL', 1),
               '\n'.join(lines[1:])+'\n', text+lines[0]+'\n',
               '\n'.join([lines[1], lines[0], *lines[2:]])+'\n',
               text.replace('capacity=52224', 'capacity=52225'),
               text.replace('timing_claim=0', 'timing_claim=1'),
               text+'SAMPLES_MS_PER_CALL 1.0\n']
        for corrupt in bad:
            with self.assertRaises(RuntimeError):
                R.parse_transition(corrupt, '', f, p)
        for stderr in ('strata gfx906 topk fit17: enabled=1 topology=guarded-ref256/reg17/reg66\n', 'FAIL: bad parity\n'):
            with self.assertRaises(RuntimeError):
                R.parse_transition(text, stderr, f, p)

    def test_matrix_cannot_expand_or_break_capacity_fallback(self):
        m = copy.deepcopy(M)
        m['future_process_plan'].append(copy.deepcopy(m['future_process_plan'][0]))
        with self.assertRaises(RuntimeError):
            R.validate_manifest(m)
        m = copy.deepcopy(M)
        p = m['future_process_plan'][10]
        p['expected_routes']['uniform_16_204800'] = 'guarded_reg51'
        with self.assertRaises(RuntimeError):
            R.validate_manifest(m)

    def test_commands_unset_traces_and_preserve_exact_controls(self):
        a = args(Path('/example'))
        for p in M['future_process_plan']:
            cmd = R.command(a, p, 1, 'strata-wide51-test')
            self.assertIn(R.S.IMAGE, cmd)
            self.assertIn('none', cmd)
            self.assertEqual(cmd[-3:], ['/work/transition.wide51', '/fixture/'+p['fixture'], '1'])
            env_args = cmd[cmd.index(R.S.IMAGE)+1:]
            for key in (*R.ABSENT, 'STRATA_GFX906_TOPK_FIT17', 'STRATA_GFX906_TOPK_WIDE51'):
                index = env_args.index(key)
                self.assertEqual(env_args[index-1], '-u')
            for key, value in p['env'].items():
                assignments = [s for s in env_args if s.startswith(key+'=')]
                self.assertEqual(assignments, [] if value is None else [key+'='+value])
            self.assertFalse(any(s in cmd for s in ('start', '--privileged')))

    def test_explicit_receipt_and_binary_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            a = args(Path(directory))
            a.binary.write_bytes(b'offline fake binary, never executable')
            a.binary_sha = R.S.sha(a.binary)
            R.verify_receipt_hashes(a)
            for key in ('binary_sha', 'manual_sha', 'resource_sha', 'manifest_sha'):
                bad = copy.copy(a)
                setattr(bad, key, '0'*64)
                with self.assertRaises(RuntimeError):
                    R.verify_receipt_hashes(bad)

    def run_mocked(self, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = args(root)
            calls = []
            settings = [{'pci': '0000:05:00.0', 'mode': 'high', 'power1_cap': 190000000},
                        {'pci': '0000:08:00.0', 'mode': 'high', 'power1_cap': 190000000}]
            def launch(cmd, name, ds, out, tag, timeout):
                device = int(cmd[-1])
                p = next(p for p in M['future_process_plan'] if tag == f'g{device}-'+p['process_id'])
                self.assertEqual(timeout, 180)
                self.assertEqual(cmd, R.command(a, p, device, name))
                calls.append((device, p['process_id']))
                receipt = {'exit_code': 7 if fail else 0, 'samples': [], 'command': cmd}
                (out/(tag+'.stdout')).write_text(output(FILES[p['fixture']], p))
                (out/(tag+'.stderr')).write_text('')
                R.S.write(out/(tag+'.driver.json'), receipt)
                return receipt
            def sha(file):
                suffix = str(file.relative_to(a.fixtures))
                return FILES[suffix]['sha256']
            with patch.object(R.S, 'R', root), patch.object(R, 'gate', return_value=(M, FILES)), \
                 patch.object(R, 'verify_receipt_hashes'), patch.object(R.S, 'sha', side_effect=sha), \
                 patch.object(R.S, 'call', return_value='') as external, \
                 patch.object(R.S, 'devices', return_value=['fake0', 'fake1']), \
                 patch.object(R.S, 'gpu', return_value=settings), \
                 patch.object(R.S, 'cooldown', return_value=settings) as cooldown, \
                 patch.object(R.S, 'cleanup') as cleanup, patch.object(R.P, 'launch', side_effect=launch), \
                 contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(RuntimeError):
                        R.execute(a)
                else:
                    R.execute(a)
                self.assertTrue(all(call.args[0] == ['docker', 'ps', '-q'] for call in external.call_args_list))
                self.assertEqual(cooldown.call_count, 1 if fail else 42)
                self.assertEqual(cleanup.call_count, 1 if fail else 42)
            summary = json.loads((a.output/'summary.json').read_text())
            self.assertTrue(summary['cleanup']['passed'])
            self.assertFalse(summary['api_start_stop_actions'])
            self.assertFalse(summary['timing_claim'])
            self.assertFalse(summary['route_observed'])
            if fail:
                self.assertEqual(len(calls), 1)
                self.assertEqual(summary['processes'][0]['exit_code'], 7)
                self.assertEqual(summary['processes'][0]['status'], 'failed')
                self.assertFalse(summary['qualification_completed'])
            else:
                self.assertEqual(calls, [(d, p['process_id']) for d in (0, 1) for p in M['future_process_plan']])
                self.assertEqual(sum(r['cases'] for r in summary['processes']), 1212)
                self.assertTrue(summary['qualification_completed'])

    def test_mocked_exact_42_sequential_processes(self):
        self.run_mocked()

    def test_failed_process_stops_sequence_and_keeps_exit_cleanup(self):
        self.run_mocked(fail=True)

    def test_preflight_failure_never_reaches_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            a = args(Path(directory))
            with patch.object(R.S, 'R', Path(directory)), patch.object(R, 'gate', side_effect=RuntimeError('bad hash')), \
                 patch.object(R.S, 'call') as call, patch.object(R.S, 'devices') as devices, patch.object(R.P, 'launch') as launch:
                with self.assertRaises(RuntimeError):
                    R.execute(a)
                call.assert_not_called()
                devices.assert_not_called()
                launch.assert_not_called()
                self.assertFalse(a.output.exists())

    def test_no_execute_and_one_shared_lock(self):
        argv = ['--binary', '/example/bin', '--manual-review', '/example/manual.json',
                '--resource-gate', '/example/resource.json', '--fixtures', '/example/fixtures',
                '--output', '/example/out', '--binary-sha', 'a'*64, '--manual-sha', R.MANUAL,
                '--resource-sha', R.RESOURCE, '--manifest-sha', R.MANIFEST, '--leave-api-stopped']
        with patch.object(R, 'execute') as execute, patch.object(R.fcntl, 'flock') as flock, \
             patch('builtins.open', mock_open()) as opened, contextlib.redirect_stdout(io.StringIO()):
            R.main(argv)
            execute.assert_not_called()
            opened.assert_not_called()
            flock.assert_not_called()
            R.main(argv+['--execute'])
            opened.assert_called_once_with(R.LOCK, 'w')
            execute.assert_called_once()
            self.assertEqual(flock.call_count, 1)
            self.assertEqual(flock.call_args.args[1], R.fcntl.LOCK_EX | R.fcntl.LOCK_NB)

if __name__ == '__main__':
    unittest.main(verbosity=2)
