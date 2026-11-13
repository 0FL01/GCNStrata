#!/usr/bin/env python3
"""CPU only: native/container calls are mocked; never accesses MI50, build or API."""
import ast
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch, Mock

import existing_correctness as helper
import capture_arm
import run_paired_200k as runner
from test_validate_200k import fixture
from validate_200k import validate_pair

HERE = Path(__file__).resolve().parent


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((HERE / 'correctness.spec.json').read_text())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.previous_umask = os.umask(0o077)
        self.addCleanup(os.umask, self.previous_umask)

    def test_exact_dependencies_and_minimal_arm_copy(self):
        self.assertEqual(runner.sha(HERE / 'existing_correctness.py'), runner.HELPER_SHA)
        self.assertEqual(runner.sha(HERE / 'correctness.spec.json'), runner.SPEC_SHA)
        original = (HERE / 'existing_correctness.py').read_text()
        function = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'run_arm')
        expected = '\n'.join(original.splitlines()[function.lineno - 1:function.end_lineno]) + '\n'
        expected = expected.replace('def run_arm(tag,r,s,ids,ds,out):', 'def run_arm(tag,r,s,ids,ds,out,*,capture_dir=None):')
        expected = expected.replace("name='strata-fit17-correctness-'+tag.lower()", "name='strata-qsa-200k-'+tag.lower()")
        expected = expected.replace("cmd=['docker','run','--rm','-i'", "cmd=['docker','run','--rm','--user',str(os.geteuid())+':'+str(os.getegid()),'-i'")
        expected = expected.replace("    cmd+=['-v',str(R)+':/src:ro'", "    if capture_dir is not None:\n        capture_dir=Path(capture_dir)\n        need(capture_dir.is_dir() and not capture_dir.is_symlink() and capture_dir.stat().st_mode & 0o777==0o700,'private capture directory')\n        cmd+=['-v',str(capture_dir)+':/captures:rw']\n    cmd+=['-v',str(R)+':/src:ro'")
        expected = expected.replace("    for k,v in sorted(r['flags'].items()):cmd+=['-e',k+'='+v]", "    group_ids=sorted(set(os.getgroups()))\n    for gid in group_ids:cmd+=['--group-add',str(gid)]\n    for k,v in sorted(r['flags'].items()):cmd+=['-e',k+'='+v]")
        expected = expected.replace("    write(out/(tag+'.manifest.json'),m);print", "    m['container_identity']={'uid':os.geteuid(),'gid':os.getegid(),'supplementary_group_ids':group_ids,'original_named_group_add':['video']}\n    write(out/(tag+'.manifest.json'),m);print")
        actual = (HERE / 'capture_arm.py').read_text().split('from existing_correctness import *\n\n', 1)[1]
        self.assertEqual(actual, expected)

    def test_exact_configuration_and_one_plus_one_schedule(self):
        case = runner.validate_config(self.spec)
        self.assertEqual((case['prompt_tokens'], case['generate']), (200000, 256))
        arms = runner.make_arms(self.spec, 'a' * 64)
        self.assertEqual([a['tag'] for a in arms], ['control', 'diagnostic'])
        self.assertNotIn(runner.DUMP, arms[0]['flags'])
        self.assertEqual(arms[1]['flags'], dict(arms[0]['flags'], STRATA_QSA_SELECT_DUMP='/captures/window'))
        for arm in arms:
            self.assertEqual(arm['flags'][runner.FLAG], '1')
            self.assertEqual(arm['flags']['STRATA_GFX906_TOPK_REG'], '2')
            self.assertNotIn('STRATA_GFX906_TOPK_WIDE51', arm['flags'])
            self.assertNotIn('STRATA_TOPK_OLD', arm['flags'])
            self.assertNotIn('STRATA_VERIFY_PROFILE', arm['flags'])
            self.assertEqual(arm['native_args'], self.spec['native_args'])
        for mutate in [('flag', 'STRATA_GFX906_TOPK_WIDE51'), ('flag', 'STRATA_VERIFY_PROFILE'), ('flag', 'STRATA_TOPK_OLD'), ('arg', '--lookup-chain')]:
            bad = copy.deepcopy(self.spec)
            if mutate[0] == 'flag': bad['ff34_flags'][mutate[1]] = '1'
            else: bad['native_args'] += [mutate[1], '1']
            with self.assertRaises(RuntimeError): runner.validate_config(bad)
        for bad_hash in [None, 'bad', runner.BASE_SHA]:
            with self.assertRaises(RuntimeError): runner.make_arms(self.spec, bad_hash)

    def protocol(self, invalid_done=False):
        info = {k: '0' for k in helper.TOPOLOGY_KEYS}
        info.update(expert_slots='19078', expert_slots_primary='9900', kv='int8', context='204800', lookup='3', mtp_max='4', spec='6', kv_resident='32768', pcie_frac='0.00')
        done = 'DONE 256 200000 1000 2000 length 2 3 0 4 5 6 7 8.0 200000 9' + (' 1' if invalid_done else '') + '\n'
        stdout = 'INFO ' + ' '.join(k + '=' + v for k, v in info.items()) + '\nREADY 204800\n' + 'T 42\n' * 256 + done
        return stdout, info

    def exercise_pair(self, mismatch=False, invalid_done=False, nonempty_capture=False):
        out = self.path / 'out'
        out.mkdir(mode=0o700)
        (out / 'private').mkdir(mode=0o700)
        (out / 'private/captures').mkdir(mode=0o700)
        if nonempty_capture:
            (out / 'private/captures/preexisting').write_text('private')
        processes, commands, requests = [], [], []
        stdout, info = self.protocol(invalid_done=invalid_done)
        class Input(io.StringIO):
            def close(self):
                requests.append(self.getvalue())
                super().close()
        class FakeProcess:
            def __init__(self, cmd, stdin=None, stdout=None, stderr=None, **kwargs):
                commands.append(cmd)
                self.stdin = Input()
                self.stdout = io.StringIO(protocol_stdout)
                self.returncode = None
                stderr.write('PLE weights locked in RAM\n')
                stderr.write('strata gfx906 topk fit17: enabled=1 topology=guarded-ref256/reg17/reg66 max_blocks=17408 nq=6 capacity_blocks=51202 cap=2051 (device steps select the per-window arm)\n')
                stderr.write('attn-query-swizzle: active=1 \n')
                stderr.write('strata serve: suffix drafts: 2 windows, 3 of 4 drafts accepted\n')
                stderr.flush()
                processes.append(self)
            def wait(self, timeout=None): self.returncode = 0; return 0
            def poll(self): return self.returncode
        protocol_stdout = stdout
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            if str(path) == '/proc/meminfo': return 'MemAvailable: 100000000 kB\n'
            return original_read(path, *args, **kwargs)
        summary = {'arms_started': [], 'arms_completed': []}
        reference = dict(passed=True, prompt_tokens=200000, actual_outputs=256, output_ids=[99 if mismatch else 42] * 256, engine_info=info)
        with patch.object(capture_arm, 'call', return_value=''), patch.object(capture_arm, 'sha', side_effect=lambda p: runner.BASE_SHA if str(p).startswith(str(runner.BASE)) else 'a' * 64), patch.object(capture_arm, 'gpu', return_value=[]), patch.object(capture_arm, 'cooldown', return_value=[]), patch.object(capture_arm.subprocess, 'Popen', FakeProcess), patch.object(Path, 'read_text', read):
            if invalid_done or nonempty_capture:
                expected = 'native DONE must' if invalid_done else 'fresh empty private capture directory'
                with self.assertRaisesRegex(RuntimeError, expected):
                    runner.paired_requests(self.spec, 'a' * 64, [42] * 200000, [], out, helper, capture_arm.run_arm, summary, reference)
                self.assertEqual(len(processes), 1)
                private = out / 'private/control/private'
                self.assertEqual(len(json.loads((private / 'control.output-ids.json').read_text())), 256)
                self.assertTrue((private / 'control.done.txt').read_text().startswith('DONE 256 200000 '))
                runner.private_tree(out)
                return
            if mismatch:
                with self.assertRaisesRegex(RuntimeError, 'full ff34 output ID mismatch'):
                    runner.paired_requests(self.spec, 'a' * 64, [42] * 200000, [], out, helper, capture_arm.run_arm, summary, reference)
                self.assertEqual(len(processes), 1)
                return
            results = runner.paired_requests(self.spec, 'a' * 64, [42] * 200000, [], out, helper, capture_arm.run_arm, summary, reference)
        self.assertEqual(len(processes), 2)
        self.assertEqual(summary['arms_started'], ['control', 'diagnostic'])
        self.assertEqual(summary['arms_completed'], ['control', 'diagnostic'])
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertEqual(request.count('GEN '), 1)
            self.assertTrue(request.startswith('GEN 256 temperature=0 top_p=1 top_k=1 seed=12345 '))
            self.assertEqual(request.count('QUIT\n'), 1)
        for i, command in enumerate(commands):
            flags = dict(command[k + 1].split('=', 1) for k, v in enumerate(command) if v == '-e')
            self.assertEqual(flags, runner.make_arms(self.spec, 'a' * 64)[i]['flags'])
            volumes = [command[k + 1] for k, v in enumerate(command) if v == '-v']
            self.assertEqual(sum(v.endswith(':rw') for v in volumes), i)
            self.assertIn(str(runner.R) + ':/src:ro', volumes)
            self.assertEqual(command[command.index('--user') + 1], str(os.geteuid()) + ':' + str(os.getegid()))
            groups = [command[k + 1] for k, v in enumerate(command) if v == '--group-add']
            self.assertEqual(groups, ['video'] + [str(gid) for gid in sorted(set(os.getgroups()))])
            self.assertEqual(command[-len(self.spec['native_args']):], self.spec['native_args'])
            self.assertIn('--network', command)
            self.assertEqual(command[command.index('--network') + 1], 'none')
            self.assertNotIn('--read-only', command)
        runner.private_tree(out)
        for tag in ['control', 'diagnostic']:
            manifest = json.loads((out / 'private' / tag / (tag + '.manifest.json')).read_text())
            self.assertEqual(manifest['container_identity'], {'uid': os.geteuid(), 'gid': os.getegid(), 'supplementary_group_ids': sorted(set(os.getgroups())), 'original_named_group_add': ['video']})
            private = out / 'private' / tag / 'private'
            self.assertEqual(len(json.loads((private / (tag + '.output-ids.json')).read_text())), 256)
            self.assertTrue((private / (tag + '.done.txt')).read_text().startswith('DONE 256 200000 '))
        return results

    def test_mocked_native_two_processes_one_request_each(self):
        control, diagnostic = self.exercise_pair()
        result = runner.counter_comparison(control, diagnostic)
        runner.require_equal_counters(result)
        self.assertTrue(result['all_equal'])
        self.assertNotIn('output_ids', result['control'])

    def test_control_quality_failure_stops_before_capture(self):
        self.exercise_pair(mismatch=True)

    def test_counter_divergence_pauses_without_retry(self):
        control, diagnostic = self.exercise_pair()
        diagnostic['ram_blobs'] += 1
        diagnostic['suffix_drafts']['accepted'] += 1
        compared = runner.counter_comparison(control, diagnostic)
        self.assertEqual(compared['decision'], 'PAUSE_COUNTER_DIVERGENCE')
        self.assertEqual(compared['diagnostic_minus_control']['ram_blobs'], 1)
        self.assertEqual(compared['diagnostic_minus_control']['suffix_accepted'], 1)
        with self.assertRaisesRegex(RuntimeError, 'PAUSE_COUNTER_DIVERGENCE'):
            runner.require_equal_counters(compared)
        self.assertIsNone(compared['diagnostic_minus_control']['decode_windows'])

    def test_imported_validator_rejects_old_64k(self):
        paths = []
        for device in [0, 1]:
            path = self.path / ('old' + str(device) + '.bin')
            path.write_bytes(fixture(device=device, pos=65536, T=4))
            paths.append(path)
        with self.assertRaisesRegex(ValueError, 'old64K rejected'):
            validate_pair(paths)

    def test_malformed_done_retains_raw_ids_and_done_no_retry(self):
        self.exercise_pair(invalid_done=True)

    def test_preexisting_capture_stops_before_diagnostic(self):
        self.exercise_pair(nonempty_capture=True)

    def package_path(self):
        package = {'runner_file_sha256': {name: runner.sha(HERE / name) for name in runner.DEPENDENCIES}}
        path = self.path / 'package.json'
        path.write_text(json.dumps(package))
        return path, package

    def test_default_cli_is_strictly_offline(self):
        path, package = self.package_path()
        output = io.StringIO()
        with patch('sys.argv', ['run_paired_200k.py', '--package', str(path)]), patch.object(runner, 'execute', side_effect=AssertionError('execution prohibited')), patch.object(runner.subprocess, 'Popen', side_effect=AssertionError('no subprocess')), patch.object(runner.subprocess, 'run', side_effect=AssertionError('no subprocess')), redirect_stdout(output):
            runner.main()
        result = json.loads(output.getvalue())
        self.assertFalse(result['executed'])
        self.assertEqual((result['processes_if_executed'], result['requests_per_process'], result['retries']), (2, 1, 0))
        package['runner_file_sha256']['existing_correctness.py'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'dependency bytes changed'):
            runner.validate_dependencies(package)

    def test_execute_cli_propagates_pause_failure_without_retry(self):
        path, package = self.package_path()
        with patch('sys.argv', ['run_paired_200k.py', '--package', str(path), '--execute', '--binary-sha256', 'a' * 64]), patch.object(runner, 'LOCKS', [str(self.path / 'lock1'), str(self.path / 'lock2')]), patch.object(runner, 'execute', side_effect=RuntimeError('PAUSE_COUNTER_DIVERGENCE')) as run:
            with self.assertRaisesRegex(RuntimeError, 'PAUSE_COUNTER_DIVERGENCE'):
                runner.main()
        self.assertEqual(run.call_count, 1)

    def test_image_toolchain_env_allowed_runtime_flags_fail_closed(self):
        flags = runner.make_arms(self.spec, 'a' * 64)[0]['flags']
        fake = Mock()
        fake.call.return_value = json.dumps(['HIP_PATH=/opt/rocm', 'HIP_PLATFORM=amd'])
        runner.image_environment(fake, flags)
        for env in [['STRATA_GFX906_TOPK_WIDE51=1'], ['STRATA_TOPK_OLD=1'], ['STRATA_VERIFY_PROFILE=1'], ['ROCR_VISIBLE_DEVICES=0'], ['HIP_VISIBLE_DEVICES=1,0']]:
            fake.call.return_value = json.dumps(env)
            with self.assertRaisesRegex(RuntimeError, 'unexpected image environment'):
                runner.image_environment(fake, flags)

    def test_private_file_mode_gate_and_nonempty_capture(self):
        path = self.path / 'bad'
        path.write_text('x')
        path.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, '0700/0600'): runner.private_tree(self.path)


if __name__ == '__main__':
    unittest.main()
