#!/usr/bin/env python3
"""Prepare/check by default. --execute runs exactly one c183 control and one diagnostic.
No build, retry, API request, promotion, flag search, or performance claim.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys

HERE = Path(__file__).resolve().parent
R = Path('/home/radneon/strata')
CAMPAIGN = R / 'logs/perf-campaign-20261009'
BASE = CAMPAIGN / 'qsa-topk-fit17-model'
BUILD = CAMPAIGN / 'qsa-200k-capture-build'
OUT = CAMPAIGN / 'qsa-200k-capture-run'
BASE_SHA = 'c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049'
REF_SHA = '13b8fd8b67d24cc65a027c990281f8d19696d0b77a72a75316638a438737fb9d'
HELPER_SHA = '850a18e95a5a2e13013f778ecb7b807809def5e62154fe9ff91acfe63b8f05bf'
SPEC_SHA = 'ff7f197850c35da8eea4a854e6959ea15f13226db9e331f0bef4d1713a200b6b'
FIXTURE_SHA = '736636e97ede94cc37e32447eefde4217a38bd17c81a7349345be98060e766c4'
IMAGE = 'sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
FLAG = 'STRATA_GFX906_TOPK_FIT17'
TRACE = FLAG + '_TRACE'
DUMP = 'STRATA_QSA_SELECT_DUMP'
COUNTERS = ['actual_outputs', 'prompt_tokens', 'accepted', 'offered', 'cache_hits', 'lookups', 'prompt_reused',
            'prompt_read', 'ram_blobs', 'file_blobs', 'file_read_mb',
            'offloaded_experts', 'lookup_chain_accepted', 'lookup_chain_offered',
            'native_suffix_accepted', 'native_suffix_offered', 'decode_windows',
            'native_done_field_count']
LOCKS = ['/tmp/strata-direct-ff34-ce.lock', '/tmp/strata-qsa-fit17-qualification.lock']
DEPENDENCIES = ['run_paired_200k.py', 'capture_arm.py', 'existing_correctness.py',
                'correctness.spec.json', 'validate_200k.py', 'analyze_qsa_dump.py', 'build_capture.py', 'access_preflight.py', 'recovery_support.py', 'original_package.v2.json']


def need(ok, label):
    if not ok:
        raise RuntimeError(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_module(name):
    path = HERE / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_dependencies(package):
    pins = package['runner_file_sha256']
    need(isinstance(pins, dict) and set(DEPENDENCIES) <= set(pins), 'all runner dependencies pinned')
    for filename in DEPENDENCIES:
        need(digest(pins[filename]), 'dependency digest required')
        path = HERE / filename
        need(path.is_file() and not path.is_symlink() and sha(path) == pins[filename], 'dependency bytes changed: ' + filename)
    need(pins['existing_correctness.py'] == HELPER_SHA, 'trusted helper must remain byte-identical')
    need(pins['correctness.spec.json'] == SPEC_SHA, 'original qualified spec must remain byte-identical')


def validate_config(spec):
    need(spec['study_binary_sha256'] == BASE_SHA, 'exact c183 control')
    need(spec['image'] == IMAGE, 'exact native image')
    need(spec['binary_path'] == str(BASE / 'strata'), 'control binary path')
    need(spec['reference_path'] == str(BASE / 'correctness-references.json') and spec['reference_sha256'] == REF_SHA, 'qualified reference identity')
    need(canonical_sha(spec['ff34_flags']) == 'efb2e4020356ba79f1f2624197b05cf0a3b45ba42dd692ca952b82ba71639474', 'exact flags; hidden flags prohibited')
    need(canonical_sha(spec['native_args']) == 'd94b0bb9ffb0463dcce52afe87bb57d8e8ea3c2d037ca2022b254b77b099d953', 'exact native args; hidden args prohibited')
    case = next((c for c in spec['cases'] if c['tag'] == 'code200000'), None)
    need(case == {'tag': 'code200000', 'prompt_tokens': 200000, 'generate': 256,
                  'fixture_path': str(CAMPAIGN / 'qsa-reduce12-model/qualification/prompt-200000.ids.json'),
                  'fixture_sha256': FIXTURE_SHA}, 'exact 200000/256 fixture case')
    return case


def make_arms(spec, diagnostic_sha):
    need(digest(diagnostic_sha) and diagnostic_sha != BASE_SHA, 'explicit new diagnostic binary SHA256')
    validate_config(spec)
    flags = {**spec['ff34_flags'], FLAG: '1', TRACE: '1'}
    need(flags['STRATA_GFX906_TOPK_REG'] == '2', 'TREG2 required')
    return [
        {'tag': 'control', 'binary': str(BASE / 'strata'), 'binary_sha256': BASE_SHA,
         'release_artifacts': {}, 'flags': flags, 'native_args': list(spec['native_args'])},
        {'tag': 'diagnostic', 'binary': str(BUILD / 'strata'), 'binary_sha256': diagnostic_sha,
         'release_artifacts': {}, 'flags': {**flags, DUMP: '/captures/window'},
         'native_args': list(spec['native_args'])},
    ]


def counter_values(result):
    values = {k: result[k] for k in COUNTERS}
    values['optional_chain_counters_available'] = result['optional_chain_counters_available']
    values.update({'suffix_' + k: result['suffix_drafts'][k] for k in ['present', 'windows', 'accepted', 'offered']})
    for key, value in values.items():
        need(value is None or type(value) in (int, float, bool), 'numeric counter type: ' + key)
        need(value is None or (math.isfinite(value) and value >= 0), 'numeric counter domain: ' + key)
    return values


def counter_comparison(control, diagnostic):
    a, b = counter_values(control), counter_values(diagnostic)
    delta = {k: None if a[k] is None or b[k] is None else b[k] - a[k] for k in a}
    differing = [k for k in a if a[k] != b[k]]
    return {'control': a, 'diagnostic': b, 'diagnostic_minus_control': delta,
            'different_fields': differing, 'all_equal': not differing,
            'decision': 'PAUSE_COUNTER_DIVERGENCE' if differing else 'COUNTERS_EQUAL'}


def require_equal_counters(comparison):
    need(comparison['all_equal'], 'PAUSE_COUNTER_DIVERGENCE: paired native/suffix counters differ; no rerun')


def paired_requests(spec, diagnostic_sha, ids, devices, out, helper, run_arm, summary, reference):
    """Fixed two-arm schedule. No attempt loop and no retry path."""
    case = validate_config(spec)
    case_spec = {**spec, 'fixture_sha256': case['fixture_sha256'], 'requested_outputs': 256}
    results = []
    for arm in make_arms(spec, diagnostic_sha):
        tag = arm['tag']
        arm_dir = out / 'private' / tag
        arm_dir.mkdir(mode=0o700)
        capture = out / 'private' / 'captures' if tag == 'diagnostic' else None
        if capture is not None:
            need(capture.is_dir() and not capture.is_symlink() and not any(capture.iterdir()), 'fresh empty private capture directory required')
            need(capture.stat().st_uid == os.geteuid(), 'capture directory owner must match runner/container UID')
        summary['arms_started'].append(tag)
        helper.write(out / 'summary.json', summary)
        result = run_arm(tag, arm, case_spec, ids, devices, arm_dir, capture_dir=capture)
        results.append(result)
        summary['arms_completed'].append(tag)
        summary.setdefault('counters_by_arm', {})[tag] = counter_values(result)
        helper.write(out / 'summary.json', summary)
        helper.quality_check(result, reference)
        result['counter_policy'] = 'paired counter equality required; pause on any delta'
        helper.write(arm_dir / (tag + '.result.json'), result)
    return results


def private_tree(root):
    for path in [root, *root.rglob('*')]:
        st = path.lstat()
        need(not stat.S_ISLNK(st.st_mode), 'symlinks prohibited in private output')
        need(stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode), 'non-regular private output')
        expected = 0o700 if stat.S_ISDIR(st.st_mode) else 0o600
        need(stat.S_IMODE(st.st_mode) == expected, 'private output mode must be 0700/0600')


def no_active_containers(helper):
    need(not helper.call(['docker', 'ps', '-q']).strip(), 'container active; API and other GPU work must remain off')


def image_environment(helper, flags):
    env = json.loads(helper.call(['docker', 'image', 'inspect', IMAGE, '--format', '{{json .Config.Env}}'])) or []
    for entry in env:
        key, _, value = entry.partition('=')
        if key.startswith('STRATA_') or key in {'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'}:
            need(key in flags and value == flags[key], 'unexpected image environment: ' + key)


def check_host_processes(helper):
    # Read-only PID/command inventory: never stop unrelated host processes.
    rows = helper.call(['ps', '-eo', 'pid=,comm=']).splitlines()
    forbidden = {'strata', 'hipcc', 'clang', 'clang++', 'g++', 'cc1plus'}
    need(not any(len(row.split()) == 2 and row.split()[1] in forbidden for row in rows), 'native inference/compiler already running')


def cleanup_owned(helper):
    failures = []
    for tag in ['control', 'diagnostic']:
        name = 'strata-qsa-200k-' + tag
        try:
            if helper.call(['docker', 'ps', '-q', '--filter', 'name=^/' + name + '$']).strip():
                try:
                    helper.call(['docker', 'stop', '--time', '30', name], 60)
                except (OSError, subprocess.SubprocessError):
                    helper.call(['docker', 'kill', name], 30)
                need(not helper.call(['docker', 'ps', '-q', '--filter', 'name=^/' + name + '$']).strip(), 'owned container still active')
        except BaseException:
            failures.append(tag)
    return failures


def execute(package, package_path, diagnostic_sha, recovery_path=None, recovery_sha=None):
    validate_dependencies(package)
    helper = load_module('existing_correctness')
    arm_helper = load_module('capture_arm')
    builder = load_module('build_capture')
    spec = json.loads((HERE / 'correctness.spec.json').read_text())
    case = validate_config(spec)
    arms = make_arms(spec, diagnostic_sha)
    need(not OUT.exists(), 'fresh output required; rerunning this pair is prohibited')
    receipt = json.loads((BUILD / 'build.json').read_text())
    recovery_binding = None
    if recovery_path is None:
        need(recovery_sha is None, 'recovery hash without receipt')
        builder.validate_receipt(receipt, package, diagnostic_sha, sha(package_path))
    else:
        recovery = load_module('recovery_support')
        recovery_binding = recovery.validate(receipt, HERE/'original_package.v2.json', recovery_path, recovery_sha, diagnostic_sha)
    for path, expected in [(BASE / 'strata', BASE_SHA), (BASE / 'build.json', receipt['base_build_json_sha256']),
                           (BUILD / 'strata', diagnostic_sha),
                           (BASE / 'correctness-references.json', REF_SHA),
                           (Path(case['fixture_path']), FIXTURE_SHA),
                           (BUILD / 'qsa_select.cu.o', receipt['select_object_sha256']),
                           (BUILD / 'libstrata_kernels.a', receipt['kernel_archive_sha256'])]:
        need(path.is_file() and not path.is_symlink() and sha(path) == expected, 'input/artifact bytes changed: ' + path.name)
    for filename, expected in receipt['compiled_sources'].items():
        need(sha(BUILD / 'compiled-source' / filename) == expected, 'compiled source snapshot changed')
    refs = json.loads((BASE / 'correctness-references.json').read_text())
    need(refs['binary_sha256'] == helper.BASE_SHA, 'qualified ff34 reference provenance')
    reference = refs['results']['200000']
    need(reference['passed'] and reference['prompt_tokens'] == 200000 and reference['actual_outputs'] == 256 and len(reference['output_ids']) == 256, 'reference coverage')
    ids = json.loads(Path(case['fixture_path']).read_text())
    need(len(ids) == 200000 and all(type(i) is int and 0 <= i < 248320 for i in ids), 'fixture coverage')
    no_active_containers(helper)
    check_host_processes(helper)
    image_environment(helper, arms[0]['flags'])
    preflight = load_module('access_preflight')
    binaries = ['/src/'+str(Path(arm['binary']).relative_to(R)) for arm in arms]
    access = preflight.run(IMAGE, R, CAMPAIGN, spec['native_args'], binaries, arms[0]['flags'])
    no_active_containers(helper)
    devices = helper.devices()
    helper.gpu(devices)
    OUT.mkdir(mode=0o700)
    (OUT / 'private').mkdir(mode=0o700)
    (OUT / 'private/captures').mkdir(mode=0o700)
    summary = {'schema_version': 1, 'purpose': 'qsa-200k-selector-capture',
               'started_utc': helper.stamp(), 'capture_passed': False, 'decision': 'RUNNING',
               'control_binary_sha256': BASE_SHA, 'diagnostic_binary_sha256': diagnostic_sha,
               'package_sha256': sha(package_path), 'build_receipt_sha256': sha(BUILD / 'build.json'),
               'build_recovery': recovery_binding,
               'fixture_sha256': FIXTURE_SHA, 'reference_sha256': REF_SHA,
               'prompt_tokens': 200000, 'requested_outputs': 256,
               'sampling': {'temperature': 0, 'top_p': 1, 'top_k': 1, 'seed': 12345},
               'arms_started': [], 'arms_completed': [], 'maximum_processes': 2,
               'permission_only_preflight_processes': 1, 'access_preflight': access,
               'requests_per_process': 1, 'retries': 0, 'performance_comparison': False,
               'timing_scope': 'diagnostic only; no performance claim',
               'api_touched': False, 'promoted': False, 'wide51_enabled': False}
    helper.write(OUT / 'summary.json', summary)
    def interrupted(signum, frame):
        raise InterruptedError('paired capture interrupted')
    previous = {sig: signal.signal(sig, interrupted) for sig in [signal.SIGINT, signal.SIGTERM]}
    error = None
    try:
        control, diagnostic = paired_requests(spec, diagnostic_sha, ids, devices, OUT, helper, arm_helper.run_arm, summary, reference)
        # Persist complete counter evidence before correctness/capture gates can fail.
        summary['counter_comparison'] = counter_comparison(control, diagnostic)
        helper.write(OUT / 'summary.json', summary)
        for result in [control, diagnostic]:
            helper.quality_check(result, reference)
            result['counter_policy'] = 'paired counter equality required; pause on any delta'
            helper.write(OUT / 'private' / result['tag'] / (result['tag'] + '.result.json'), result)
        need(control['output_ids'] == diagnostic['output_ids'], 'paired full output ID mismatch')
        need(control['fit17_host_topology'] == diagnostic['fit17_host_topology'], 'paired FIT17 topology mismatch')
        summary['all_256_reference_ids_equal_per_arm'] = True
        private_tree(OUT)
        # Import strict 200K validator only after the isolated capture is complete.
        validate_dependencies(package)
        validator = load_module('validate_200k')
        paths = sorted((OUT / 'private/captures').glob('*.bin'))
        need(len(paths) == 2 and set(paths) == set((OUT / 'private/captures').iterdir()), 'exactly two private stage capture files')
        summary['capture_validation'] = validator.validate_pair(paths, {'baseline_sha256': BASE_SHA, 'fixture_sha256': FIXTURE_SHA,
                                                                      'prompt_tokens': 200000, 'requested_outputs': 256})
        summary['capture_files'] = [{'filename': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths]
        helper.write(OUT / 'summary.json', summary)
        require_equal_counters(summary['counter_comparison'])
        summary['capture_passed'] = True
        summary['decision'] = 'PASS_CAPTURE_ONLY'
    except BaseException as caught:
        error = caught
        summary['error_type'] = type(caught).__name__
        summary['error'] = str(caught)
        summary['decision'] = ('PAUSE_COUNTER_DIVERGENCE' if summary.get('counter_comparison', {}).get('all_equal') is False else 'PAUSE_VALIDATION_FAILURE')
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        failures = cleanup_owned(helper)
        summary['owned_containers_cleaned'] = not failures
        summary['cleanup_failed_arms'] = failures
        try:
            no_active_containers(helper)
            summary['all_containers_stopped'] = True
        except BaseException:
            summary['all_containers_stopped'] = False
        if failures or not summary['all_containers_stopped']:
            summary['capture_passed'] = False
            summary['decision'] = 'PAUSE_CLEANUP_FAILURE'
        summary['finished_utc'] = helper.stamp()
        helper.write(OUT / 'summary.json', summary)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    private_tree(OUT)
    print(json.dumps({'capture_passed': summary['capture_passed'], 'decision': summary['decision'],
                      'arms_completed': summary['arms_completed'],
                      'counter_different_fields': summary.get('counter_comparison', {}).get('different_fields', []),
                      'performance_comparison': False, 'api_touched': False}), flush=True)
    if error is not None:
        raise error
    need(summary['capture_passed'], 'PAUSE: capture did not pass final cleanup')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=HERE / 'package.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--binary-sha256')
    parser.add_argument('--restore-receipt', type=Path)
    parser.add_argument('--restore-receipt-sha256')
    args = parser.parse_args()
    package = json.loads(args.package.read_text())
    validate_dependencies(package)
    spec = json.loads((HERE / 'correctness.spec.json').read_text())
    validate_config(spec)
    if not args.execute:
        print(json.dumps({'validated': True, 'executed': False, 'processes_if_executed': 2,
                          'requests_per_process': 1, 'retries': 0, 'prompt_tokens': 200000,
                          'outputs': 256, 'wide51_enabled': False, 'performance_comparison': False}))
        return
    need(args.binary_sha256 is not None, '--execute requires explicit --binary-sha256')
    os.umask(0o077)
    with ExitStack() as stack:
        for path in LOCKS:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            stream = stack.enter_context(os.fdopen(fd, 'r+'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.restore_receipt is None:
            need(args.restore_receipt_sha256 is None, 'recovery hash without receipt')
            execute(package, args.package, args.binary_sha256)
        else:
            need(args.restore_receipt_sha256 is not None, 'explicit recovery receipt hash required')
            execute(package, args.package, args.binary_sha256, args.restore_receipt, args.restore_receipt_sha256)


if __name__ == '__main__':
    main()
