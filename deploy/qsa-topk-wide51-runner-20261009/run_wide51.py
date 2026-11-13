#!/usr/bin/env python3
"""Bounded 42-process WIDE51 correctness runner; no GPU action without --execute."""
import argparse
import collections
import fcntl
import hashlib
import json
from pathlib import Path
import re
import signal

HERE = Path(__file__).resolve().parent
HELPERS = {
    'private_parser_support.py': '95579778a7162c1c13ac730eca3c04d47e5dbd61d7304d1b1f0afef99b07d374',
    'qsa_probe_safety.py': '749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45',
}
for filename, expected in HELPERS.items():
    if hashlib.sha256((HERE/filename).read_bytes()).hexdigest() != expected:
        raise RuntimeError('vetted helper hash mismatch: '+filename)
import private_parser_support as P
S = P.S
PREFIX = 'strata-wide51-'
LOCK = '/tmp/strata-direct-ff34-ce.lock'
CANDIDATE = '98f30cd9fb2fd80619a8caf70c6c1d74bb5c7c14d50ca312b36158cd9518ae08'
BASELINE = '5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a'
MANIFEST = '82e1d2be8927458a1f8040afca379f858e060933cef6955a07881108fd151773'
MANUAL = 'eaa6954d12312b3c8048056b3b999df7ce78d335dbaa4f036eca92f07939853a'
RESOURCE = 'e25b59127276328bcfdc768fd1051db758f11507aacb4e651001c3cbdad1c13f'
ABSENT = ('STRATA_TOPK_OLD', 'STRATA_TOPK_TRACE', 'STRATA_GFX906_TOPK_FIT17_TRACE')

def fixture_file(root, name):
    file = root/name
    S.need(not Path(name).is_absolute() and file.resolve().is_relative_to(root.resolve()), 'fixture path escape')
    return file

def route(fixture, case, env):
    if fixture['nq'] > 8:
        return 'fallback_ref'
    if case['nkv'][-1] <= 24576:
        return 'guarded_ref'
    if env['STRATA_GFX906_TOPK_FIT17'] != '1':
        return 'guarded_reg66_legacy'
    if case['max_active_blocks'] <= 17408:
        return 'guarded_reg17'
    return 'guarded_reg51' if env['STRATA_GFX906_TOPK_WIDE51'] == '1' and fixture['capacity'] <= 52224 else 'guarded_reg66'

def validate_manifest(m):
    S.need(m.get('gpu_executed') is False and m.get('timing_claim') is False, 'fixture scope')
    fixtures = m['fixtures']
    S.need([(f['capacity'], f['nq']) for f in fixtures] ==
           [(cap, nq) for cap in (52223, 52224, 52225) for nq in (1, 4, 6, 8, 9)], 'exact 15 fixtures required')
    byfile = {f['file']: f for f in fixtures}
    S.need(len(byfile) == 15, 'duplicate fixture')
    for f in fixtures:
        S.need(len(f['cases']) == (22 if f['nq'] == 1 else 30), 'fixture case count')
        for c in f['cases']:
            S.need(len(c['nkv']) == f['nq'] and all(1 <= n <= 262144 and n//4+1 <= f['capacity'] for n in c['nkv']), 'row capacity contract')
            S.need(c['max_active_blocks'] == max(n//4+1 for n in c['nkv']), 'active-block metadata')
    plans = m['future_process_plan']
    S.need(len(plans) == 21 and len({p['process_id'] for p in plans}) == 21, 'exact 21 unique processes required')
    expected = [(f['file'], '1', '1') for f in fixtures]
    control = next(f for f in fixtures if f['capacity'] == 52224 and f['nq'] == 4)
    expected += [(control['file'], fit, wide) for fit, wide in
                 [('1', None), ('1', '0'), ('1', '2'), ('1', '01'), (None, '1'), ('0', '1')]]
    actual = []
    for p in plans:
        env = p['env']
        S.need(set(env) == set(ABSENT) | {'STRATA_GFX906_TOPK_REG', 'STRATA_GFX906_TOPK_FIT17', 'STRATA_GFX906_TOPK_WIDE51'}, 'unexpected environment control')
        S.need(env['STRATA_GFX906_TOPK_REG'] == '2' and all(env[k] is None for k in ABSENT), 'REG/trace/OLD contract')
        f = byfile[p['fixture']]
        S.need(p['fixture_sha256'] == f['sha256'], 'process fixture binding')
        S.need(p['expected_routes'] == {c['name']: route(f, c, env) for c in f['cases']}, 'source-model routes changed')
        S.need(p['separate_os_process_required'] is True and p['constant_addresses'] is True and
               p['constant_capacity'] is True and p['graph_instantiations'] == 1, 'graph/process contract')
        actual.append((p['fixture'], env['STRATA_GFX906_TOPK_FIT17'], env['STRATA_GFX906_TOPK_WIDE51']))
    S.need(actual == expected, 'unapproved process matrix')
    return byfile

def verify_receipt_hashes(a):
    for file, supplied, pinned, label in [
        (a.manual_review, a.manual_sha, MANUAL, 'manual review'),
        (a.resource_gate, a.resource_sha, RESOURCE, 'resource gate'),
        (a.fixtures/'manifest.json', a.manifest_sha, MANIFEST, 'fixture manifest')]:
        S.need(supplied == pinned and S.sha(file) == supplied, label+' hash mismatch')
    S.need(S.sha(a.binary) == a.binary_sha, 'binary hash mismatch')

def gate(a):
    verify_receipt_hashes(a)
    manual = json.loads(a.manual_review.read_text())
    resource = json.loads(a.resource_gate.read_text())
    S.need(manual.get('decision') == 'PASS' and manual.get('scope') == 'pre-GPU only' and not manual.get('blockers'), 'manual ISA PASS required')
    S.need(manual['candidate_source']['sha256'] == CANDIDATE and manual['baseline_source_sha256'] == BASELINE, 'manual source identity')
    S.need(manual['bound_artifact_sha256']['resource-gate.json'] == a.resource_sha, 'manual/resource binding')
    S.need(manual['manual_guard_review']['both_guard_exits_precede_every_barrier'] is True and
           manual['manual_guard_review']['all_guard_instructions_scalar'] is True, 'manual uniform pre-barrier guards')
    S.need(manual['elf_review']['unchanged_kernel_count'] == 13 and
           manual['elf_review']['all_existing_function_bytes_identical'] is True, 'manual unchanged kernels')
    S.need(resource.get('passed') is True and not resource.get('failures'), 'automated resource PASS required')
    old = resource['unchanged_kernels']
    S.need(len(old) == 13 and all(all(r[k] is True for k in ['metadata_equal', 'machine_code_equal', 'remarks_equal']) for r in old.values()), 'automated unchanged kernels')
    S.need(resource['candidate_wide51']['private_segment_fixed_size'] == 852 and
           resource['candidate_wide51']['wavefront_size'] == 64, 'reviewed resource identity')
    m = json.loads((a.fixtures/'manifest.json').read_text())
    byfile = validate_manifest(m)
    for f in m['fixtures']:
        file = fixture_file(a.fixtures, f['file'])
        S.need(file.stat().st_size == f['bytes'] and S.sha(file) == f['sha256'], 'fixture hash/size mismatch')
    return m, byfile

def parse_transition(text, stderr, fixture, plan):
    expected = []
    for c in fixture['cases']:
        expected.append(f"CASE_PASS nq={fixture['nq']} case={c['name']} last_nkv={c['nkv'][-1]} max_active_blocks={c['max_active_blocks']} source_expected_route={plan['expected_routes'][c['name']]} route_observed=0 IDs=graph=reference=CPU padding=end_canaries=inputs_unchanged=PASS")
    env = plan['env']
    fit = env['STRATA_GFX906_TOPK_FIT17'] if env['STRATA_GFX906_TOPK_FIT17'] is not None else '<unset>'
    wide = env['STRATA_GFX906_TOPK_WIDE51'] if env['STRATA_GFX906_TOPK_WIDE51'] is not None else '<unset>'
    expected.append(f"TRANSITION_PASS nq={fixture['nq']} capacity={fixture['capacity']} cases={len(fixture['cases'])} graph_instantiations=1 constant_addresses=1 constant_capacities=1 fit17={fit} wide51={wide} route_observed=0 timing_claim=0")
    S.need(text.splitlines() == expected, 'missing/duplicate/malformed/out-of-order CASE_PASS or exact completion')
    S.need('strata gfx906 topk fit17:' not in stderr and 'strata topk fit17' not in stderr, 'forbidden legacy topology trace')
    S.need(not re.search(r'(^|\n)\s*(FAIL|ERROR)\b', stderr, re.I), 'driver error in stderr')
    return {'cases': len(fixture['cases']), 'parity_passed': True, 'route_observed': False,
            'source_expected_route_counts': dict(collections.Counter(plan['expected_routes'].values())),
            'timing_claim': False}

def command(a, plan, device, name):
    cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
           '--device', '/dev/kfd', '--device', '/dev/dri', '--group-add', 'video', '--ipc', 'host',
           '-e', 'HIP_VISIBLE_DEVICES=0,1', '-v', str(a.binary.parent)+':/work:ro',
           '-v', str(a.fixtures)+':/fixture:ro', '--entrypoint', '/usr/bin/env', S.IMAGE]
    # Reset every controlled variable before assigning explicit exact strings.
    for key in (*ABSENT, 'STRATA_VERIFY_PROFILE', 'STRATA_GFX906_TOPK_REG',
                'STRATA_GFX906_TOPK_FIT17', 'STRATA_GFX906_TOPK_WIDE51'):
        cmd += ['-u', key]
    for key, value in plan['env'].items():
        if value is not None:
            cmd.append(key+'='+value)
    cmd += ['/work/'+a.binary.name, '/fixture/'+plan['fixture'], str(device)]
    return cmd

def execute(a):
    S.need(a.leave_api_stopped and a.binary.resolve().is_relative_to(S.R), 'offline binary scope')
    S.need(not a.output.exists(), 'output already exists')
    m, byfile = gate(a)
    S.need(not S.call(['docker', 'ps', '-q']).strip(), 'API and all containers must already be stopped')
    ds = S.devices()
    power = S.gpu(ds)
    a.output.mkdir(mode=0o700)
    out = a.output
    summary = {'scope': '42 sequential WIDE51 correctness processes only', 'started_utc': S.stamp(),
        'binary_sha256': a.binary_sha, 'manual_sha256': a.manual_sha, 'resource_sha256': a.resource_sha,
        'manifest_sha256': a.manifest_sha, 'processes': [], 'power_before': power,
        'timing_claim': False, 'route_observed': False, 'api_start_stop_actions': False,
        'qualification_completed': False, 'cleanup': {'passed': False}}
    S.write(out/'summary.json', summary)
    names = []
    failure = None
    def stop(signum, frame):
        raise InterruptedError('requested stop')
    old = {sig: signal.signal(sig, stop) for sig in [signal.SIGINT, signal.SIGTERM]}
    try:
        for device in (0, 1):
            for plan in m['future_process_plan']:
                verify_receipt_hashes(a)
                f = byfile[plan['fixture']]
                S.need(S.sha(fixture_file(a.fixtures, f['file'])) == f['sha256'], 'fixture changed before run')
                S.need(not S.call(['docker', 'ps', '-q']).strip(), 'another container became active')
                before = S.cooldown(ds, out)
                tag = f'g{device}-'+plan['process_id']
                name = PREFIX+tag
                names.append(name)
                row = {'tag': tag, 'device': device, 'process_id': plan['process_id'], 'env': plan['env'],
                       'fixture_sha256': f['sha256'], 'gpu_before': before, 'status': 'running'}
                summary['processes'].append(row)
                S.write(out/'summary.json', summary)
                try:
                    driver = P.launch(command(a, plan, device, name), name, ds, out, tag, 180)
                    row['driver'] = driver
                    row['exit_code'] = driver.get('exit_code')
                    S.need(row['exit_code'] == 0, 'process did not exit zero')
                    row.update(parse_transition((out/(tag+'.stdout')).read_text(),
                        (out/(tag+'.stderr')).read_text(), f, plan))
                    row['status'] = 'passed'
                except BaseException as error:
                    row.update(status='failed', error_type=type(error).__name__, error=str(error))
                    receipt = out/(tag+'.driver.json')
                    if receipt.exists():
                        row['driver'] = json.loads(receipt.read_text())
                        row['exit_code'] = row['driver'].get('exit_code')
                    raise
                finally:
                    S.write(out/'summary.json', summary)
        S.need(len(summary['processes']) == 42 and all(r['status'] == 'passed' for r in summary['processes']), 'exact 42-process completion')
        summary['qualification_completed'] = True
    except BaseException as error:
        failure = error
        summary.update(error_type=type(error).__name__, error=str(error), qualification_completed=False)
    finally:
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        try:
            for name in names:
                S.cleanup(name)
            S.need(not S.call(['docker', 'ps', '-q']).strip(), 'unexpected active container after cleanup')
            after = S.gpu(ds)
            S.need([(r['pci'], r['mode'], r['power1_cap']) for r in after] ==
                   [(r['pci'], r['mode'], r['power1_cap']) for r in power], 'power settings changed')
            summary['cleanup'] = {'passed': True, 'api_left_stopped': True, 'power_after': after}
        except BaseException as error:
            summary['cleanup'] = {'passed': False, 'error_type': type(error).__name__, 'error': str(error)}
            summary['qualification_completed'] = False
        summary['finished_utc'] = S.stamp()
        S.write(out/'summary.json', summary)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    S.need(summary['cleanup']['passed'], 'cleanup failed')
    if failure:
        raise failure
    print(json.dumps({'qualification_completed': True, 'processes': 42,
                      'timing_claim': False, 'route_observed': False, 'api_start_stop_actions': False}))

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'manual-review', 'resource-gate', 'fixtures', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('binary-sha', 'manual-sha', 'resource-sha', 'manifest-sha'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--leave-api-stopped', action='store_true')
    a = p.parse_args(argv)
    S.need(all(getattr(a, name).is_absolute() for name in ('binary', 'manual_review', 'resource_gate', 'fixtures', 'output')), 'absolute paths required')
    S.need(all(re.fullmatch('[0-9a-f]{64}', getattr(a, name)) for name in ('binary_sha', 'manual_sha', 'resource_sha', 'manifest_sha')), 'four explicit SHA256 values required')
    S.need(a.leave_api_stopped, 'offline only; --leave-api-stopped required')
    if not a.execute:
        print('arguments valid; not executed')
        return
    S.need(__debug__, 'python -O forbidden')
    with open(LOCK, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(a)

if __name__ == '__main__':
    main()
