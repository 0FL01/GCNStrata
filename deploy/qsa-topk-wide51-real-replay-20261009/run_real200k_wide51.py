#!/usr/bin/env python3
"""Exactly four correctness processes then eight ABBA processes; --execute required."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal

HERE = Path(__file__).resolve().parent
PINS = {
    'private_parser_support.py': '95579778a7162c1c13ac730eca3c04d47e5dbd61d7304d1b1f0afef99b07d374',
    'qsa_probe_safety.py': '749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45',
    'validate_200k.py': 'e8ce3393b236574b27ae780c65c7ea0f7fc92e1c9659f2fcd33689870eb5762e',
    'analyze_qsa_dump.py': '53ad4c4d185d761aa35656e463d34b07a9c7bc91deeafb0e36e2b8bc3ea3daaa',
}
for name, expected in PINS.items():
    if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('original helper hash mismatch: '+name)
import private_parser_support as P
import validate_200k as V
S = P.S
LOCK = '/tmp/strata-direct-ff34-ce.lock'
PREFIX = 'strata-wide51-real200k-'
RESOURCE_BUILD = 'eb75d9e714d2f73d901f8e3af5303450a1444cb4ae0b4b1c11e1b469f307c80d'
MANUAL = 'eaa6954d12312b3c8048056b3b999df7ce78d335dbaa4f036eca92f07939853a'
RESOURCE = 'e25b59127276328bcfdc768fd1051db758f11507aacb4e651001c3cbdad1c13f'
BOUNDARY = 'd6dc96d487c1345b0bd8956d334e39f5a7f23d4c8abee53e0ee56606c7f79387'
SOURCE = {'fit17': '5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a',
          'wide51': '98f30cd9fb2fd80619a8caf70c6c1d74bb5c7c14d50ca312b36158cd9518ae08'}
OBJECT = {'fit17': '7f2d1cf8b25b795d92863f30fe15cf52898edb80e5c2aeeb4387663eed9ba2d2',
          'wide51': 'a1f628704df0c71221f2e46a932343bdd57129c8990cdc1db68aa5f77161e8b8'}
CONSUMER = '0eac4ceb11f783e14977f35afdbe0bf6c39fbddaa0fe8c39675e4a0f346a667e'
CONTROLLED = ('STRATA_GFX906_TOPK_REG', 'STRATA_GFX906_TOPK_FIT17',
    'STRATA_GFX906_TOPK_WIDE51', 'STRATA_TOPK_TRACE', 'STRATA_TOPK_OLD',
    'STRATA_GFX906_TOPK_FIT17_TRACE', 'STRATA_VERIFY_PROFILE', 'STRATA_QSA_SELECT_DUMP',
    'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')
SCOPE = 'SCOPE warm-cache isolated existing selector including guarded no-op launch; no scorer, no whole-model timing, one captured layer'

def bound(path, expected):
    S.need(path.is_file() and not path.is_symlink() and S.sha(path) == expected, 'file hash mismatch: '+str(path))

def files(a):
    return {a.resource_build: RESOURCE_BUILD, a.manual_review: MANUAL, a.resource_gate: RESOURCE,
        a.boundary_summary: BOUNDARY, a.capture_summary: a.capture_summary_sha,
        a.build/'build.receipt.json': a.build_sha, a.build/'qsa_topk_fixture_replay.cpp': CONSUMER,
        a.build/'qsa_select.fit17.o': OBJECT['fit17'], a.build/'qsa_select.wide51.o': OBJECT['wide51'],
        a.build/'replay.fit17': a.baseline_sha, a.build/'replay.wide51': a.candidate_sha,
        a.real0: a.real0_sha, a.real1: a.real1_sha}

def rehash(a):
    for path, expected in files(a).items():
        bound(path, expected)
    for name, expected in PINS.items():
        bound(HERE/name, expected)

def gate(a):
    rehash(a)
    resource_build = json.loads(a.resource_build.read_text())
    S.need(resource_build['compiled'] and resource_build['resource_isa_gate_passed'] and
           resource_build['support_archive_used'] is False, 'reviewed selector build required')
    for variant in SOURCE:
        S.need(resource_build['files']['qsa_select.'+variant+'.cu'] == SOURCE[variant] and
               resource_build['files']['qsa_select.'+variant+'.o'] == OBJECT[variant], 'reviewed object/source binding')
    manual = json.loads(a.manual_review.read_text())
    S.need(manual['decision'] == 'PASS' and not manual['blockers'] and
           manual['bound_artifact_sha256']['build.receipt.json'] == RESOURCE_BUILD and
           manual['bound_artifact_sha256']['resource-gate.json'] == RESOURCE, 'manual review binding')
    resource = json.loads(a.resource_gate.read_text())
    S.need(resource['passed'] and not resource['failures'], 'resource PASS required')
    boundary = json.loads(a.boundary_summary.read_text())
    rows = boundary['processes']
    S.need(boundary['qualification_completed'] and boundary['cleanup']['passed'] and
           boundary['manual_sha256'] == MANUAL and boundary['resource_sha256'] == RESOURCE and
           len(rows) == 42 and sum(r['cases'] for r in rows) == 1212 and
           all(r['status'] == 'passed' and r['parity_passed'] and r['exit_code'] == 0 for r in rows),
           'all 42 prior boundary processes and 1212 graph cases required')
    build = json.loads((a.build/'build.receipt.json').read_text())
    S.need(build['passed'] is True and build['exit_code'] == 0 and build['gpu_executed'] is False and
           build['support_archive_used'] is False and
           build['resource_build_sha256'] == RESOURCE_BUILD, 'direct reviewed-object replay build required')
    for name, expected in [('qsa_topk_fixture_replay.cpp', CONSUMER),
            ('qsa_select.fit17.o', OBJECT['fit17']), ('qsa_select.wide51.o', OBJECT['wide51']),
            ('replay.fit17', a.baseline_sha), ('replay.wide51', a.candidate_sha)]:
        S.need(build['files'][name] == expected, 'replay build binding: '+name)
    capture = json.loads(a.capture_summary.read_text())
    S.need(capture['capture_passed'] is True and capture['decision'] == 'PASS_CAPTURE_ONLY' and
           capture['owned_containers_cleaned'] is True and capture['all_containers_stopped'] is True and
           not capture['cleanup_failed_arms'] and capture['counter_comparison']['all_equal'] is True,
           'qualified capture pair and successful cleanup required')
    S.need(capture['control_binary_sha256'] == V.BASELINE_SHA and capture['fixture_sha256'] == V.FIXTURE_SHA and
           capture['prompt_tokens'] == 200000 and capture['requested_outputs'] == 256 and
           capture['all_256_reference_ids_equal_per_arm'] is True, 'private native capture provenance')
    provenance = dict(baseline_sha256=V.BASELINE_SHA, fixture_sha256=V.FIXTURE_SHA,
                      prompt_tokens=200000, requested_outputs=256)
    analyzed = V.validate_pair([a.real0, a.real1], provenance)
    captures = {r['filename']: (r['sha256'], r['bytes']) for r in capture['capture_files']}
    S.need(len(captures) == 2, 'two captured stages required')
    for device, path in enumerate([a.real0, a.real1]):
        row = analyzed['files'][device]
        S.need(row['header']['device'] == device and row['all_ids_match'] is True,
               'fixture must replay on its captured physical device')
        S.need(captures.get(path.name) == (row['sha256'], row['bytes']), 'capture summary hash/size binding')
    return analyzed

def parse_result(text, stderr, h, device, batch, samples):
    lines = text.splitlines()
    S.need(len(lines) == 5 and lines[0] == 'PARITY_PASS production=reference=captured; padding/end_canaries=PASS'
           and lines[4] == SCOPE, 'exact consumer completion and parity required')
    prefix = (f"META captured_device={h['device']} replay_device={device} layer={h['layer']} "
        f"stage=[{h['layer_begin']},{h['layer_end']}) pos0={h['pos0']} nq={h['T']} "
        f"capacity_blocks={h['max_blocks']} cap={h['cap']} batch={batch} samples={samples} device=")
    S.need(lines[1].startswith(prefix) and len(lines[1]) > len(prefix), 'header-driven exact metadata required')
    S.need(lines[2].startswith('SAMPLES_MS_PER_CALL '), 'one sample record required')
    values = list(map(float, lines[2].split()[1:]))
    S.need(len(values) == samples and all(math.isfinite(x) and x > 0 for x in values), 'finite positive sample count')
    result = re.fullmatch(r'RESULT median_ms_per_call=([0-9.eE+-]+) min=([0-9.eE+-]+) max=([0-9.eE+-]+) input_bytes_unchanged=PASS graph_ids=PASS', lines[3])
    S.need(result is not None, 'final graph/input/canary parity required')
    median = sorted(values)[len(values)//2]
    reported = list(map(float, result.groups()))
    S.need(all(math.isfinite(v) and abs(v-w) < 1e-7 for v,w in zip(reported, [median,min(values),max(values)])), 'inconsistent result statistics')
    S.need(not re.search(r'(^|\n)\s*(FAIL|ERROR)\b', stderr, re.I), 'driver error in stderr')
    S.need('strata gfx906 topk fit17:' not in stderr and 'strata topk fit17' not in stderr,
           'legacy route tracing must remain absent')
    return {'median_ms_per_call': median, 'samples_ms': values, 'all_parity_passed': True,
            'route_observed': False, 'captured_header': h}

def schedule():
    rows = []
    for device in (0, 1):
        for variant in ('fit17', 'wide51'):
            rows.append(dict(device=device, variant=variant, phase='correctness', label=variant, batch=1, samples=1))
    for device in (0, 1):
        for label, variant in [('A1','fit17'),('B1','wide51'),('B2','wide51'),('A2','fit17')]:
            rows.append(dict(device=device, variant=variant, phase='abba', label=label, batch=64, samples=7))
    return rows

def command(a, plan, name):
    fixture = [a.real0, a.real1][plan['device']]
    cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
        '--device', '/dev/kfd', '--device', '/dev/dri', '--group-add', 'video', '--ipc', 'host',
        '-v', str(a.build)+':/work:ro', '-v', str(fixture.parent)+':/fixture:ro',
        '--entrypoint', '/usr/bin/env', S.IMAGE]
    for key in CONTROLLED:
        cmd += ['-u', key]
    cmd += ['HIP_VISIBLE_DEVICES=0,1', 'STRATA_GFX906_TOPK_REG=2', 'STRATA_GFX906_TOPK_FIT17=1', 'STRATA_TOPK_TRACE=1']
    if plan['variant'] == 'wide51':
        cmd += ['STRATA_GFX906_TOPK_WIDE51=1']
    return cmd + ['/work/replay.'+plan['variant'], '/fixture/'+fixture.name, str(plan['device']), str(plan['batch']), str(plan['samples'])]

def exclusive():
    S.need(not S.call(['docker','ps','-q']).strip(), 'all containers/API must already be stopped')
    forbidden = {'strata', 'hipcc', 'clang', 'clang++', 'g++', 'cc1plus', 'replay.fit17', 'replay.wide51'}
    rows = S.call(['ps','-eo','comm=']).splitlines()
    S.need(not any(row.strip() in forbidden for row in rows), 'native inference/replay/compiler already running')

def score(rows, device):
    S.need([r['label'] for r in rows] == ['A1','B1','B2','A2'] and
           [r['variant'] for r in rows] == ['fit17','wide51','wide51','fit17'], 'one exact ABBA sequence required')
    aa = [rows[i]['median_ms_per_call'] for i in (0,3)]
    bb = [rows[i]['median_ms_per_call'] for i in (1,2)]
    return {'device':device, 'baseline_run_medians_ms':aa, 'candidate_run_medians_ms':bb,
        'baseline_mean_of_medians_ms':sum(aa)/2, 'candidate_mean_of_medians_ms':sum(bb)/2,
        'latency_reduction_pct':100*(1-sum(bb)/sum(aa)), 'all_B_faster_than_all_A':max(bb)<min(aa),
        'route_observed':False, 'model_pp_tg_gain_claim':False, 'predeclared_speed_threshold':None}

def execute(a):
    S.need(a.leave_api_stopped and a.build.resolve().is_relative_to(S.R), 'offline build scope')
    S.need(not a.output.exists(), 'fresh output required; this bounded run cannot resume/retry')
    analyzed = gate(a)
    exclusive()
    ds = S.devices()
    power = S.gpu(ds)
    a.output.mkdir(mode=0o700)
    out = a.output
    summary = {'scope':'real-200K warm-cache isolated public selector replay; two captured layers only',
        'started_utc':S.stamp(), 'bindings':{str(k):v for k,v in files(a).items()},
        'runner_sha256':S.sha(Path(__file__)), 'helper_sha256':PINS, 'cpu_validation':analyzed,
        'power_before':power, 'processes':[], 'components':[], 'maximum_processes':12,
        'correctness_processes':4, 'latency_processes':8, 'retries':0, 'promoted':False,
        'api_start_stop_actions':False, 'route_observed':False, 'model_pp_tg_gain_claim':False,
        'all_correctness_passed':False, 'qualification_completed':False, 'cleanup':{'passed':False}}
    S.write(out/'summary.json', summary)
    names = []
    failure = None
    def stop(signum, frame):
        raise InterruptedError('requested stop')
    old = {sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for plan in schedule():
            if plan['phase'] == 'abba':
                S.need(summary['all_correctness_passed'], 'all four correctness processes required before any ABBA')
            rehash(a)
            exclusive()
            before = S.cooldown(ds,out)
            # Cooldown can wait fifteen minutes. Recheck after that wait, at the launch boundary.
            rehash(a)
            exclusive()
            tag = f"{plan['phase']}-g{plan['device']}-{plan['label']}"
            name = PREFIX+tag.lower()
            names.append(name)
            row = dict(plan, tag=tag, gpu_before=before, status='running',
                source_expected_route='guarded_reg51' if plan['variant']=='wide51' else 'guarded_reg66')
            summary['processes'].append(row)
            S.write(out/'summary.json',summary)
            try:
                driver = P.launch(command(a,plan,name),name,ds,out,tag,180)
                row['driver'] = driver
                S.need(driver.get('exit_code') == 0, 'process exited nonzero')
                result = parse_result((out/(tag+'.stdout')).read_text(),(out/(tag+'.stderr')).read_text(),
                    analyzed['files'][plan['device']]['header'],plan['device'],plan['batch'],plan['samples'])
                if plan['phase'] == 'correctness':
                    result.pop('median_ms_per_call')
                    result.pop('samples_ms')
                    result['timing_claim'] = False
                row.update(result,status='passed')
            except BaseException as error:
                row.update(status='failed',error_type=type(error).__name__,error=str(error))
                raise
            finally:
                S.write(out/'summary.json',summary)
            if len(summary['processes']) == 4:
                summary['all_correctness_passed'] = all(r['status']=='passed' and r['all_parity_passed'] for r in summary['processes'])
            if plan['phase']=='abba' and plan['label']=='A2':
                rows = [r for r in summary['processes'] if r['device']==plan['device'] and r['phase']=='abba']
                summary['components'].append(score(rows,plan['device']))
            S.write(out/'summary.json',summary)
        S.need(len(summary['processes'])==12 and len(summary['components'])==2, 'exact bounded completion')
        summary['qualification_completed'] = True
    except BaseException as error:
        failure = error
        summary.update(error_type=type(error).__name__,error=str(error),qualification_completed=False)
    finally:
        for sig in old:
            signal.signal(sig,signal.SIG_IGN)
        try:
            for name in names:
                S.cleanup(name)
            exclusive()
            after = S.gpu(ds)
            S.need([(r['pci'],r['mode'],r['power1_cap']) for r in after] ==
                   [(r['pci'],r['mode'],r['power1_cap']) for r in power], 'power settings changed')
            summary['cleanup'] = dict(passed=True,api_left_stopped=True,power_after=after)
        except BaseException as error:
            summary['cleanup'] = dict(passed=False,error_type=type(error).__name__,error=str(error))
            summary['qualification_completed'] = False
        summary['finished_utc'] = S.stamp()
        S.write(out/'summary.json',summary)
        receipts = {str(p.relative_to(out)):S.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        S.write(out/'RECEIPTS.sha256.json',receipts)
        for sig,handler in old.items():
            signal.signal(sig,handler)
    S.need(summary['cleanup']['passed'], 'cleanup failed')
    if failure:
        raise failure
    print(json.dumps({'qualification_completed':True,'components':summary['components'],'model_pp_tg_gain_claim':False}))

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    paths = ('build','resource-build','manual-review','resource-gate','boundary-summary','capture-summary','real0','real1','output')
    hashes = ('build-sha','baseline-sha','candidate-sha','capture-summary-sha','real0-sha','real1-sha')
    for name in paths:
        p.add_argument('--'+name,type=Path,required=True)
    for name in hashes:
        p.add_argument('--'+name,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--leave-api-stopped',action='store_true')
    a = p.parse_args(argv)
    S.need(all(getattr(a,n.replace('-','_')).is_absolute() for n in paths), 'absolute paths required')
    S.need(all(re.fullmatch('[0-9a-f]{64}',getattr(a,n.replace('-','_'))) for n in hashes), 'six explicit SHA256 values required')
    S.need(a.leave_api_stopped and __debug__, 'offline only; --leave-api-stopped required; Python -O forbidden')
    if not a.execute:
        print('arguments valid; not executed and no inputs/GPU/Docker inspected')
        return
    os.umask(0o077)
    with open(LOCK,'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(a)

if __name__ == '__main__':
    main()
