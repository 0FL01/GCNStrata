#!/usr/bin/env python3
"""Restore known-working 05/08 production without changing any GPU table."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

import baseline as b

spec = importlib.util.spec_from_file_location('recovery950', b.BASE / 'recovery-950.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
REF = b.BASE / 'all-925-20261113T120714Z' / 'identity.json'
PROBE = r'''
import json,os,re,sys
pid=int(sys.argv[1]);base='/proc/'+str(pid)
data=open(base+'/environ','rb').read().split(b'\0')
keys=('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')
env={k.decode():v.decode() for x in data if b'=' in x for k,v in [x.split(b'=',1)] if k.decode() in keys}
maps=open(base+'/maps').read(); devices=sorted(set(re.findall(r'/dev/dri/(renderD\d+)',maps)))
print(json.dumps({'env':env,'mapped_render_devices':devices}))
'''


def engine_placement(container_pid):
    found = []
    for pid, devices in b.holders().items():
        if 'kfd' not in devices or not b.descendant(int(pid), container_pid):
            continue
        exe_hash = b.cmd('sudo', '-n', 'sha256sum', '/proc/' + pid + '/exe').decode().split()[0]
        if exe_hash != b.EXPECTED_BINARY:
            continue
        data = json.loads(b.cmd('sudo', '-n', 'python3', '-c', PROBE, pid))
        found.append({'pid': int(pid), 'binary_sha256': exe_hash, **data})
    b.c.require(len(found) == 1, 'Cannot identify exactly one qualified engine')
    entry = found[0]
    b.c.require(entry['env'] == {'HIP_VISIBLE_DEVICES': '0,1'} and
                entry['mapped_render_devices'] == ['renderD128', 'renderD129'],
                'Qualified engine does not use only expected production GPUs')
    return entry


def normal_fan_sample():
    """Continue temperature/activity monitoring once normal fan control takes over."""
    cards = {}
    for device in b.BDFS:
        d = Path('/sys/bus/pci/devices') / device
        hw = list((d / 'hwmon').glob('hwmon*'))
        b.c.require(len(hw) == 1, 'GPU hwmon missing or ambiguous')
        values = {k: int((hw[0] / name).read_text()) for k, name in {
            'edge_mC': 'temp1_input', 'junction_mC': 'temp2_input', 'hbm_mC': 'temp3_input',
            'ppt_uW': 'power1_input', 'vddgfx_mV': 'in0_input', 'sclk_hz': 'freq1_input',
            'mclk_hz': 'freq2_input'}.items()}
        values['busy_pct'] = int((d / 'gpu_busy_percent').read_text())
        values['vram_used'] = int((d / 'mem_info_vram_used').read_text())
        b.c.require(values['edge_mC'] < 90000 and values['junction_mC'] < 95000 and
                    values['hbm_mC'] < 84000, 'Temperature abort threshold')
        cards[device] = values
    fans = [p for p in Path('/sys/class/hwmon').glob('hwmon*')
            if (p / 'name').read_text().strip() == 'nct6793']
    b.c.require(len(fans) == 1, 'Duct fan controller missing or ambiguous')
    fan = {name: int((fans[0] / name).read_text())
           for name in ('pwm2', 'pwm2_enable', 'pwm2_mode', 'fan2_input')}
    b.c.require(fan['pwm2'] > 0 and fan['pwm2_mode'] == 1 and fan['fan2_input'] > 0,
                'Normal fan actuator or tach failure')
    return {'monotonic': time.monotonic(), 'cards': cards, 'fan': fan}


def stop_until_released(stop):
    while not stop.is_set():
        try:
            subprocess.run(['docker', 'stop', '--time', '15', b.c.PRODUCTION],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=60, check=False)
        except subprocess.TimeoutExpired:
            pass
        stop.wait(1)


def main():
    saved = json.loads((r.RECOVERY / 'pre-reboot.json').read_text())
    post = json.loads((r.RECOVERY / 'post-reboot.json').read_text())
    ref = json.loads(REF.read_text())
    cfg = json.loads(b.CONFIG.read_text())
    original = b.c.inspect()
    config_hash = b.c.digest(b.CONFIG)
    b.c.require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == post['boot_id'] and
                original['Id'] == saved['production_id'] and not original['State']['Running'] and
                original['HostConfig']['RestartPolicy']['Name'] == 'no' and
                saved['restart_policy'] == {'Name': 'unless-stopped', 'MaximumRetryCount': 0} and
                original['Image'] == b.EXPECTED_IMAGE and config_hash == ref['config_sha256'] and
                b.c.digest(b.BINARY) == b.EXPECTED_BINARY and cfg['gpu'] == [0, 1] and
                cfg.get('backend') == 'hip' and not cfg.get('hip_ordinal') and
                cfg['exe'] == '/qualified-strata' and not cfg.get('vision', {}).get('gpu') and
                cfg.get('vision', {}).get('cuda_device') is None, 'Production identity or GPU config changed')
    visibility = ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
    container_env = dict(x.split('=', 1) for x in original['Config']['Env'])
    b.c.require({k: container_env[k] for k in visibility if k in container_env} ==
                {'HIP_VISIBLE_DEVICES': '0,1'} and
                not any(k in cfg.get('env', {}) for k in visibility), 'Unexpected visibility override')
    key = container_env.get('STRATA_API_KEY', '')
    b.c.require(bool(key), 'API key missing')
    targets = [(r.RECOVERY / (d + '.target.pp_table')).read_bytes() for d in b.BDFS]
    b.c.require(r.all925.gpu_tables() == targets and r.all925.clock_settings() == ref['clock_settings'] and
                [b.c.digest(r.RECOVERY / (d + '.target.pp_table')) for d in b.BDFS] == saved['target_sha256'],
                'GPU target table or clock settings changed')
    b.c.require(not b.holders() and not b.c.docker('ps', '-aq', '--filter',
                'name=strata-unsloth64k-').strip() and
                not b.cmd('ss', '-H', '-ltn', 'sport', '=', ':8080').strip(),
                'GPU holder, benchmark container or API listener remains')
    b.c.require(not r.gpu_faults(r.boot_start()), 'New-boot GPU fault present')
    initial_ras = b.ras()
    b.c.require(all(v == 0 for group in initial_ras.values() for v in group.values()), 'RAS error present')
    initial = b.sensor()
    b.c.require(all(v['busy_pct'] <= 3 for v in initial['cards'].values()), 'GPU not idle')
    inactive_vram = {d: initial['cards'][d]['vram_used'] for d in b.BDFS[2:]}
    out = r.RECOVERY / ('restore-production-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    out.mkdir(mode=0o700)
    stop, ready, unsafe = threading.Event(), threading.Event(), threading.Event()
    fan_released, fan_confirmed = threading.Event(), threading.Event()
    fan_lock = threading.Lock()
    faults = []
    started = policy_attempted = False

    def monitor():
        try:
            with (out / 'telemetry.jsonl').open('x') as f:
                while not stop.is_set():
                    with fan_lock:
                        sample = normal_fan_sample() if fan_released.is_set() else b.sensor()
                    b.c.require(b.ras() == initial_ras and not r.gpu_faults(r.boot_start()),
                                'GPU fault during production startup')
                    b.c.require(all(sample['cards'][d]['busy_pct'] <= 3 for d in b.BDFS[2:]),
                                'Unexpected second-pair GPU activity')
                    if fan_confirmed.is_set():
                        active = subprocess.run(['systemctl', 'is-active', '--quiet',
                                                 'gpu-duct-fan-control.service']).returncode
                        b.c.require(active == 0, 'Normal fan service stopped')
                    f.write(json.dumps(sample) + '\n')
                    f.flush()
                    ready.set()
                    stop.wait(1)
        except BaseException as exc:
            faults.append(type(exc).__name__ + ': ' + str(exc))
            unsafe.set()
            ready.set()
            stop_until_released(stop)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        b.c.require(ready.wait(45) and not unsafe.is_set(), 'Monitor did not start')
        started = True
        b.c.docker('start', b.c.PRODUCTION, timeout=180)
        deadline, loaded = time.monotonic() + 900, False
        while time.monotonic() < deadline:
            b.c.require(not unsafe.is_set(), 'Monitor aborted production startup')
            try:
                v1 = b.c.api('/v1/status', key)
                state = b.c.inspect()['State']
                loaded = (v1.get('loaded') is True and v1.get('context', {}).get('native') == 262144 and
                          v1.get('vision', {}).get('enabled') is True and
                          state.get('Health', {}).get('Status') == 'healthy')
                if loaded:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        b.c.require(loaded, 'Production readiness deadline exceeded')
        state = b.c.inspect()
        b.c.require(state['Id'] == original['Id'] and state['State']['Running'], 'Container changed')
        placement = engine_placement(state['State']['Pid'])
        final = b.sensor()
        b.c.require(all(final['cards'][d]['busy_pct'] <= 3 and
                        final['cards'][d]['vram_used'] <= inactive_vram[d] + 64 * 1024 * 1024 and
                        final['cards'][d]['junction_mC'] < 45000 for d in b.BDFS[2:]),
                    'Second pair is not idle, unused, or cool')
        status, v1 = b.c.api('/status', key), b.c.api('/v1/status', key)
        b.c.require(type(status.get('busy')) is bool and type(status.get('queued')) is int and
                    type(v1.get('activity', {}).get('in_flight')) is int, 'Unexpected activity schema')
        with urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5) as response:
            health = json.load(response)
        b.c.require(health.get('status') == 'ok' and health.get('loaded') is True and
                    health.get('api_key') is True and health.get('images') is True and
                    health.get('max_context') == 262144 and
                    health.get('model') == 'Qwen3.8-Flash-Next-Unsloth-UD-Q2_K_XL-Q8-PLE',
                    'Health model/context/auth/images mismatch')
        b.c.require(r.all925.gpu_tables() == targets and r.all925.clock_settings() == ref['clock_settings'] and
                    b.c.digest(b.CONFIG) == config_hash and b.c.digest(b.BINARY) == b.EXPECTED_BINARY and
                    b.ras() == initial_ras and not r.gpu_faults(r.boot_start()) and not unsafe.is_set(),
                    'Hardware or runtime changed during startup')
        with fan_lock:
            fan_released.set()
        b.cmd('sudo', '-n', 'systemctl', 'start', 'gpu-duct-fan-control.service', timeout=60)
        b.c.require(b.cmd('systemctl', 'is-active', 'gpu-duct-fan-control.service').strip() == b'active',
                    'Normal fan policy failed to start')
        fan_confirmed.set()
        before_policy = b.c.inspect()
        b.c.require(before_policy['Id'] == original['Id'] and before_policy['State']['Running'] and
                    before_policy['State'].get('Health', {}).get('Status') == 'healthy' and
                    not unsafe.is_set(), 'Production not healthy before restart-policy restoration')
        policy_attempted = True
        b.c.docker('update', '--restart=unless-stopped', b.c.PRODUCTION, timeout=30)
        final_state = b.c.inspect()
        b.c.require(final_state['Id'] == original['Id'] and final_state['State']['Running'] and
                    final_state['State'].get('Health', {}).get('Status') == 'healthy' and
                    final_state['HostConfig']['RestartPolicy'] == saved['restart_policy'] and
                    b.ras() == initial_ras and not r.gpu_faults(r.boot_start()) and not unsafe.is_set(),
                    'Fault or unhealthy service during final policy restoration')
        status, v1 = b.c.api('/status', key), b.c.api('/v1/status', key)
        b.c.require(type(status.get('busy')) is bool and type(status.get('queued')) is int and
                    v1.get('loaded') is True and type(v1.get('activity', {}).get('in_flight')) is int,
                    'Final activity status invalid')
        stop.set()
        worker.join(timeout=45)
        b.c.require(not worker.is_alive() and not unsafe.is_set(), 'Monitor failed to stop cleanly')
        b.c.write_json(out / 'result.json', {'passed': True, 'same_container': True,
            'healthy': True, 'model': health['model'], 'context': 262144, 'images': True, 'auth': True,
            'status': {'busy': status.get('busy'), 'queued': status.get('queued'),
                       'loaded': v1.get('loaded'), 'in_flight': v1.get('activity', {}).get('in_flight')},
            'engine_placement': placement, 'inactive_vram_before': inactive_vram,
            'inactive_vram_after': {d: final['cards'][d]['vram_used'] for d in b.BDFS[2:]},
            'restart_policy': saved['restart_policy'], 'normal_fan_policy': True,
            'target_sha256': saved['target_sha256'], 'boot_id': post['boot_id']})
    except BaseException as exc:
        stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        if policy_attempted:
            try:
                r.disable_restart_after_failure()
            except BaseException as policy_exc:
                faults.append('Restart policy disable failed: ' + type(policy_exc).__name__)
        if started:
            try:
                if b.c.inspect()['State']['Running']:
                    b.c.docker('stop', '--time', '15', b.c.PRODUCTION, timeout=60)
                b.c.require(not b.c.inspect()['State']['Running'], 'Production did not stop')
            except BaseException as stop_exc:
                faults.append('Production stop failed: ' + type(stop_exc).__name__)
        try:
            b.cmd('sudo', '-n', 'systemctl', 'stop', 'gpu-duct-fan-control.service', timeout=60)
            b.cmd('sudo', '-n', '/usr/local/sbin/fan-percent', '100', timeout=30)
            b.fan_state()
        except BaseException as fan_exc:
            faults.append('Fan100 failed: ' + type(fan_exc).__name__)
        b.c.write_json(out / 'error.json', {'error': str(exc), 'faults': faults})
        raise
    print('Restoration evidence:', out, flush=True)


if __name__ == '__main__':
    main()
