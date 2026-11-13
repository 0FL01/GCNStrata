#!/usr/bin/env python3
"""After approved reboot, stage 925/925/925/950, then test 0b/0e twice.

No GPU autostart. An unsuccessful apply/test leaves Strata stopped, fan100,
and its temporary restart policy disabled. Host hangs and SIGKILL need recovery.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

import baseline as b

spec = importlib.util.spec_from_file_location('all925', b.BASE / 'all-925.py')
all925 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(all925)
RECOVERY = b.BASE / 'recovery-950-20261113'
PREVIOUS = b.BASE / 'all-925-20261113T120714Z'
STOCK = '3d341947c3b5613e296d768ad35b654a46b485f77576099c5a1c7d9920d9ffe6'


def boot_start():
    seconds = int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines()
                       if x.startswith('btime ')))
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(seconds))


def gpu_faults(since):
    # The headless 05 display connector emits these current-boot EDID
    # messages; they are not GPU compute/reset faults. Preserve all others.
    known = {
        '[drm:dm_helpers_read_local_edid [amdgpu]] *ERROR* EDID err: 2, on connector: DP-2',
        'amdgpu 0000:05:00.0: [drm] *ERROR* Failed to read EDID',
        'amdgpu 0000:05:00.0: [drm] *ERROR* No EDID found on connector: DP-2.',
    }
    return [line for line in b.kernel_faults(since) if line not in known]


def disable_restart_after_failure():
    b.c.docker('update', '--restart=no', b.c.PRODUCTION, timeout=30)
    b.c.require(b.c.inspect()['HostConfig']['RestartPolicy']['Name'] == 'no',
                'Restart policy did not return to no')


def main():
    before = json.loads((RECOVERY / 'pre-reboot.json').read_text())
    history = json.loads((PREVIOUS / 'identity.json').read_text())
    oldboot = before['boot_id']
    currentboot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    b.c.require(currentboot != oldboot, 'Approved reboot not observed')
    original = b.c.inspect()
    cfg = json.loads(b.CONFIG.read_text())
    config_hash = b.c.digest(b.CONFIG)
    b.c.require(not original['State']['Running'] and original['Id'] == before['production_id'] and
                original['HostConfig']['RestartPolicy']['Name'] == 'no' and
                before['restart_policy'] == {'Name': 'unless-stopped', 'MaximumRetryCount': 0} and
                original['Image'] == b.EXPECTED_IMAGE and config_hash == history['config_sha256'] and
                b.c.digest(b.BINARY) == b.EXPECTED_BINARY and cfg['gpu'] == [0, 1],
                'Production identity/state or temporary policy changed')
    b.c.require(not b.holders() and not b.c.docker('ps', '-aq', '--filter',
                'name=strata-unsloth64k-').strip() and
                not b.cmd('ss', '-H', '-ltn', 'sport', '=', ':8080').strip(),
                'GPU clients, test container or API listener present')
    enabled = subprocess.run(['systemctl', 'is-enabled', 'amd-gpu-tuning.service'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    b.c.require(enabled.returncode == 1 and enabled.stdout.strip() == b'disabled',
                'GPU tuning autostart not confirmed disabled')
    tuning = b.cmd('systemctl', 'show', 'amd-gpu-tuning.service', '-p', 'RequiredBy',
                   '-p', 'WantedBy', '-p', 'TriggeredBy').decode().splitlines()
    b.c.require(all(line.endswith('=') for line in tuning) and
                'amd-gpu-tuning.service' not in b.cmd('systemctl', 'show', 'docker.service',
                                                       '-p', 'Requires').decode(),
                'GPU tuning still has activation dependencies')
    stocks = all925.gpu_tables()
    b.c.require(len(stocks) == 4 and all(hashlib.sha256(x).hexdigest() == STOCK for x in stocks),
                'Stock readback after reboot differs from known stock')
    targets = [(RECOVERY / (d + '.target.pp_table')).read_bytes() for d in b.BDFS]
    for i, (device, uid, stock, target, expected) in enumerate(zip(
            b.BDFS, before['uids'], stocks, targets, before['target_sha256'])):
        path = Path('/sys/bus/pci/devices') / device
        previous = (PREVIOUS / (device + '.target.pp_table')).read_bytes()
        b.c.require((path / 'unique_id').read_text().strip() == uid and len(target) == 1730 and
                    target[:5] == bytes([0xc2, 0x06, 11, 0, 2]) and
                    int.from_bytes(target[474:478], 'little') == 3 and
                    hashlib.sha256(target).hexdigest() == expected and
                    int.from_bytes(target[554:556], 'little') == (3800 if i == 3 else 3700) and
                    (target == previous if i < 3 else all(
                        x == y for n, (x, y) in enumerate(zip(previous, target)) if n not in (554, 555))),
                    'Device or staged target mismatch')
        if i >= 2:
            oldstock = (PREVIOUS / (device + '.original.pp_table')).read_bytes()
            b.c.require(stock == oldstock, 'Second-pair stock differs from its own original')
        (RECOVERY / (device + '.post-reboot-stock.pp_table')).write_bytes(stock)
    fixture = b.OLD / 'prompt-65536.ids.json'
    b.c.require(b.c.digest(fixture) == b.EXPECTED_FIXTURE, 'Fixture changed')
    ids = json.loads(fixture.read_text())
    b.c.require(len(ids) == b.c.N and all(type(t) is int and 0 <= t < b.c.VOCAB for t in ids),
                'Invalid fixture')
    boot_utc = boot_start()
    ras_before = b.ras()
    b.c.require(all(v == 0 for group in ras_before.values() for v in group.values()) and
                not gpu_faults(boot_utc), 'Fault or RAS error since reboot')
    b.c.write_json(RECOVERY / 'post-reboot.json', {'boot_id': currentboot, 'previous_boot_id': oldboot,
        'boot_utc': boot_utc, 'production_stopped': True, 'temporary_restart_policy': 'no',
        'stock_sha256': [hashlib.sha256(x).hexdigest() for x in stocks],
        'uids': before['uids'], 'ras_zero': True, 'kernel_faults': 0})
    env = dict(x.split('=', 1) for x in original['Config']['Env'])
    key = env.get('STRATA_API_KEY', '')
    b.c.require(bool(key), 'No API key')
    env.update({str(k): str(v) for k, v in cfg.get('env', {}).items()})
    env['HIP_VISIBLE_DEVICES'] = '2,3'
    env = {k: v for k, v in env.items() if not b.c.SECRET.search(k)}
    args = [x for x in cfg['args'] if x != '--vision']
    for flag in ('--prompt-cache', '--prompt-cache-every', '--conversation-cache-mib',
                 '--conversation-cache-slots'):
        b.c.replace_arg(args, flag, 0)
    b.c.replace_arg(args, '--eos-ids', '2147483647')
    out = RECOVERY / ('run-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    out.mkdir(mode=0o700)
    unsafe, stop, ready, restoring = (threading.Event() for _ in range(4))
    names, faults, results = [], [], []
    started = policy_update_attempted = False
    worker = None

    def monitor():
        try:
            with (out / 'telemetry.jsonl').open('x') as f:
                while not stop.is_set():
                    sample = b.sensor()
                    b.c.require(b.ras() == ras_before and not gpu_faults(boot_utc),
                                'New GPU hardware or RAS fault')
                    f.write(json.dumps(sample) + '\n')
                    f.flush()
                    ready.set()
                    stop.wait(1)
        except BaseException as exc:
            faults.append(type(exc).__name__ + ': ' + str(exc))
            unsafe.set()
            ready.set()
            while not stop.is_set():
                for name in list(names):
                    try:
                        subprocess.run(['docker', 'rm', '--force', name], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=60, check=False)
                    except subprocess.TimeoutExpired:
                        pass
                if restoring.is_set():
                    try:
                        subprocess.run(['docker', 'stop', '--time', '15', b.c.PRODUCTION],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       timeout=45, check=False)
                    except subprocess.TimeoutExpired:
                        pass
                stop.wait(1)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        b.cmd('sudo', '-n', 'systemctl', 'stop', 'gpu-duct-fan-control.service', timeout=60)
        b.c.require(b.cmd('systemctl', 'show', '-p', 'ActiveState', '--value',
                          'gpu-duct-fan-control.service').strip() == b'inactive', 'Fan monitor still active')
        monitors = subprocess.run(['pgrep', '-x', 'btop|nvtop|rocm-smi|amd-smi'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        b.c.require(monitors.returncode == 1 and not b.holders() and all925.gpu_tables() == stocks,
                    'GPU monitor, client or stock changed')
        b.sensor()
        b.c.require(b.ras() == ras_before and not gpu_faults(boot_utc), 'Fault before writes')
        for i, device in enumerate(b.BDFS):
            all925.write_table(device, targets[i])
            expected = [targets[j] if j <= i else stocks[j] for j in range(4)]
            clocks = all925.clock_settings()
            b.c.require(all925.gpu_tables() == expected and
                        all(clocks[j] == history['clock_settings'][j] for j in range(i + 1)) and
                        b.ras() == ras_before and not gpu_faults(boot_utc),
                        'GPU readback, clocks or hardware fault after write')
            b.sensor()
        b.c.require(all925.gpu_tables() == targets and all925.clock_settings() == history['clock_settings'],
                    'Final table or clock settings mismatch')
        b.c.write_json(out / 'applied.json', {'passed': True, 'nominal_mV': [925, 925, 925, 950],
            'target_sha256': [hashlib.sha256(x).hexdigest() for x in targets],
            'clock_settings_unchanged': True})
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        b.c.require(ready.wait(45) and not unsafe.is_set(), 'Monitor failed to start')
        for n in (1, 2):
            b.c.require(not unsafe.is_set() and not b.holders() and all925.gpu_tables() == targets,
                        'Unsafe state before workload')
            results.append(b.run_case('pair-0b-0e-' + str(n), args, cfg, original, env,
                                      ids, out, unsafe, names))
        b.c.require(not unsafe.is_set() and results[0]['output_ids'] == results[1]['output_ids'] and
                    results[0]['drafts_accepted'] == results[1]['drafts_accepted'] and
                    results[0]['drafts_offered'] == results[1]['drafts_offered'],
                    'Second-pair output or MTP mismatch')
        b.c.write_json(out / 'results.json', results)
        b.c.require(not b.holders() and all925.gpu_tables() == targets and
                    all925.clock_settings() == history['clock_settings'] and b.ras() == ras_before and
                    not gpu_faults(boot_utc), 'Unsafe state before restore')
        for name in names:
            b.c.require(not b.c.docker('ps', '-aq', '--filter', 'name=^/' + name + '$').strip(),
                        'Benchmark container remains')
        b.c.require(b.c.digest(b.CONFIG) == config_hash and b.c.digest(b.BINARY) == b.EXPECTED_BINARY and
                    b.c.inspect()['Id'] == original['Id'], 'Production identity changed')
        started = True
        restoring.set()
        b.c.docker('start', b.c.PRODUCTION, timeout=180)
        deadline, loaded = time.monotonic() + 900, False
        while time.monotonic() < deadline:
            b.c.require(not unsafe.is_set(), 'Monitor aborted production startup')
            try:
                status = b.c.api('/v1/status', key)
                loaded = (status.get('loaded') is True and
                          status.get('context', {}).get('native') == 262144 and
                          status.get('vision', {}).get('enabled') is True)
                if loaded:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        b.c.require(loaded and b.c.inspect()['Id'] == original['Id'] and
                    b.c.inspect()['State']['Running'], 'Production did not restore')
        stop.set()
        worker.join(timeout=45)
        b.c.require(not worker.is_alive() and not unsafe.is_set() and b.ras() == ras_before and
                    not gpu_faults(boot_utc), 'Fault during restore')
        b.cmd('sudo', '-n', 'systemctl', 'start', 'gpu-duct-fan-control.service', timeout=60)
        policy_update_attempted = True
        b.c.docker('update', '--restart=unless-stopped', b.c.PRODUCTION, timeout=30)
        b.c.require(b.c.inspect()['HostConfig']['RestartPolicy'] == before['restart_policy'],
                    'Original restart policy not restored')
        b.c.write_json(out / 'restore.json', {'passed': True, 'loaded': True,
            'fan_policy_active': True, 'restart_policy': before['restart_policy']})
    except BaseException as exc:
        stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        if policy_update_attempted:
            try:
                disable_restart_after_failure()
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
    print('Result directory:', out, flush=True)


if __name__ == '__main__':
    main()
