#!/usr/bin/env python3
"""Set own-table 925 mV caps on all cards and measure the second pair.

Amude only; all writes are volatile. On failure leave production stopped and
fan100. SIGKILL, uninterruptible kernel writers and host failure need recovery.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

import baseline as b

STOCK = '3d341947c3b5613e296d768ad35b654a46b485f77576099c5a1c7d9920d9ffe6'


def clock_settings():
    states = []
    for device in b.BDFS:
        d = Path('/sys/bus/pci/devices') / device
        hw = list((d / 'hwmon').glob('hwmon*'))
        b.c.require(len(hw) == 1, 'GPU hwmon missing or ambiguous')
        states.append({'sclk': (d / 'pp_dpm_sclk').read_text().replace('*', '').split(),
                       'mclk': (d / 'pp_dpm_mclk').read_text().replace('*', '').split(),
                       'power_cap': (hw[0] / 'power1_cap').read_text().strip()})
    return states


def gpu_tables():
    return [(Path('/sys/bus/pci/devices') / device / 'pp_table').read_bytes() for device in b.BDFS]


def checked_targets(originals):
    targets = []
    for i, original in enumerate(originals):
        digest = hashlib.sha256(original).hexdigest()
        expected = b.EXPECTED_TABLES[0] if i < 2 else STOCK
        b.c.require(len(original) == 1730 and int.from_bytes(original[:2], 'little') == 1730 and
                    original[2:5] == bytes([11, 0, 2]) and
                    int.from_bytes(original[474:478], 'little') == 3 and digest == expected,
                    'Unexpected card table layout or identity')
        value = int.from_bytes(original[0x22a:0x22c], 'little')
        b.c.require(value == (3700 if i < 2 else 4650), 'Unexpected initial voltage field')
        target = bytearray(original)
        target[0x22a:0x22c] = (3700).to_bytes(2, 'little')
        b.c.require(all(x == y for n, (x, y) in enumerate(zip(original, target))
                        if n not in (0x22a, 0x22b)), 'Non-voltage table change')
        targets.append(bytes(target))
    return targets


def write_table(device, data):
    """Single root sysfs writer, tracked across timeout and ordinary signals."""
    path = str(Path('/sys/bus/pci/devices') / device / 'pp_table')
    writer = ('import os,sys; print(os.getpid(),flush=True); data=sys.stdin.buffer.read(); '
              'fd=os.open(sys.argv[1],os.O_WRONLY); n=os.write(fd,data); os.close(fd); assert n==len(data)')
    with tempfile.TemporaryFile(mode='w+b') as writer_log:
        process = None
        oldmask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            process = subprocess.Popen(['sudo', '-n', 'python3', '-c', writer, path],
                                       stdin=subprocess.PIPE, stdout=writer_log,
                                       stderr=subprocess.PIPE, start_new_session=True)
            signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)
            process.communicate(input=data, timeout=60)
        except BaseException as exc:
            if process is not None:
                try:
                    subprocess.run(['sudo', '-n', 'kill', '-KILL', '--', '-' + str(process.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=10, check=False)
                except subprocess.TimeoutExpired:
                    pass
            writer_log.seek(0)
            pid = writer_log.read().decode(errors='replace').strip() or 'unknown'
            state = 'unknown'
            if pid.isdecimal():
                try:
                    state = Path('/proc', pid, 'stat').read_text().split(') ', 1)[1][0]
                except FileNotFoundError:
                    state = 'absent'
            raise RuntimeError('GPU writer interrupted (' + type(exc).__name__ + '), PID ' + pid +
                               ' state ' + state + '; no further GPU operations') from exc
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)
        writer_log.seek(0)
        b.c.require(process.returncode == 0 and writer_log.read().strip().isdigit(), 'GPU writer failed')


def main():
    out = b.BASE / ('all-925-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    out.mkdir(mode=0o700)
    start_utc = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    original = b.c.inspect()
    cfg = json.loads(b.CONFIG.read_text())
    config_hash = b.c.digest(b.CONFIG)
    b.c.require(original['State']['Running'] and original['Image'] == b.EXPECTED_IMAGE and
                cfg['gpu'] == [0, 1] and cfg['exe'] == '/qualified-strata' and
                b.c.digest(b.BINARY) == b.EXPECTED_BINARY, 'Production identity changed')
    fixture = b.OLD / 'prompt-65536.ids.json'
    b.c.require(b.c.digest(fixture) == b.EXPECTED_FIXTURE, 'Fixture changed')
    ids = json.loads(fixture.read_text())
    b.c.require(len(ids) == b.c.N and all(type(t) is int and 0 <= t < b.c.VOCAB for t in ids),
                'Invalid fixture')
    originals = gpu_tables()
    targets = checked_targets(originals)
    states = clock_settings()
    for device, old, target in zip(b.BDFS, originals, targets):
        (out / (device + '.original.pp_table')).write_bytes(old)
        (out / (device + '.target.pp_table')).write_bytes(target)
    b.c.write_json(out / 'identity.json', {'container_id': original['Id'], 'image': original['Image'],
        'config_sha256': config_hash, 'binary_sha256': b.EXPECTED_BINARY,
        'fixture_sha256': b.EXPECTED_FIXTURE, 'start_utc': start_utc,
        'gpu_bdfs': b.BDFS, 'selected_pairs': [b.BDFS[:2], b.BDFS[2:]], 'clock_settings': states,
        'original_sha256': [hashlib.sha256(x).hexdigest() for x in originals],
        'target_sha256': [hashlib.sha256(x).hexdigest() for x in targets]})
    env = dict(x.split('=', 1) for x in original['Config']['Env'])
    key = env.get('STRATA_API_KEY', '')
    b.c.require(bool(key), 'No API key')
    env.update({str(k): str(v) for k, v in cfg.get('env', {}).items()})
    env = {k: v for k, v in env.items() if not b.c.SECRET.search(k)}
    args = [a for a in cfg['args'] if a != '--vision']
    for flag in ('--prompt-cache', '--prompt-cache-every', '--conversation-cache-mib',
                 '--conversation-cache-slots'):
        b.c.replace_arg(args, flag, 0)
    b.c.replace_arg(args, '--eos-ids', '2147483647')
    before = b.holders()
    b.c.require(any(b.descendant(int(p), original['State']['Pid']) and 'kfd' in devs
                    for p, devs in before.items()), 'GPU holder guard failed positive check')
    ras_before = b.ras()
    b.c.require(all(v == 0 for group in ras_before.values() for v in group.values()) and
                not b.kernel_faults(start_utc), 'GPU fault before maintenance')
    b.c.require(b.cmd('systemctl', 'is-active', 'gpu-duct-fan-control.service').strip() == b'active',
                'Fan policy not active')
    stopped = restore_started = False
    unsafe, stop_monitor, ready, restoring = (threading.Event() for _ in range(4))
    names, fault, results = [], [], []
    worker = None

    def monitor():
        try:
            with (out / 'telemetry.jsonl').open('x') as f:
                while not stop_monitor.is_set():
                    sample = b.sensor()
                    b.c.require(b.ras() == ras_before and not b.kernel_faults(start_utc),
                                'New GPU hardware or RAS fault')
                    f.write(json.dumps(sample) + '\n')
                    f.flush()
                    ready.set()
                    stop_monitor.wait(1)
        except BaseException as exc:
            fault.append(type(exc).__name__ + ': ' + str(exc))
            unsafe.set()
            ready.set()
            while not stop_monitor.is_set():
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
                stop_monitor.wait(1)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        idle = b.c.idle_gate(key)
        status, v1 = b.c.api('/status', key), b.c.api('/v1/status', key)
        b.c.require(status.get('busy') is False and status.get('queued') == 0 and
                    v1.get('loaded') is True and v1.get('activity', {}).get('in_flight') == 0 and
                    all(g['busy_percent'] <= 3 for g in b.c.gpu()), 'Idle gate lost')
        b.c.write_json(out / 'idle.json', idle)
        stopped = True
        b.c.docker('stop', '--time', '120', b.c.PRODUCTION, timeout=180)
        b.c.require(not b.c.inspect()['State']['Running'] and not b.holders() and
                    not b.cmd('ss', '-H', '-ltn', 'sport', '=', ':8080').strip(),
                    'Production, port or GPU clients still active')
        b.cmd('sudo', '-n', 'systemctl', 'stop', 'gpu-duct-fan-control.service', timeout=60)
        b.c.require(b.cmd('systemctl', 'show', '-p', 'ActiveState', '--value',
                          'gpu-duct-fan-control.service').strip() == b'inactive', 'Fan monitor still active')
        monitors = subprocess.run(['pgrep', '-x', 'btop|nvtop|rocm-smi|amd-smi'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        b.c.require(monitors.returncode == 1 and not b.holders() and gpu_tables() == originals,
                    'GPU monitoring/client or table state changed')
        b.sensor()
        b.c.require(b.ras() == ras_before and not b.kernel_faults(start_utc) and
                    clock_settings() == states, 'Hardware changed before write')
        # The first pair is already 925; only the second pair needs writes.
        for i in (2, 3):
            write_table(b.BDFS[i], targets[i])
            expected = [targets[j] if j <= i else originals[j] for j in range(4)]
            b.c.require(gpu_tables() == expected and clock_settings() == states and
                        b.ras() == ras_before and not b.kernel_faults(start_utc),
                        'GPU table, clocks or hardware fault after write')
            b.sensor()
        b.c.require(gpu_tables() == targets, 'Not all four 925 target tables read back')
        b.c.write_json(out / 'applied.json', {'passed': True, 'all_four_limit_mV': 925,
            'target_sha256': [hashlib.sha256(x).hexdigest() for x in targets],
            'other_table_bytes_unchanged': True, 'clocks_and_power_cap_unchanged': True})
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        b.c.require(ready.wait(45) and not unsafe.is_set(), 'Monitor did not start')
        for tag, visible in (('pair-05-08', '0,1'), ('pair-0b-0e', '2,3')):
            b.c.require(not unsafe.is_set() and not b.holders() and gpu_tables() == targets,
                        'Unsafe state before workload')
            env['HIP_VISIBLE_DEVICES'] = visible
            results.append(b.run_case(tag, args, cfg, original, env, ids, out, unsafe, names))
        reference = json.loads((b.REFERENCE / 'results.json').read_text())[0]
        b.c.require(not unsafe.is_set() and results[0]['output_ids'] == reference['output_ids'] and
                    results[0]['drafts_accepted'] == reference['drafts_accepted'] and
                    results[0]['drafts_offered'] == reference['drafts_offered'],
                    '05/08 native output or MTP mismatch')
        b.c.write_json(out / 'results.json', results)
        b.c.require(not b.holders() and gpu_tables() == targets and clock_settings() == states and
                    b.ras() == ras_before and not b.kernel_faults(start_utc), 'Unsafe state before restore')
        for name in names:
            b.c.require(not b.c.docker('ps', '-aq', '--filter', 'name=^/' + name + '$').strip(),
                        'Benchmark container remains')
        b.c.require(b.c.digest(b.BINARY) == b.EXPECTED_BINARY and b.c.digest(b.CONFIG) == config_hash and
                    b.c.inspect()['Id'] == original['Id'], 'Production identity changed')
        restore_started = True
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
        stop_monitor.set()
        worker.join(timeout=45)
        b.c.require(not worker.is_alive() and not unsafe.is_set() and b.ras() == ras_before and
                    not b.kernel_faults(start_utc), 'Fault during restore')
        b.cmd('sudo', '-n', 'systemctl', 'start', 'gpu-duct-fan-control.service', timeout=60)
        b.c.write_json(out / 'restore.json', {'passed': True, 'loaded': True, 'fan_policy_active': True})
    except BaseException as exc:
        stop_monitor.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        if restore_started:
            try:
                if b.c.inspect()['State']['Running']:
                    b.c.docker('stop', '--time', '15', b.c.PRODUCTION, timeout=60)
            except BaseException as stop_exc:
                fault.append('Production stop failed: ' + type(stop_exc).__name__)
        if stopped:
            try:
                b.cmd('sudo', '-n', 'systemctl', 'stop', 'gpu-duct-fan-control.service', timeout=60)
                b.cmd('sudo', '-n', '/usr/local/sbin/fan-percent', '100', timeout=30)
                b.fan_state()
            except BaseException as fan_exc:
                fault.append('Fan100 failed: ' + type(fan_exc).__name__)
        b.c.write_json(out / 'error.json', {'error': str(exc), 'faults': fault})
        raise
    print('Result directory:', out, flush=True)


if __name__ == '__main__':
    main()
