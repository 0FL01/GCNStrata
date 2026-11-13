#!/usr/bin/env python3
"""Approved 05/08 maintenance-window baseline; no GPU table writes.

Runs on amude. Reuses the existing native controller's command, protocol and
cleanup. SIGKILL, host failure and Docker failure still require operator recovery.
"""
import glob
import bisect
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

BASE = Path('/home/radneon/strata/deploy/gfx906-baseline-20261113')
OLD = Path('/home/radneon/strata/deploy/unsloth-q2-k-xl-20261111')
CONFIG = Path('/home/radneon/strata/configs/unsloth-q2-k-xl-20261111.json')
BINARY = Path('/home/radneon/strata/releases/gfx906-bounds-main-53e6c6a5/strata')
EXPECTED_BINARY = '67877d40ee363a4f33e7d9cd35da73e1d7c6ac284f5c9ee26870315552a52234'
EXPECTED_FIXTURE = '63ce3d4042027a00b2193feab143f1d9443b5c31a863a962ec9543ccb54d9695'
EXPECTED_IMAGE = 'sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
EXPECTED_TABLES = ['face662c087e324da09bac5353775f6df1228b0a5527ab774d9691ad4799ed67'] * 2
CANDIDATE_SHA = '44873548440e1f657f9baddf3156b0648bf3a2854ef1be843aaeb4fe3fc085ea'
REFERENCE = BASE / 'baseline-20261113T111903Z'
BDFS = ['0000:05:00.0', '0000:08:00.0', '0000:0b:00.0', '0000:0e:00.0']
HOLDER_CODE = r'''
import glob,json,os,stat
paths=['/dev/kfd']+glob.glob('/dev/dri/renderD*')
dev={os.stat(p).st_rdev:os.path.basename(p) for p in paths}
out={}; denied=[]
for p in glob.glob('/proc/[0-9]*'):
 pid=int(p.rsplit('/',1)[-1]); found=set()
 try: fds=os.listdir(p+'/fd')
 except FileNotFoundError: continue
 except PermissionError: denied.append(pid); continue
 for fd in fds:
  try: s=os.stat(p+'/fd/'+fd)
  except FileNotFoundError: continue
  except PermissionError: denied.append(pid); continue
  if stat.S_ISCHR(s.st_mode) and s.st_rdev in dev: found.add(dev[s.st_rdev])
 if found: out[pid]=sorted(found)
print(json.dumps({'holders':out,'denied':len(set(denied))}))
'''

spec = importlib.util.spec_from_file_location('native_controller', OLD / 'controller.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def cmd(*args, timeout=30):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout).stdout


def holders():
    result = json.loads(cmd('sudo', '-n', 'python3', '-c', HOLDER_CODE))
    c.require(result['denied'] == 0, 'Cannot inspect all GPU holders')
    return result['holders']


def descendant(pid, root):
    while pid > 1:
        if pid == root:
            return True
        try:
            status = Path(f'/proc/{pid}/status').read_text()
            pid = int(next(s.split()[1] for s in status.splitlines() if s.startswith('PPid:')))
        except (OSError, StopIteration, ValueError):
            return False
    return False


def tables():
    return [c.digest(Path('/sys/bus/pci/devices') / b / 'pp_table') for b in BDFS[:2]]


def clock_settings():
    out = []
    for b in BDFS[:2]:
        d = Path('/sys/bus/pci/devices') / b
        clocks = [(d / f).read_text().replace('*', '').split() for f in ('pp_dpm_sclk', 'pp_dpm_mclk')]
        h = list((d / 'hwmon').glob('hwmon*'))
        c.require(len(h) == 1, 'GPU hwmon missing or ambiguous')
        out.append((clocks, (h[0] / 'power1_cap').read_text().strip()))
    return out


def apply_900(ras_before, start_utc, unsafe, profiles, fault, restore=False):
    """Apply the checked 925/900 pair in either direction; fail closed."""
    try:
        source_hash = [CANDIDATE_SHA] * 2 if restore else EXPECTED_TABLES
        target_hash = EXPECTED_TABLES[0] if restore else CANDIDATE_SHA
        source_value, target_value = (3600, 3700) if restore else (3700, 3600)
        c.require(not holders() and tables() == source_hash, 'Not in known GPU-exclusive state')
        c.require(cmd('systemctl', 'show', '-p', 'ActiveState', '--value',
                      'gpu-duct-fan-control.service').strip() == b'inactive', 'Fan monitor still active')
        monitors = subprocess.run(['pgrep', '-x', 'btop|nvtop|rocm-smi|amd-smi'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        c.require(monitors.returncode == 1, 'GPU monitoring process present or check failed')
        sensor()
        c.require(ras() == ras_before and not kernel_faults(start_utc), 'Hardware fault before write')
        before_clocks = clock_settings()
        for b, candidate in zip(BDFS[:2], profiles):
            live = (Path('/sys/bus/pci/devices') / b / 'pp_table').read_bytes()
            c.require(len(live) == len(candidate) == 1730 and
                      [i for i, (x, y) in enumerate(zip(live, candidate)) if x != y] == [0x22a] and
                      int.from_bytes(live[0x22a:0x22c], 'little') == source_value and
                      int.from_bytes(candidate[0x22a:0x22c], 'little') == target_value and
                      hashlib.sha256(candidate).hexdigest() == target_hash,
                      'Candidate does not match checked 925-to-900 field-only change')
        writer = ('import os,sys; print(os.getpid(),flush=True); p=sys.argv[1]; data=sys.stdin.buffer.read(); '
                  'fd=os.open(p,os.O_WRONLY); n=os.write(fd,data); os.close(fd); '
                  'assert n==len(data)')
        for b, data in zip(BDFS[:2], profiles):
            path = str(Path('/sys/bus/pci/devices') / b / 'pp_table')
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
                    # A signal or timeout can leave sudo's root child alive. An
                    # uninterruptible kernel task may survive even SIGKILL.
                    if process is not None:
                        try:
                            subprocess.run(['sudo', '-n', 'kill', '-KILL', '--', '-' + str(process.pid)],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           timeout=10, check=False)
                        except subprocess.TimeoutExpired:
                            pass
                    writer_log.seek(0)
                    writer_pid = writer_log.read().decode(errors='replace').strip() or 'unknown'
                    state = 'unknown'
                    if writer_pid.isdecimal():
                        try:
                            state = Path('/proc', writer_pid, 'stat').read_text().split(') ', 1)[1][0]
                        except FileNotFoundError:
                            state = 'absent'
                    raise RuntimeError('GPU writer interrupted (' + type(exc).__name__ + '); writer PID ' +
                                       writer_pid + ' state ' + state + '; no further GPU operations') from exc
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)
                writer_log.seek(0)
                output = writer_log.read()
                c.require(process.returncode == 0 and output.strip().isdigit(), 'GPU writer failed')
            c.require(c.digest(path) == target_hash, 'GPU table readback mismatch')
            c.require(clock_settings() == before_clocks and ras() == ras_before and
                      not kernel_faults(start_utc), 'Hardware state changed during candidate write')
            sensor()
    except BaseException as exc:
        fault.append('Candidate write failure: ' + type(exc).__name__ + ': ' + str(exc))
        unsafe.set()
        raise


def qualify_900(results, telemetry):
    reference = json.loads((REFERENCE / 'results.json').read_text())
    c.require(len(reference) == len(results) == 2, 'Need two reference and candidate runs')
    baseline_samples = [json.loads(line) for line in (REFERENCE / 'telemetry.jsonl').open()]
    candidate_samples = [json.loads(line) for line in telemetry.open()]
    def means(runs, samples, bdf):
        times = [s['monotonic'] for s in samples]
        values = []
        for r in runs:
            a, z = r['gen_monotonic'], r['done_monotonic']
            left, right = bisect.bisect_right(times, a) - 1, bisect.bisect_left(times, z)
            c.require(left >= 0 and right < len(times), 'Incomplete request telemetry')
            window = samples[left:right + 1]
            c.require(max(y['monotonic'] - x['monotonic'] for x, y in zip(window, window[1:])) < 2.5,
                      'Request telemetry gap')
            vals = [s['cards'][bdf]['junction_mC'] for s in window if a <= s['monotonic'] <= z]
            c.require(bool(vals), 'No request temperature samples')
            values.append(statistics.mean(vals))
        return values
    perf = {k: statistics.median(r[k] for r in results) /
               statistics.median(r[k] for r in reference) for k in ('pp_tok_s', 'tg_tok_s')}
    output_ok = all(r['output_ids'] == reference[0]['output_ids'] and
                    r['drafts_accepted'] == reference[0]['drafts_accepted'] and
                    r['drafts_offered'] == reference[0]['drafts_offered']
                    for r in reference + results)
    temperature = {}
    for b in BDFS[:2]:
        before, after = means(reference, baseline_samples, b), means(results, candidate_samples, b)
        spread = max(before) - min(before)
        temperature[b] = {'baseline_mean_mC': before, 'candidate_mean_mC': after,
                          'baseline_spread_mC': spread,
                          'improved_beyond_spread': statistics.median(after) < statistics.median(before) - spread}
    passed = output_ok and all(v >= 0.97 for v in perf.values()) and all(
        value['improved_beyond_spread'] for value in temperature.values())
    return {'passed': passed, 'performance_ratio': perf, 'output_and_mtp_match': output_ok,
            'junction_temperature': temperature}


def ras():
    out = {}
    for b in BDFS:
        for name in ('gfx_err_count', 'umc_err_count'):
            p = Path('/sys/bus/pci/devices') / b / 'ras' / name
            fields = dict(line.split(':', 1) for line in p.read_text().splitlines())
            out[b + '/' + name] = {k: int(v.strip()) for k, v in fields.items()}
    return out


def fan_state():
    fans = [p for p in Path('/sys/class/hwmon').glob('hwmon*')
            if (p / 'name').read_text().strip() == 'nct6793']
    c.require(len(fans) == 1, 'Duct fan controller missing or ambiguous')
    fan = {f: int((fans[0] / f).read_text())
           for f in ('pwm2', 'pwm2_enable', 'pwm2_mode', 'fan2_input')}
    # nct6775 reports enable=0 for mode 0 + pwm=255 (full-speed override).
    c.require(fan['pwm2'] == 255 and fan['pwm2_enable'] in (0, 1) and
              fan['pwm2_mode'] == 1 and fan['fan2_input'] > 0,
              'Fan 100% or tach failure')
    return fan


def sensor():
    cards = {}
    for b in BDFS:
        d = Path('/sys/bus/pci/devices') / b
        h = list((d / 'hwmon').glob('hwmon*'))
        c.require(len(h) == 1, 'GPU hwmon missing or ambiguous')
        vals = {k: int((h[0] / f).read_text()) for k, f in {
            'edge_mC': 'temp1_input', 'junction_mC': 'temp2_input', 'hbm_mC': 'temp3_input',
            'ppt_uW': 'power1_input', 'vddgfx_mV': 'in0_input', 'sclk_hz': 'freq1_input',
            'mclk_hz': 'freq2_input'}.items()}
        vals['busy_pct'] = int((d / 'gpu_busy_percent').read_text())
        vals['vram_used'] = int((d / 'mem_info_vram_used').read_text())
        c.require(vals['edge_mC'] < 90000 and vals['junction_mC'] < 95000 and
                  vals['hbm_mC'] < 84000, 'Temperature abort threshold')
        cards[b] = vals
    return {'monotonic': time.monotonic(), 'cards': cards, 'fan': fan_state()}


def kernel_faults(since):
    data = cmd('journalctl', '-k', '--since', since, '--no-pager', '-o', 'cat', timeout=30)
    pattern = re.compile(r'(amdgpu.*(fault|error|reset|timeout)|hardware error|\bOops\b)', re.I)
    return [line for line in data.decode(errors='replace').splitlines() if pattern.search(line)]


def run_case(tag, args, cfg, original, env, ids, out, unsafe, names):
    """The existing controller's native protocol, with GEN/DONE monotonic markers."""
    import uuid
    name = 'strata-unsloth64k-' + uuid.uuid4().hex[:12]
    names.append(name)
    command = c.command(name, cfg, original, env, args)
    lines, tokens = queue.Queue(), []
    process = reader = None
    start, gen, ready = time.monotonic(), None, None
    info, metrics = {}, None
    with (out / (tag + '.stdout')).open('xb') as raw, (out / (tag + '.stderr')).open('xb') as err:
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=err, env={**os.environ, **env}, bufsize=0)
            def pump():
                try:
                    for data in process.stdout:
                        raw.write(data)
                        raw.flush()
                        lines.put(data.decode('utf-8', errors='strict'))
                finally:
                    lines.put(None)
            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            while metrics is None:
                c.require(not unsafe.is_set(), 'Sensor or hardware fault')
                c.require(time.monotonic() - start < 5400, 'Per-run deadline exceeded')
                try:
                    line = lines.get(timeout=1)
                except queue.Empty:
                    continue
                c.require(line is not None, 'Native engine exited before DONE')
                c.require(not line.startswith(('ERR', 'FATAL')), 'Native engine error')
                if line.startswith('INFO '):
                    info.update(kv.split('=', 1) for kv in line.split()[1:] if '=' in kv)
                elif line.startswith('RESUME '):
                    c.require(int(line.split()[1]) == 0, 'Native prompt reuse')
                elif line.startswith('READY '):
                    c.require(ready is None and int(line.split()[1]) == 262144 and
                              info.get('kv') == 'int8' and info.get('context') == '262144',
                              'Unexpected native READY/INFO')
                    ready = time.monotonic()
                    request = ('GEN 512 temperature=1 top_p=0.95 top_k=20 seed=12345 '
                               + ','.join(map(str, ids)) + '\n').encode()
                    gen = time.monotonic()
                    def send():
                        try:
                            rest = memoryview(request)
                            while rest:
                                written = process.stdin.write(rest)
                                if not written:
                                    raise OSError('Short stdin write')
                                rest = rest[written:]
                            process.stdin.flush()
                        except OSError:
                            lines.put('ERR stdin write failed\n')
                    threading.Thread(target=send, daemon=True).start()
                elif line.startswith('T '):
                    c.require(gen is not None, 'Token before GEN')
                    tokens.append(int(line[2:]))
                elif line.startswith('DONE '):
                    done = time.monotonic()
                    c.require(gen is not None, 'DONE before GEN')
                    try:
                        metrics = c.parse_done(line, tokens, len(ids), 512)
                    except BaseException as exc:
                        c.write_json(out / (tag + '.failure.json'),
                                     {'error': str(exc), 'native_done': line.rstrip(),
                                      'observed_tokens': len(tokens), 'gen_monotonic': gen,
                                      'done_monotonic': done})
                        raise
                    metrics.update(tag=tag, passed=True, engine_info=info,
                                   gen_monotonic=gen, done_monotonic=done,
                                   load_startup_wall_s=ready - start,
                                   request_wall_s=done - gen)
            process.stdin.write(b'QUIT\n')
            process.stdin.flush()
            process.stdin.close()
            c.require(process.wait(timeout=120) == 0, 'Native engine nonzero exit')
            reader.join(timeout=10)
            c.require(not reader.is_alive(), 'Native stdout did not close')
        finally:
            c.cleanup(name, process)
            if reader:
                reader.join(timeout=10)
    c.write_json(out / (tag + '.result.json'), metrics)
    return metrics


def main():
    candidate_mode = sys.argv[1:] == ['--candidate-900']
    c.require(len(sys.argv) == 1 or candidate_mode or (len(sys.argv) == 3 and sys.argv[1] == '--resume'),
              'Usage: baseline.py [--candidate-900 | --resume prior-baseline-directory]')
    prior = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else None
    if prior:
        c.require(prior.parent == BASE and prior.name.startswith('baseline-'),
                  'Invalid previous experiment directory')
    BASE.mkdir(mode=0o700, exist_ok=True)
    out = BASE / ('baseline-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    out.mkdir(mode=0o700)
    start_utc = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    c.require(c.digest(BINARY) == EXPECTED_BINARY, 'Qualified binary changed')
    fixture = OLD / 'prompt-65536.ids.json'
    c.require(c.digest(fixture) == EXPECTED_FIXTURE, 'Fixture changed')
    ids = json.loads(fixture.read_text())
    c.require(len(ids) == c.N and all(type(t) is int and 0 <= t < c.VOCAB for t in ids),
              'Invalid fixture')
    cfg = json.loads(CONFIG.read_text())
    original = c.inspect()
    c.require((original['State']['Running'] != bool(prior)) and original['Image'] == EXPECTED_IMAGE and
              cfg['gpu'] == [0, 1] and cfg['exe'] == '/qualified-strata', 'Runtime changed')
    c.require(tables() == EXPECTED_TABLES, 'Unexpected live GPU tables')
    for b in BDFS[:2]:
        (out / (b + '.pp_table')).write_bytes((Path('/sys/bus/pci/devices') / b / 'pp_table').read_bytes())
    config_hash = c.digest(CONFIG)
    profiles = []
    if candidate_mode:
        ref_identity = json.loads((REFERENCE / 'identity.json').read_text())
        c.require(ref_identity['config_sha256'] == config_hash and
                  ref_identity['binary_sha256'] == EXPECTED_BINARY and
                  ref_identity['tables_sha256'] == EXPECTED_TABLES and
                  c.digest(REFERENCE / 'results.json') == '2a02268d5b4044bafa8d384d8108d61baf79b4298ebd222f93e635f42c9d4acb' and
                  c.digest(REFERENCE / 'telemetry.jsonl') == 'd44eadb0b3e7468edaf9196b9020fd9deb680734d95e623fd31c5ba561bb03aa',
                  'Reference baseline changed')
        for b in BDFS[:2]:
            profile = '/var/lib/amd-gpu-tuning/' + b.replace(':', '-') + '-1300-900.pp_table'
            data = cmd('sudo', '-n', 'cat', profile)
            c.require(len(data) == 1730 and hashlib.sha256(data).hexdigest() == CANDIDATE_SHA,
                      'Existing 900 profile changed')
            profiles.append(data)
    env = dict(x.split('=', 1) for x in original['Config']['Env'])
    key = env.get('STRATA_API_KEY', '')
    c.require(bool(key), 'No API key')
    env.update({str(k): str(v) for k, v in cfg.get('env', {}).items()})
    env['HIP_VISIBLE_DEVICES'] = '0,1'
    env = {k: v for k, v in env.items() if not c.SECRET.search(k)}
    args = [a for a in cfg['args'] if a != '--vision']
    for k in ('--prompt-cache', '--prompt-cache-every', '--conversation-cache-mib',
              '--conversation-cache-slots'):
        c.replace_arg(args, k, 0)
    c.replace_arg(args, '--eos-ids', '2147483647')
    identity = {'image': original['Image'], 'container_id': original['Id'],
                'binary_sha256': EXPECTED_BINARY, 'fixture_sha256': EXPECTED_FIXTURE,
                'config_sha256': config_hash, 'tables_sha256': tables(),
                'gpu_bdfs': BDFS[:2], 'start_utc': start_utc}
    if prior:
        old_identity = json.loads((prior / 'identity.json').read_text())
        old_before = json.loads((prior / 'holders-before.json').read_text())
        old_after = json.loads((prior / 'holders-after.json').read_text())
        c.require(old_identity['container_id'] == original['Id'] and
                  old_identity['config_sha256'] == config_hash and
                  old_identity['tables_sha256'] == EXPECTED_TABLES and
                  old_before['production_detected'] and old_after['count'] == 0 and
                  old_after['port_8080_listening'] is False and (prior / 'idle.json').exists(),
                  'Prior maintenance evidence does not match current state')
        start_utc = old_identity['start_utc']
        identity['maintenance_start_utc'] = start_utc
        c.require(not holders(), 'GPU devices held in stopped window')
        c.require(not cmd('ss', '-H', '-ltn', 'sport', '=', ':8080').strip(),
                  'Port 8080 listening in stopped window')
        c.require(subprocess.run(['systemctl', 'is-active', '--quiet', 'gpu-duct-fan-control.service']).returncode != 0,
                  'Fan service unexpectedly active in stopped window')
        fan_state()
        c.write_json(out / 'resume.json', {'previous_directory': str(prior),
                                           'previous_idle_and_holder_checks_verified': True})
    else:
        before = holders()
        pid = original['State']['Pid']
        c.require(any(descendant(int(p), pid) and 'kfd' in devs for p, devs in before.items()),
                  'GPU holder check failed to detect running production')
        c.write_json(out / 'holders-before.json', {'count': len(before), 'production_detected': True})
    c.write_json(out / 'identity.json', identity)
    ras_before = ras()
    if prior:
        c.require(all(value == 0 for group in ras_before.values() for value in group.values()),
                  'RAS error since known-zero maintenance state')
    c.require(not kernel_faults(start_utc), 'New hardware fault before stop')
    if not prior:
        c.require(cmd('systemctl', 'is-active', 'gpu-duct-fan-control.service').strip() == b'active',
                  'Fan policy not active initially')
    stopped = fan_stopped = bool(prior)
    unsafe = threading.Event()
    stop_monitor = threading.Event()
    monitor_ready = threading.Event()
    restoring = threading.Event()
    names = []
    result = []
    fault = []
    def monitor():
        try:
            with (out / 'telemetry.jsonl').open('x') as f:
                while not stop_monitor.is_set():
                    sample = sensor()
                    c.require(ras() == ras_before, 'New RAS error')
                    errors = kernel_faults(start_utc)
                    c.require(not errors, 'New kernel GPU/hardware fault')
                    f.write(json.dumps(sample) + '\n')
                    f.flush()
                    monitor_ready.set()
                    stop_monitor.wait(1)
        except BaseException as exc:
            fault.append(type(exc).__name__ + ': ' + str(exc))
            unsafe.set()
            monitor_ready.set()
            while not stop_monitor.is_set():
                for name in list(names):
                    try:
                        subprocess.run(['docker', 'rm', '--force', name], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=60, check=False)
                    except subprocess.TimeoutExpired:
                        pass
                if restoring.is_set():
                    try:
                        subprocess.run(['docker', 'stop', '--time', '15', c.PRODUCTION],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       timeout=45, check=False)
                    except subprocess.TimeoutExpired:
                        pass
                stop_monitor.wait(1)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    worker = None
    restore_started = False
    qualified = False
    try:
        if not prior:
            idle = c.idle_gate(key)
            s, v = c.api('/status', key), c.api('/v1/status', key)
            c.require(s.get('busy') is False and s.get('queued') == 0 and
                      v.get('loaded') is True and v.get('activity', {}).get('in_flight') == 0 and
                      all(g['busy_percent'] <= 3 for g in c.gpu()), 'Idle gate lost')
            c.write_json(out / 'idle.json', idle)
            stopped = True
            c.docker('stop', '--time', '120', c.PRODUCTION, timeout=180)
            c.require(not c.inspect()['State']['Running'], 'Production did not stop')
            ports = cmd('ss', '-H', '-ltn', 'sport', '=', ':8080').decode().strip()
            c.require(not ports, 'Port 8080 still listening')
            after = holders()
            c.require(not after, 'GPU devices still held after production stop')
            c.write_json(out / 'holders-after.json', {'count': 0, 'port_8080_listening': False})
            fan_stopped = True
            try:
                cmd('sudo', '-n', 'systemctl', 'stop', 'gpu-duct-fan-control.service', timeout=60)
            except BaseException:
                unsafe.set()
                raise
        if candidate_mode:
            apply_900(ras_before, start_utc, unsafe, profiles, fault)
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        c.require(monitor_ready.wait(45) and not unsafe.is_set(), 'Monitor failed to start')
        for n in (1, 2):
            c.require(not unsafe.is_set(), 'Monitor failed before run')
            result.append(run_case('baseline-' + str(n), args, cfg, original, env,
                                   ids, out, unsafe, names))
            c.require(not unsafe.is_set(), 'Monitor aborted run')
        c.write_json(out / 'results.json', result)
        if candidate_mode:
            verdict = qualify_900(result, out / 'telemetry.jsonl')
            c.write_json(out / 'qualification.json', verdict)
            qualified = verdict['passed']
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        try:
            c.require(not unsafe.is_set(), 'Sensor or hardware fault: ' + '; '.join(fault))
            c.require(ras() == ras_before and not kernel_faults(start_utc), 'New hardware fault')
            c.require(not candidate_mode or qualified, 'Candidate unqualified; production remains stopped')
            c.require(tables() == ([CANDIDATE_SHA] * 2 if candidate_mode else EXPECTED_TABLES),
                      'GPU table changed')
            for name in names:
                c.require(not c.docker('ps', '-aq', '--filter', 'name=^/' + name + '$').strip(),
                          'Owned test container remains')
            current = c.inspect()
            c.require(current['Id'] == original['Id'] and current['Image'] == EXPECTED_IMAGE and
                      c.digest(CONFIG) == config_hash and c.digest(BINARY) == EXPECTED_BINARY,
                      'Production identity changed')
            if stopped and not current['State']['Running']:
                c.require(worker is not None and worker.is_alive() and monitor_ready.is_set() and
                          not unsafe.is_set(), 'No working safety monitor for restoration')
                fan_state()
                c.require(not holders(), 'GPU clients remain after benchmark')
                restore_started = True
                restoring.set()
                c.docker('start', c.PRODUCTION, timeout=180)
                deadline = time.monotonic() + 900
                ready = False
                while time.monotonic() < deadline:
                    c.require(not unsafe.is_set(), 'Hardware monitor aborted restoration')
                    try:
                        v = c.api('/v1/status', key)
                        ready = (v.get('loaded') is True and
                                 v.get('context', {}).get('native') == 262144 and
                                 v.get('vision', {}).get('enabled') is True)
                        if ready:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(1)
                c.require(ready, 'Production readiness deadline exceeded')
                c.require(c.inspect()['Id'] == original['Id'] and c.inspect()['State']['Running'],
                          'Restored production identity/state changed')
            stop_monitor.set()
            if worker:
                worker.join(timeout=45)
                c.require(not worker.is_alive(), 'Monitor failed to stop')
            c.require(not unsafe.is_set() and ras() == ras_before and not kernel_faults(start_utc),
                      'Hardware fault during restoration')
            if fan_stopped:
                cmd('sudo', '-n', 'systemctl', 'start', 'gpu-duct-fan-control.service', timeout=60)
            c.write_json(out / 'restore.json', {'passed': True, 'same_container': True,
                                               'config_unchanged': True, 'fan_policy_active': fan_stopped})
        except BaseException as exc:
            stop_monitor.set()
            try:
                need_fan100 = fan_stopped or (stopped and not c.inspect()['State']['Running'])
            except BaseException:
                need_fan100 = stopped
            if need_fan100:
                try:
                    cmd('sudo', '-n', 'systemctl', 'stop', 'gpu-duct-fan-control.service', timeout=60)
                except BaseException as fan_exc:
                    fault.append('Fan policy stop failure: ' + type(fan_exc).__name__)
                try:
                    cmd('sudo', '-n', '/usr/local/sbin/fan-percent', '100', timeout=30)
                    fan_state()
                except BaseException as fan_exc:
                    fault.append('Fan100 failure: ' + type(fan_exc).__name__)
            if restore_started:
                try:
                    if c.inspect()['State']['Running']:
                        c.docker('stop', '--time', '15', c.PRODUCTION, timeout=60)
                    c.require(not c.inspect()['State']['Running'], 'Failed to stop unsafe production')
                except BaseException as stop_exc:
                    fault.append('Production stop failure: ' + type(stop_exc).__name__)
            c.write_json(out / 'recovery-error.json', {'type': type(exc).__name__, 'error': str(exc),
                                                       'faults': fault})
            raise
    print('Result directory:', out, flush=True)


if __name__ == '__main__':
    main()
