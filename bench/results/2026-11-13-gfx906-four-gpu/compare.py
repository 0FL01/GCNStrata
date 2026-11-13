#!/usr/bin/env python3
"""Matched current Strata two/four GPU test; no tuning or production config writes."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path('/home/radneon/strata/deploy/four-gpu-64k-20261113')
spec = importlib.util.spec_from_file_location('baseline', '/home/radneon/strata/deploy/gfx906-baseline-20261113/baseline.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
c = b.c

def main():
    MODES = ('--serial-control', '--four-resident-off', '--four-resident-off-nan', '--equal-tables', '--four-pipe')
    c.require(len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in MODES), 'Usage: compare.py [' + ' | '.join(MODES) + ']')
    equal_tables = sys.argv[1:] == ['--equal-tables']
    four_pipe = sys.argv[1:] == ['--four-pipe']
    serial_control = sys.argv[1:] == ['--serial-control']
    dbg_nan = sys.argv[1:] == ['--four-resident-off-nan']
    resident_off = dbg_nan or sys.argv[1:] == ['--four-resident-off']
    overrides = ({'STRATA_VERIFY_ALL_RESIDENT':'0'} if resident_off else {}) | ({'STRATA_DBG_NAN':'1'} if dbg_nan else {})
    out = ROOT / time.strftime('run-%Y%m%dT%H%M%SZ', time.gmtime())
    out.mkdir(parents=True, mode=0o700)
    cfg = json.loads(b.CONFIG.read_text())
    original = c.inspect()
    c.require(original['State']['Running'] and original['Image'] == b.EXPECTED_IMAGE and cfg['gpu'] == [0, 1], 'Production identity changed')
    c.require(c.digest(b.BINARY) == b.EXPECTED_BINARY and c.digest(b.OLD / 'prompt-65536.ids.json') == b.EXPECTED_FIXTURE, 'Binary/fixture changed')
    config_hash = c.digest(b.CONFIG)
    table_hashes = [c.digest(Path('/sys/bus/pci/devices') / d / 'pp_table') for d in b.BDFS]
    ids = json.loads((b.OLD / 'prompt-65536.ids.json').read_text())
    env = dict(x.split('=', 1) for x in original['Config']['Env'])
    key = env['STRATA_API_KEY']
    env.update(cfg.get('env', {}))
    env = {k: str(v) for k, v in env.items() if not c.SECRET.search(k)}
    args = [a for a in cfg['args'] if a != '--vision']
    for flag in ('--prompt-cache', '--prompt-cache-every', '--conversation-cache-mib', '--conversation-cache-slots'):
        c.replace_arg(args, flag, 0)
    c.replace_arg(args, '--eos-ids', '2147483647')
    (out / 'config.before.json').write_text(b.CONFIG.read_text())
    (out / 'compose.before.yaml').write_text(Path('/home/radneon/strata/compose.yaml').read_text())
    c.write_json(out / 'identity.json', {'image':original['Image'], 'container':original['Id'], 'binary':b.EXPECTED_BINARY, 'fixture':b.EXPECTED_FIXTURE, 'tables':table_hashes, 'config':config_hash})
    start = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    initial_ras = b.ras()
    unsafe, stop, ready = threading.Event(), threading.Event(), threading.Event()
    restoring = threading.Event()
    names, faults, results = [], [], []
    stopped = fan_stopped = False
    def monitor():
        try:
            with (out / 'telemetry.jsonl').open('x') as f:
                while not stop.is_set():
                    sample = b.sensor()
                    c.require(b.ras() == initial_ras and not b.kernel_faults(start), 'New GPU/RAS fault')
                    f.write(json.dumps(sample) + '\n'); f.flush(); ready.set()
                    stop.wait(1)
        except BaseException as e:
            faults.append(str(e)); unsafe.set(); ready.set()
            if restoring.is_set():
                c.docker('stop', '--time', '15', c.PRODUCTION, timeout=45)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, interrupted)
    worker = None
    try:
        c.write_json(out / 'idle.json', c.idle_gate(key))
        stopped = True
        c.docker('stop', '--time', '120', c.PRODUCTION, timeout=180)
        c.require(not c.inspect()['State']['Running'] and not b.holders(), 'GPU clients remain')
        fan_stopped = True
        b.cmd('sudo', '-n', 'systemctl', 'stop', 'gpu-duct-fan-control.service', timeout=60)
        if equal_tables:
            # Second pair gets the first pair's exact live table (1300 MHz / 925 mV): known-working on identical cards.
            dev = Path('/sys/bus/pci/devices')
            src = (dev / b.BDFS[0] / 'pp_table').read_bytes()
            c.require(src == (dev / b.BDFS[1] / 'pp_table').read_bytes() and len(src) == 1730, 'First-pair tables differ')
            for d in b.BDFS[2:]:
                (out / (d + '.before.pp_table')).write_bytes((dev / d / 'pp_table').read_bytes())
                r = subprocess.run(['sudo', '-n', 'python3', '-c', 'import os,sys; d=sys.stdin.buffer.read(); fd=os.open(sys.argv[1],os.O_WRONLY); assert os.write(fd,d)==len(d); os.close(fd)', str(dev / d / 'pp_table')], input=src, timeout=60)
                c.require(r.returncode == 0 and (dev / d / 'pp_table').read_bytes() == src, 'Table write/readback failed')
            time.sleep(3)
            table_hashes = [c.digest(dev / d / 'pp_table') for d in b.BDFS]
            c.write_json(out / 'equal-tables.json', {d: {'sclk': (dev / d / 'pp_dpm_sclk').read_text().split('\n'), 'cap': [x.read_text().strip() for x in (dev / d / 'hwmon').glob('hwmon*/power1_cap')]} for d in b.BDFS})
        worker = threading.Thread(target=monitor, daemon=True); worker.start()
        c.require(ready.wait(30) and not unsafe.is_set(), 'Monitor start failed')
        # ABBA: current supported two-stage pipeline versus native four-stage placement.
        cases = [('two-serial-1',2), ('two-serial-2',2)] if serial_control else [('two-1',2), ('four-1',4), ('four-2',4), ('two-2',2)]
        if resident_off:
            cases = [('four-resident-off-1',4), ('four-resident-off-2',4)]
        if equal_tables:
            cases = [('four-eq-1',4), ('four-eq-2',4)]
        if four_pipe:
            cases = [('four-pipe-1',4), ('four-pipe-2',4)]
        for tag, n in cases:
            runargs = list(args)
            runenv = dict(env, HIP_VISIBLE_DEVICES=','.join(map(str, range(n))))
            runenv.update(overrides)
            if n == 4:
                c.replace_arg(runargs, '--layer-split', 'auto')
                if not four_pipe: c.replace_arg(runargs, '--pipeline-windows', '0')
            elif serial_control:
                c.replace_arg(runargs, '--pipeline-windows', '0')
            c.write_json(out / (tag + '.settings.json'), {'args':runargs, 'gpu':list(range(n)), 'overrides':overrides})
            r = b.run_case(tag, runargs, cfg, original, runenv, ids, out, unsafe, names)
            c.require(not unsafe.is_set(), 'Hardware monitor failed')
            results.append(r)
            c.write_json(out / 'results.json', results)
    except BaseException as e:
        c.write_json(out / 'error.json', {'type':type(e).__name__, 'error':str(e), 'faults':faults})
        raise
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        try:
            c.require(not unsafe.is_set() and b.ras() == initial_ras and not b.kernel_faults(start), 'Hardware unsafe; production stays stopped')
            for name in names:
                c.require(not c.docker('ps','-aq','--filter','name=^/'+name+'$').strip(), 'Benchmark container remains')
            c.require(not b.holders() and c.digest(b.CONFIG) == config_hash and table_hashes == [c.digest(Path('/sys/bus/pci/devices')/d/'pp_table') for d in b.BDFS], 'Restore identity/holder mismatch')
            if stopped:
                restoring.set()
                c.require(not unsafe.is_set(), 'Hardware fault before restore')
                c.restore(original, b.CONFIG, config_hash, key, out)
            c.require(not unsafe.is_set() and b.ras() == initial_ras and not b.kernel_faults(start), 'Hardware fault during restore')
            stop.set()
            if worker: worker.join(timeout=40)
            if fan_stopped: b.cmd('sudo','-n','systemctl','start','gpu-duct-fan-control.service',timeout=60)
        finally:
            stop.set()
            if worker: worker.join(timeout=40)
    print(out, flush=True)

if __name__ == '__main__': main()
