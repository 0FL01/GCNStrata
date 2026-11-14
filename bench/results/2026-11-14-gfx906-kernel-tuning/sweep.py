#!/usr/bin/env python3
"""Four-GPU pipelined sweep: one production stop, one run per variant.
usage: sweep.py name[:K=V,--flag=V,...] ...   (EXE=<host binary> swaps the engine; K=V env override, --flag=V arg override; variant 'base' = no overrides)"""
import hashlib, importlib.util, json, os, signal, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path('/home/radneon/strata/deploy/four-gpu-64k-20261113')
spec = importlib.util.spec_from_file_location('baseline', '/home/radneon/strata/deploy/gfx906-baseline-20261113/baseline.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
c = b.c
_ras, _kf = b.ras, b.kernel_faults
# correctable HBM (umc ce) errors are logged and reported, not fatal: only uncorrectable ones / other kernel faults abort
b.ras = lambda: {k: {f: n for f, n in v.items() if f != 'ce'} for k, v in _ras().items()}
b.kernel_faults = lambda since: [l for l in _kf(since) if not ('correctable hardware errors' in l and 'uncorrectable' not in l)]

# production binary since 2026-11-14 (gfx906-k906-02759a20); the production config adds STRATA_HIP_PROMPT_F16=1,
# which a variant must name itself: the base settings here still come from the 2026-11-11 config
b.BINARY = Path('/home/radneon/strata/releases/gfx906-k906-02759a20/strata')
b.EXPECTED_BINARY = 'e29f4509378e5b7958ec4cb4f313206605a727938ec507da975f2f4fbdc27332'

def parse(v):
    name, _, rest = v.partition(':')
    env, arg = {}, []
    for item in filter(None, rest.split(',')):
        k, _, val = item.partition('=')
        if k.startswith('--'): arg.append((k, val))
        else: env[k] = val
    return name, env, arg

_exe = {'path': None}
_command = c.command
def _cmd(name, cfg, original, env, args):
    cmd = _command(name, cfg, original, env, args)
    if _exe['path']:
        hit = [i for i, x in enumerate(cmd) if x.endswith(':/qualified-strata:ro')]
        c.require(len(hit) == 1 and Path(_exe['path']).is_file(), 'EXE override failed')
        cmd[hit[0]] = _exe['path'] + ':/qualified-strata:ro'
    return cmd
c.command = _cmd

def main():
    variants = [parse(v) for v in sys.argv[1:]]
    c.require(variants and len({v[0] for v in variants}) == len(variants), __doc__)
    out = ROOT / time.strftime('sweep-%Y%m%dT%H%M%SZ', time.gmtime()); out.mkdir(parents=True, mode=0o700)
    cfg = json.loads(b.CONFIG.read_text())
    original = c.inspect()
    c.require(original['State']['Running'] and original['Image'] == b.EXPECTED_IMAGE, 'Production identity changed')
    c.require(c.digest(b.BINARY) == b.EXPECTED_BINARY and c.digest(b.OLD / 'prompt-65536.ids.json') == b.EXPECTED_FIXTURE, 'Binary/fixture changed')
    config_hash = c.digest(b.CONFIG)
    dev = Path('/sys/bus/pci/devices')
    table_hashes = [c.digest(dev / d / 'pp_table') for d in b.BDFS]
    c.require(len(set(table_hashes)) == 1, 'GPU tables differ')
    ids = json.loads((b.OLD / 'prompt-65536.ids.json').read_text())
    env = dict(x.split('=', 1) for x in original['Config']['Env'])
    key = env['STRATA_API_KEY']
    env.update(cfg.get('env', {}))
    env = {k: str(v) for k, v in env.items() if not c.SECRET.search(k)}
    args = [a for a in cfg['args'] if a != '--vision']
    for flag in ('--prompt-cache', '--prompt-cache-every', '--conversation-cache-mib', '--conversation-cache-slots'):
        c.replace_arg(args, flag, 0)
    c.replace_arg(args, '--eos-ids', '2147483647')
    c.replace_arg(args, '--layer-split', 'auto')
    start = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    initial_ras = b.ras()
    unsafe, stop, ready, restoring = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    names, faults, results = [], [], []
    stopped = fan_stopped = False
    def monitor():
        try:
            with (out / 'telemetry.jsonl').open('x') as f:
                while not stop.is_set():
                    for attempt in range(3):   # a card's SMU can miss one poll (EIO); three in a row is a fault
                        try: sample = b.sensor(); break
                        except OSError:
                            if attempt == 2: raise
                            time.sleep(0.5)
                    c.require(b.ras() == initial_ras and not b.kernel_faults(start), 'New GPU/RAS fault')
                    f.write(json.dumps(sample) + '\n'); f.flush(); ready.set()
                    stop.wait(1)
        except BaseException as e:
            faults.append(str(e)); unsafe.set(); ready.set()
            if restoring.is_set():
                c.docker('stop', '--time', '15', c.PRODUCTION, timeout=45)
    def interrupted(signum, frame): raise KeyboardInterrupt('Interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, interrupted)
    worker = None
    try:
        c.write_json(out / 'idle.json', c.idle_gate(key))
        stopped = True
        c.docker('stop', '--time', '120', c.PRODUCTION, timeout=180)
        c.require(not c.inspect()['State']['Running'] and not b.holders(), 'GPU clients remain')
        fan_stopped = True
        b.cmd('sudo', '-n', 'systemctl', 'stop', 'gpu-duct-fan-control.service', timeout=60)
        if os.environ.get('TABLE'):
            tp = Path(os.environ['TABLE'])
            for d in b.BDFS:
                src = (tp / (d + '.pp_table')).read_bytes() if tp.is_dir() else tp.read_bytes()
                c.require(len(src) == 1730, 'Bad table size')
                r = subprocess.run(['sudo', '-n', 'python3', '-c', 'import os,sys; d=sys.stdin.buffer.read(); fd=os.open(sys.argv[1],os.O_WRONLY); assert os.write(fd,d)==len(d); os.close(fd)', str(dev / d / 'pp_table')], input=src, timeout=60)
                c.require(r.returncode == 0 and (dev / d / 'pp_table').read_bytes() == src, 'Table write/readback failed')
            time.sleep(3)
            table_hashes = [c.digest(dev / d / 'pp_table') for d in b.BDFS]
            print('table', os.environ['TABLE'], [(d[5:7], (dev / d / 'pp_dpm_sclk').read_text().split()[-2], hashlib.sha256((dev / d / 'pp_table').read_bytes()).hexdigest()[:8]) for d in b.BDFS], flush=True)
        worker = threading.Thread(target=monitor, daemon=True); worker.start()
        c.require(ready.wait(30) and not unsafe.is_set(), 'Monitor start failed')
        for name, venv, varg in variants:
            runargs = list(args)
            for k, v in varg: c.replace_arg(runargs, k, v)
            venv = dict(venv); exe = venv.pop('EXE', None)  # EXE=<host path>: run that binary instead of the production one
            _exe['path'] = exe
            runenv = dict(env, HIP_VISIBLE_DEVICES='0,1,2,3'); runenv.update(venv)
            c.write_json(out / (name + '.settings.json'), {'env': venv, 'arg': varg, 'args': runargs})
            try:
                r = b.run_case(name, runargs, cfg, original, runenv, ids, out, unsafe, names)
            except Exception as e:
                c.require(not unsafe.is_set(), 'Hardware monitor failed')
                r = {'tag': name, 'passed': False, 'error': str(e)}
            c.require(not unsafe.is_set(), 'Hardware monitor failed')
            results.append(r); c.write_json(out / 'results.json', results)
            print(name, r.get('error') or 'PP %.1f TG %.2f acc %s/%s' % (r['prompt_tokens'] * 1000 / r['prompt_ms'], r['generated'] * 1000 / r['decode_ms'], r['drafts_accepted'], r['drafts_offered']) + ' out ' + hashlib.sha256(json.dumps(r['output_ids']).encode()).hexdigest()[:10], flush=True)
    except BaseException as e:
        c.write_json(out / 'error.json', {'type': type(e).__name__, 'error': str(e), 'faults': faults})
        raise
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        try:
            c.require(not unsafe.is_set() and b.ras() == initial_ras and not b.kernel_faults(start), 'Hardware unsafe; production stays stopped')
            for name in names:
                c.require(not c.docker('ps', '-aq', '--filter', 'name=^/' + name + '$').strip(), 'Benchmark container remains')
            c.require(not b.holders() and c.digest(b.CONFIG) == config_hash and table_hashes == [c.digest(dev / d / 'pp_table') for d in b.BDFS], 'Restore identity/holder mismatch')
            if stopped:
                restoring.set()
                c.restore(original, b.CONFIG, config_hash, key, out)
            stop.set()
            if worker: worker.join(timeout=40)
            if fan_stopped: b.cmd('sudo', '-n', 'systemctl', 'start', 'gpu-duct-fan-control.service', timeout=60)
        finally:
            stop.set()
            if worker: worker.join(timeout=40)
    print('ras', json.dumps(_ras()), flush=True)
    print(out, flush=True)

if __name__ == '__main__': main()
