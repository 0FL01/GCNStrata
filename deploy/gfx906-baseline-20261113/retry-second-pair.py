#!/usr/bin/env python3
"""One monitored retry of the failed 0b/0e 925 mV workload; never restores production."""
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

import baseline as b

RUN = b.BASE / 'all-925-20261113T120714Z'


def main():
    identity = json.loads((RUN / 'identity.json').read_text())
    original = b.c.inspect()
    cfg = json.loads(b.CONFIG.read_text())
    b.c.require(not original['State']['Running'] and original['Id'] == identity['container_id'] and
                original['Image'] == b.EXPECTED_IMAGE and b.c.digest(b.CONFIG) == identity['config_sha256'] and
                b.c.digest(b.BINARY) == b.EXPECTED_BINARY, 'Production identity/state changed')
    b.c.require(not b.holders() and not b.c.docker('ps', '-aq', '--filter',
                'name=strata-unsloth64k-').strip() and
                not b.cmd('ss', '-H', '-ltn', 'sport', '=', ':8080').strip(),
                'GPU client, test container or API listener remains')
    b.c.require(all((Path('/sys/bus/pci/devices') / d / 'pp_table').read_bytes() ==
                    (RUN / (d + '.target.pp_table')).read_bytes() for d in b.BDFS),
                'All-four 925 target tables changed')
    b.c.require(b.cmd('systemctl', 'show', '-p', 'ActiveState', '--value',
                      'gpu-duct-fan-control.service').strip() == b'inactive', 'Fan monitor active')
    b.sensor()
    ras_before = b.ras()
    b.c.require(all(v == 0 for group in ras_before.values() for v in group.values()) and
                not b.kernel_faults(identity['start_utc']), 'GPU hardware fault since first attempt')
    fixture = b.OLD / 'prompt-65536.ids.json'
    b.c.require(b.c.digest(fixture) == b.EXPECTED_FIXTURE, 'Fixture changed')
    ids = json.loads(fixture.read_text())
    b.c.require(len(ids) == b.c.N and all(type(t) is int and 0 <= t < b.c.VOCAB for t in ids),
                'Invalid fixture')
    env = dict(x.split('=', 1) for x in original['Config']['Env'])
    env.update({str(k): str(v) for k, v in cfg.get('env', {}).items()})
    env['HIP_VISIBLE_DEVICES'] = '2,3'
    env = {k: v for k, v in env.items() if not b.c.SECRET.search(k)}
    args = [a for a in cfg['args'] if a != '--vision']
    for flag in ('--prompt-cache', '--prompt-cache-every', '--conversation-cache-mib',
                 '--conversation-cache-slots'):
        b.c.replace_arg(args, flag, 0)
    b.c.replace_arg(args, '--eos-ids', '2147483647')
    out = b.BASE / ('retry-second-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    out.mkdir(mode=0o700)
    unsafe, stop, ready = threading.Event(), threading.Event(), threading.Event()
    names, faults = [], []

    def monitor():
        try:
            with (out / 'telemetry.jsonl').open('x') as f:
                while not stop.is_set():
                    sample = b.sensor()
                    b.c.require(b.ras() == ras_before and not b.kernel_faults(identity['start_utc']),
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
                stop.wait(1)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        b.c.require(ready.wait(45) and not unsafe.is_set(), 'Monitor failed to start')
        result = b.run_case('pair-0b-0e', args, cfg, original, env, ids, out, unsafe, names)
        b.c.require(not unsafe.is_set(), 'Monitor aborted workload')
        b.c.write_json(out / 'results.json', [result])
    except BaseException as exc:
        b.c.write_json(out / 'error.json', {'error': str(exc), 'faults': faults})
        raise
    finally:
        stop.set()
        worker.join(timeout=45)
        if worker.is_alive():
            raise RuntimeError('Monitor did not stop; leave production stopped')
    print('Result directory:', out, flush=True)


if __name__ == '__main__':
    main()
