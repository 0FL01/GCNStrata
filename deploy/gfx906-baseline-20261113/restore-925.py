#!/usr/bin/env python3
"""Restore the captured working 05/08 tables after the unqualified 900 trial."""
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

import baseline as b

RUN = b.BASE / 'baseline-20261113T113944Z'


def main():
    original = b.c.inspect()
    identity = json.loads((RUN / 'identity.json').read_text())
    outcome = json.loads((RUN / 'qualification.json').read_text())
    recovery = json.loads((RUN / 'recovery-error.json').read_text())
    b.c.require(not outcome['passed'] and not recovery['faults'] and
                recovery['error'] == 'Candidate unqualified; production remains stopped',
                'Trial was not cleanly unqualified')
    b.c.require(not original['State']['Running'] and original['Id'] == identity['container_id'] and
                original['Image'] == b.EXPECTED_IMAGE and b.c.digest(b.CONFIG) == identity['config_sha256'] and
                b.c.digest(b.BINARY) == b.EXPECTED_BINARY, 'Original production state changed')
    b.c.require(not b.c.docker('ps', '-aq', '--filter', 'name=strata-unsloth64k-').strip(),
                'Benchmark container remains')
    b.c.require(not b.holders() and not b.cmd('ss', '-H', '-ltn', 'sport', '=', ':8080').strip(),
                'GPU client or API listener remains')
    baseline_ras = b.ras()
    b.c.require(all(v == 0 for group in baseline_ras.values() for v in group.values()) and
                not b.kernel_faults(identity['start_utc']), 'Hardware fault since candidate start')
    profiles = [(RUN / (device + '.pp_table')).read_bytes() for device in b.BDFS[:2]]
    fault, unsafe, stop, ready = [], threading.Event(), threading.Event(), threading.Event()
    started = False
    worker = None
    env = dict(x.split('=', 1) for x in original['Config']['Env'])
    key = env.get('STRATA_API_KEY', '')
    b.c.require(bool(key), 'API key missing')

    def monitor():
        try:
            while not stop.is_set():
                b.sensor()
                b.c.require(b.ras() == baseline_ras and not b.kernel_faults(identity['start_utc']),
                            'Hardware fault during production startup')
                ready.set()
                stop.wait(1)
        except BaseException as exc:
            fault.append(type(exc).__name__ + ': ' + str(exc))
            unsafe.set()
            ready.set()
            subprocess.run(['docker', 'stop', '--time', '15', b.c.PRODUCTION],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=60, check=False)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        b.apply_900(baseline_ras, identity['start_utc'], unsafe, profiles, fault, restore=True)
        b.c.require(b.tables() == b.EXPECTED_TABLES and not unsafe.is_set(), 'Working tables not restored')
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        b.c.require(ready.wait(45) and not unsafe.is_set(), 'Recovery monitor failed to start')
        started = True
        b.c.docker('start', b.c.PRODUCTION, timeout=180)
        end = time.monotonic() + 900
        loaded = False
        while time.monotonic() < end:
            b.c.require(not unsafe.is_set(), 'Monitor aborted restoration')
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
                    b.c.inspect()['State']['Running'], 'Production readiness failed')
        stop.set()
        worker.join(timeout=45)
        b.c.require(not worker.is_alive() and not unsafe.is_set() and b.ras() == baseline_ras and
                    not b.kernel_faults(identity['start_utc']), 'Recovery monitoring failed')
        b.cmd('sudo', '-n', 'systemctl', 'start', 'gpu-duct-fan-control.service', timeout=60)
        b.c.write_json(RUN / 'restore-925.json', {'passed': True, 'tables': b.tables(),
                                                 'production_loaded': True, 'fan_policy_active': True})
    except BaseException as exc:
        stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        if started:
            try:
                if b.c.inspect()['State']['Running']:
                    b.c.docker('stop', '--time', '15', b.c.PRODUCTION, timeout=60)
            except BaseException as stop_exc:
                fault.append('Production stop failed: ' + type(stop_exc).__name__)
        try:
            b.cmd('sudo', '-n', 'systemctl', 'stop', 'gpu-duct-fan-control.service', timeout=60)
            b.cmd('sudo', '-n', '/usr/local/sbin/fan-percent', '100', timeout=30)
            b.fan_state()
        except BaseException as fan_exc:
            fault.append('Fan100 failed: ' + type(fan_exc).__name__)
        b.c.write_json(RUN / 'restore-925-error.json', {'error': str(exc), 'faults': fault})
        raise


if __name__ == '__main__':
    main()
