#!/usr/bin/env python3
"""Prepared only; execution needs explicit approval and externally quiesced ingress.

The final idle check is NOT an atomic admission barrier. SIGINT/SIGTERM and ordinary
failures trigger bounded restoration. SIGKILL/host/Docker failure need operator recovery.
Native engines omit API credentials; all other production effective env is retained.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid

BASE = Path(__file__).resolve().parent
DOCKER = shutil.which('docker') or '/usr/bin/docker'
PRODUCTION = 'strata-api'
SECRET = re.compile(r'KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL', re.I)
N, OUTPUT, VOCAB = 65536, 512, 248320


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def docker(*args, timeout=30):
    p = subprocess.run([DOCKER, *args], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    require(p.returncode == 0, 'Docker operation failed: ' + args[0])
    return p.stdout


def inspect():
    return json.loads(docker('inspect', PRODUCTION))[0]


def api(path, key):
    request = urllib.request.Request('http://127.0.0.1:8080' + path,
                                    headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def gpu():
    result, seen = [], set()
    for p in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
        if not (p / 'mem_info_vram_used').exists() or p.resolve() in seen:
            continue
        seen.add(p.resolve())
        result.append({'device': str(p.resolve()),
                       'busy_percent': int((p / 'gpu_busy_percent').read_text()),
                       'vram_used_bytes': int((p / 'mem_info_vram_used').read_text())})
    require(len(result) >= 2, 'Cannot establish idle state of both production GPUs')
    return result


def idle_gate(key):
    samples = []
    for _ in range(3):
        status, v1, cards = api('/status', key), api('/v1/status', key), gpu()
        require(status.get('busy') is False and status.get('queued') == 0,
                'Production active/queued or status schema changed; refusing stop')
        require(v1.get('loaded') is True and
                v1.get('activity', {}).get('in_flight') == 0,
                'Production not loaded/idle; refusing stop')
        require(all(0 <= g['busy_percent'] <= 3 for g in cards),
                'GPU activity exceeds idle threshold; refusing stop')
        # Raw status can contain user request text, so retain only these fields.
        samples.append({'busy': False, 'queued': 0, 'in_flight': 0, 'gpus': cards})
        time.sleep(2)
    return samples


def replace_arg(args, flag, value):
    require(args.count(flag) <= 1, 'Duplicate production flag: ' + flag)
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args.extend([flag, str(value)])


def host_path(path, mounts):
    for m in sorted(mounts, key=lambda m: -len(m['Destination'])):
        dest = m['Destination']
        if path == dest or path.startswith(dest + '/'):
            return Path(m['Source']) / path[len(dest):].lstrip('/')
    raise RuntimeError('Path is not covered by production mounts: ' + path)


def command(name, cfg, original, env, args):
    hc = original['HostConfig']
    cmd = [DOCKER, 'run', '--rm', '-i', '--name', name, '--network', 'none',
           '--restart', 'no', '--workdir', cfg['cwd'], '--entrypoint', cfg['exe'],
           '--ipc', hc['IpcMode'], '--runtime', hc['Runtime'],
           '--shm-size', str(hc['ShmSize'])]
    if original['Config']['User']:
        cmd += ['--user', original['Config']['User']]
    for d in hc['Devices']:
        cmd += ['--device', ':'.join([d['PathOnHost'], d['PathInContainer'],
                                     d['CgroupPermissions']])]
    for g in hc['GroupAdd'] or []:
        cmd += ['--group-add', g]
    for u in hc['Ulimits'] or []:
        cmd += ['--ulimit', f"{u['Name']}={u['Soft']}:{u['Hard']}"]
    for s in hc['SecurityOpt'] or []:
        cmd += ['--security-opt', s]
    for m in original['Mounts']:
        require(m['Type'] == 'bind' and m['Propagation'] == 'rprivate',
                'Unexpected mount type/propagation')
        cmd += ['--volume', m['Source'] + ':' + m['Destination'] +
                (':rw' if m['RW'] else ':ro')]
    # Names only in argv. Values stay in-memory, never in an env file.
    for k in sorted(env):
        cmd += ['--env', k]
    return cmd + [original['Image'], '--serve', *args]


def cleanup(name, process):
    # Only the isolated container created under our fresh UUID; never production.
    subprocess.run([DOCKER, 'rm', '--force', name], timeout=60,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if process is not None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    # --rm may already have removed it; prove no orphan remains with a successful
    # daemon query, rather than treating any failed inspect as proof of absence.
    remaining = docker('ps', '--all', '--quiet', '--filter', 'name=^/' + name + '$')
    require(not remaining.strip(), 'Isolated engine container cleanup failed')


def parse_done(line, tokens, prompt_n, generated):
    f = line.split()
    require(len(f) >= 16, 'DONE lacks current prompt_read/offload fields')
    metrics = {'generated': int(f[1]), 'prompt_tokens': int(f[2]),
               'prompt_ms': float(f[3]), 'decode_ms': float(f[4]), 'finish': f[5],
               'drafts_accepted': int(f[6]), 'drafts_offered': int(f[7]),
               'reused': int(f[8]), 'hits': int(f[9]), 'lookups': int(f[10]),
               'ram_blobs': int(f[11]), 'file_blobs': int(f[12]),
               'file_mb': float(f[13]), 'prompt_read': int(f[14]),
               'offloaded': int(f[15]), 'DONE': line.rstrip(), 'output_ids': tokens}
    require(metrics['generated'] == len(tokens) == generated, 'Output count mismatch')
    require(metrics['prompt_tokens'] == metrics['prompt_read'] == prompt_n and
            metrics['reused'] == 0, 'Prompt reused or not read in full')
    require(metrics['finish'] == 'length', 'Unexpected finish (including nonfinite)')
    require(all(type(t) is int and 0 <= t < VOCAB for t in tokens), 'Invalid output IDs')
    require(all(math.isfinite(metrics[k]) for k in ['prompt_ms', 'decode_ms', 'file_mb'])
            and metrics['prompt_ms'] > 0 and metrics['decode_ms'] > 0,
            'Invalid/nonfinite native timings')
    metrics['pp_tok_s'] = prompt_n * 1000 / metrics['prompt_ms']
    metrics['tg_tok_s'] = len(tokens) * 1000 / metrics['decode_ms']
    return metrics


def run_case(tag, args, cfg, original, env, ids, out, limit,
             generated=OUTPUT, embeddings=None):
    name = 'strata-unsloth64k-' + uuid.uuid4().hex[:12]
    cmd = command(name, cfg, original, env, args)
    write_json(out / (tag + '.manifest.json'),
               {'argv': cmd, 'effective_environment_without_secrets': env,
                'fixture_sha256': digest(BASE / 'prompt-65536.ids.json'),
                'prompt_tokens': len(ids), 'generated_tokens': generated,
                'embeddings': embeddings, 'bound_s': limit, 'fresh_engine': True})
    lines = queue.Queue()
    tokens, info = [], {}
    process = reader = None
    start = time.monotonic()
    request_start = ready = metrics = None
    raw = (out / (tag + '.stdout')).open('xb')
    err = (out / (tag + '.stderr')).open('xb')
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=err,
                                   env={**os.environ, **env}, bufsize=0)

        def pump():
            try:
                for b in process.stdout:
                    raw.write(b)
                    raw.flush()
                    lines.put(b.decode('utf-8', errors='strict'))
            finally:
                lines.put(None)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        while metrics is None:
            require(time.monotonic() - start < limit, 'Per-run time bound exceeded')
            require(not (BASE / 'cancel').exists(), 'Cancellation marker found')
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                continue
            require(line is not None, 'Native engine exited before DONE')
            require(not line.startswith(('ERR', 'FATAL')), 'Native engine error; see raw logs')
            if line.startswith('INFO '):
                info.update(kv.split('=', 1) for kv in line.split()[1:] if '=' in kv)
            elif line.startswith('RESUME '):
                require(int(line.split()[1]) == 0, 'Native prompt reuse detected')
            elif line.startswith('READY '):
                require(ready is None and int(line.split()[1]) == 262144,
                        'Wrong/repeated READY context')
                require(info.get('kv') == 'int8' and info.get('context') == '262144',
                        'INFO does not match production KV/context')
                ready = time.monotonic()
                request_start = time.monotonic()
                head = 'GENI' if embeddings else 'GEN'
                request = (f'{head} {generated} temperature=1 top_p=0.95 top_k=20 seed=12345 '
                           + (embeddings + ' ' if embeddings else '')
                           + ','.join(map(str, ids)) + '\n').encode()

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

                # A stuck pipe writer cannot defeat the main thread deadline.
                threading.Thread(target=send, daemon=True).start()
            elif line.startswith('T '):
                require(request_start is not None, 'Token before request')
                tokens.append(int(line[2:]))
            elif line.startswith('DONE '):
                require(request_start is not None, 'DONE before request')
                metrics = parse_done(line, tokens, len(ids), generated)
                metrics.update(load_startup_wall_s=ready - start,
                               request_wall_s=time.monotonic() - request_start,
                               engine_info=info, tag=tag, passed=True)
        process.stdin.write(b'QUIT\n')
        process.stdin.flush()
        process.stdin.close()
        require(process.wait(timeout=120) == 0, 'Native engine nonzero exit')
        reader.join(timeout=10)
        require(not reader.is_alive(), 'Stdout did not close after exit')
    except BaseException as exc:
        write_json(out / (tag + '.result.json'),
                   {'tag': tag, 'passed': False, 'error_type': type(exc).__name__,
                    'error': str(exc), 'output_count': len(tokens)})
        raise
    finally:
        try:
            cleanup(name, process)
            if reader is not None:
                reader.join(timeout=10)
        finally:
            raw.close()
            err.close()
    write_json(out / (tag + '.result.json'), metrics)
    return metrics


def restore(original, cfgpath, config_hash, key, out):
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)
    try:
        require(inspect()['Id'] == original['Id'], 'Production container was replaced')
        docker('start', PRODUCTION, timeout=180)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                v1 = api('/v1/status', key)
                if (v1.get('loaded') is True and
                        v1.get('context', {}).get('native') == 262144 and
                        v1.get('vision', {}).get('enabled') is True):
                    require(digest(cfgpath) == config_hash, 'Production config changed')
                    state = inspect()
                    require(state['Id'] == original['Id'] and state['State']['Running'],
                            'Original production container is not running')
                    write_json(out / 'restore.json', {'passed': True,
                               'same_container': True, 'config_unchanged': True,
                               'loaded': True, 'context': 262144, 'vision': True})
                    return
            except (OSError, ValueError):
                pass
            time.sleep(5)
        raise RuntimeError('Production restoration readiness deadline exceeded')
    except BaseException as exc:
        write_json(out / 'restore.json', {'passed': False, 'error_type': type(exc).__name__,
                                         'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--idle-window-approved', action='store_true',
                        help='Operator has approval AND has quiesced production ingress')
    parser.add_argument('--compatibility-approved', action='store_true',
                        help='Parent verified native/pack/tokenizer/PLE/MTP compatibility')
    parser.add_argument('--unsloth-native', default='/models/qwen38-flash-next-ud-q2_k_xl/'
                        'Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00004.gguf')
    parser.add_argument('--unsloth-pack', default='/models/qwen38-flash-next-ud-q2_k_xl/pack')
    parser.add_argument('--unsloth-ple', default='/sata/shefowl-qwen38-hybrid/'
                        'Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0-00002-of-00002.gguf')
    parser.add_argument('--vision-smoke', type=Path,
                        help='Optional parent-prepared JSON with ids, embeddings, max_new')
    parser.add_argument('--run-timeout-s', type=int, default=5400)
    cli = parser.parse_args()
    require(cli.execute and cli.idle_window_approved and cli.compatibility_approved,
            'Requires --execute --idle-window-approved --compatibility-approved')
    require(60 <= cli.run_timeout_s <= 7200, 'Run bound must be 60..7200 seconds')
    lock = (BASE / 'controller.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    setup = json.loads((BASE / 'setup.json').read_text())
    cfgpath = Path(setup['config_path'])
    require(digest(cfgpath) == setup['config_sha256'], 'Production config changed since preparation')
    require(digest(setup['binary_path']) == setup['binary_sha256'], 'Qualified binary changed')
    require(digest(setup['server_protocol_path']) == setup['server_protocol_sha256'],
            'Server protocol source changed since preparation')
    require(digest(BASE / 'prompt-65536.ids.json') == setup['fixture']['sha256'], 'Fixture changed')
    ids = json.loads((BASE / 'prompt-65536.ids.json').read_text())
    require(len(ids) == N and all(type(t) is int and 0 <= t < VOCAB for t in ids), 'Invalid fixture')
    cfg, original = json.loads(cfgpath.read_text()), inspect()
    require(original['State']['Running'] and original['Image'] == setup['image'] and
            original['Id'] == setup['production_container_id'], 'Production identity/state changed')
    require(original['Mounts'] == setup['mounts'], 'Production mounts changed')
    for k, v in setup['host_config'].items():
        require(original['HostConfig'][k] == v, 'Production runtime changed: ' + k)
    require(not original['HostConfig']['Privileged'] and
            not original['HostConfig']['CapAdd'] and
            not original['HostConfig']['DeviceRequests'] and not cfg.get('lib_dirs'),
            'Unsupported runtime overrides; review controller')
    env = dict(v.split('=', 1) for v in original['Config']['Env'])
    key = env.get('STRATA_API_KEY', '')  # In memory only; never a file/log/argv.
    require(bool(key), 'Production API authentication is required')
    require(cfg.get('backend') == 'hip' and cfg.get('gpu') == [0, 1], 'GPU layout changed')
    env['HIP_VISIBLE_DEVICES'] = '0,1'
    env.update({str(k): str(v) for k, v in cfg.get('env', {}).items()})
    env = {k: v for k, v in env.items() if not SECRET.search(k)}
    common = [a for a in cfg['args'] if a != '--vision']
    require(not any(a.startswith('--expert-profile-save') for a in common),
            'Expert profile writes must not be enabled')
    for k, v in setup['methodology']['cache_changes'].items():
        replace_arg(common, k, v)
    replace_arg(common, '--eos-ids', '2147483647')
    for k, v in {'--expert-cache': '9900', '--prefill': '4096', '--kv': 'int8',
                 '--kv-resident': '32768', '--layer-split': '27',
                 '--max-context': '262144', '--mtp': '/src/runtime/mtp'}.items():
        require(common[common.index(k) + 1] == v, 'Production parameter changed: ' + k)
    require(cfg['exe'] == '/qualified-strata', 'Not using qualified production binary')
    candidate = list(common)
    for k, v in {'--pack': cli.unsloth_pack, '--native': cli.unsloth_native,
                 '--ple-gguf': cli.unsloth_ple}.items():
        replace_arg(candidate, k, v)
    # The baseline already explicitly names its three native shards. The new
    # candidate must name only its own four shards, not append to the baseline.
    while '--native-dense-gguf' in candidate:
        i = candidate.index('--native-dense-gguf')
        del candidate[i:i + 2]
    require(cli.unsloth_native.endswith('-00001-of-00004.gguf'), 'Expected four-shard candidate')
    for number in (1, 2, 3, 4):
        candidate += ['--native-dense-gguf', cli.unsloth_native.replace(
            '-00001-of-00004.gguf', f'-{number:05d}-of-00004.gguf')]
    for args in (common, candidate):
        for flag in ['--pack', '--native', '--ple-gguf', '--mtp', '--expert-profile']:
            path = host_path(args[args.index(flag) + 1], original['Mounts'])
            require(path.exists(), 'Required path absent: ' + str(path))
    smoke = None
    if cli.vision_smoke:
        smoke = json.loads(cli.vision_smoke.read_text())
        require(isinstance(smoke.get('ids'), list) and 0 < len(smoke['ids']) < 262144 and
                all(type(t) is int and 0 <= t < VOCAB for t in smoke['ids']), 'Invalid smoke IDs')
        require(type(smoke.get('max_new')) is int and 1 <= smoke['max_new'] <= 256,
                'Smoke max_new must be 1..256')
        require(isinstance(smoke.get('embeddings'), str) and
                not any(c.isspace() for c in smoke['embeddings']) and
                smoke['embeddings'].startswith('/src/deploy/unsloth-q2-vision-20261110/'),
                'Smoke embeddings must be under the parent vision experiment mount')
        require(host_path(smoke['embeddings'], original['Mounts']).is_file(), 'Embeddings absent')
    require(not (BASE / 'cancel').exists(), 'Cancellation marker present')
    out = BASE / ('run-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                  + '-' + uuid.uuid4().hex[:6])
    out.mkdir(mode=0o700)
    if smoke:
        write_json(out / 'vision-smoke.input.json', smoke)
        write_json(out / 'vision-smoke.evidence.json',
                   {'input_json': str(cli.vision_smoke), 'input_sha256': digest(cli.vision_smoke),
                    'embeddings_sha256': digest(host_path(smoke['embeddings'], original['Mounts']))})

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Controller interrupted')

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    results, stop_attempted = [], False
    try:
        write_json(out / 'idle-gate.json', idle_gate(key))
        status = api('/status', key)
        require(status.get('busy') is False and status.get('queued') == 0 and
                all(g['busy_percent'] <= 3 for g in gpu()), 'Idle window lost')
        stop_attempted = True
        docker('stop', '--time', '120', PRODUCTION, timeout=180)
        require(not inspect()['State']['Running'], 'Production did not stop')
        if smoke:
            require(all(g['busy_percent'] <= 3 for g in gpu()), 'GPU not idle before smoke')
            results.append(run_case('vision-smoke', [*candidate, '--vision'], cfg, original,
                                    env, smoke['ids'], out, cli.run_timeout_s,
                                    smoke['max_new'], smoke['embeddings']))
        for tag, args in [('prod-q2_0-1', common), ('q2_k_xl-1', candidate),
                          ('q2_k_xl-2', candidate), ('prod-q2_0-2', common)]:
            require(all(g['busy_percent'] <= 3 for g in gpu()), 'GPU not idle between engines')
            results.append(run_case(tag, args, cfg, original, env, ids, out, cli.run_timeout_s))
    finally:
        # Restoration must run even if evidence writing fails (e.g. full disk).
        try:
            write_json(out / 'results.partial.json', results)
        finally:
            if stop_attempted:
                restore(original, cfgpath, setup['config_sha256'], key, out)
    summaries = {}
    for arm in ['prod-q2_0', 'q2_k_xl']:
        cases = [r for r in results if r['tag'].startswith(arm)]
        summaries[arm] = {k: {'runs': [r[k] for r in cases],
                             'mean': statistics.mean(r[k] for r in cases)}
                          for k in ['pp_tok_s', 'tg_tok_s', 'load_startup_wall_s', 'request_wall_s']}
    write_json(out / 'summary.json', {'passed': True, 'results': results, 'arms': summaries,
               'metric_scope': 'PP=prompt/native prompt_ms; TG=actual output/native decode_ms. '
               'Last prompt token runs in first decode window. Load/startup and request wall separate. '
               'Fresh engines, not cold OS page cache; no output equality or quality claim.'})
    print(str(out))


if __name__ == '__main__':
    main()
