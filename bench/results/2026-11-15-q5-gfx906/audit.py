#!/usr/bin/env python3
"""Task-local Q3 prefix-logit / fresh-engine ABBA runner. Parent owns production."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import uuid

ROOT = Path('/home/radneon/strata-rd/q5-audit-20261114')
PROD = Path('/home/radneon/strata')
CONFIG = PROD / 'configs/unsloth-q3-k-xl-4gpu-f16-iq4-20261114.json'
IMAGE = 'sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
FIXTURE = PROD / 'deploy/unsloth-q2-k-xl-20261111/prompt-65536.ids.json'
FIXTURE_SHA = '63ce3d4042027a00b2193feab143f1d9443b5c31a863a962ec9543ccb54d9695'
VOCAB = 248320
CACHES = ('--prompt-cache', '--prompt-cache-every', '--conversation-cache-mib', '--conversation-cache-slots')
PATH_FLAGS = ('--pack', '--native', '--native-dense-gguf', '--ple-gguf', '--mtp', '--expert-profile', '--mtp-draft-vocab')


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def replace(args, key, value):
    while key in args:
        at = args.index(key)
        del args[at:at + 2]
    args.extend((key, str(value)))


def value(args, key):
    return args[args.index(key) + 1]


def gpu_state():
    out = []
    for d in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
        if not (d / 'mem_info_vram_used').exists():
            continue
        row = {'device': str(d.resolve())}
        for name in ('gpu_busy_percent', 'mem_info_vram_used', 'pp_dpm_sclk', 'power_dpm_force_performance_level'):
            f = d / name
            row[name] = f.read_text().strip() if f.exists() else None
        for f in sorted(d.glob('hwmon/hwmon*/temp*_input')):
            row[f.name] = f.read_text().strip()
        out.append(row)
    return out


def idle():
    # This is a final interlock, not a replacement for the parent's ingress/queue/holder check.
    running = subprocess.check_output(['docker', 'ps', '-q', '--filter', 'name=^/strata-api$'], text=True)
    require(not running.strip(), 'Production is still running; only the parent may stop it')
    holders = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target == '/dev/kfd' or target.startswith('/dev/dri/renderD'):
                    holders.append(int(proc.name))
                    break
        except (PermissionError, FileNotFoundError):
            continue
    require(not holders, 'Visible GPU holders remain: ' + str(holders))
    state = gpu_state()
    require(len(state) == 4, 'Expected four AMD GPUs')
    require(all(int(x['gpu_busy_percent']) == 0 for x in state), 'GPU activity remains')
    require(Path('/proc/sys/vm/compact_unevictable_allowed').read_text().strip() == '0',
            'Production memory-lock prerequisite vm.compact_unevictable_allowed=0 is absent')
    return state


def clock_tables():
    return ['\n'.join(line.replace('*', '').strip() for line in r['pp_dpm_sclk'].splitlines())
            for r in gpu_state()]


def host_path(s):
    for mount, host in (('/src', PROD), ('/models', Path('/home/radneon/models-nvme')),
                        ('/sata', Path('/home/radneon/models'))):
        if s.startswith(mount + '/'):
            return host / s[len(mount) + 1:]
    raise RuntimeError('Unexpected input mount: ' + s)


def settings(quality_tokens=None):
    cfg = json.loads(CONFIG.read_text())
    require(cfg['gpu'] == [0, 1, 2, 3], 'Expected current four-GPU config')
    args = [s.replace('/models/qwen38-flash-next-ud-q3_k_xl', '/sata/qwen38-flash-next-ud-q3_k_xl') if s.endswith('.gguf') else s
            for s in cfg['args'] if s != '--vision']
    require('--expert-profile-save' not in args, 'Profile writes forbidden')
    for flag in CACHES:
        replace(args, flag, 0)
    replace(args, '--eos-ids', 2147483647)
    if quality_tokens is not None:
        replace(args, '--short-read', quality_tokens + 64)
    require(args.count('--native-dense-gguf') == 4, 'Expected four explicit Q3 dense shards')
    for flag, want in (('--kv', 'int8'), ('--spec', '6'), ('--mtp-max-t', '2'),
                       ('--suffix-draft', '0'), ('--pipeline-windows', '2'), ('--max-context', '262144')):
        require(value(args, flag) == want, 'Config changed: ' + flag)
    inputs = []
    for i, flag in enumerate(args):
        if flag in PATH_FLAGS:
            f = host_path(args[i + 1])
            st = f.stat()
            inputs.append({'flag': flag, 'path': str(f), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns})
    env = {k: str(v) for k, v in cfg.get('env', {}).items()}
    require(all(k.startswith('STRATA_') or k in ('DO_NOT_TRACK', 'HF_HUB_DISABLE_TELEMETRY') for k in env),
            'Unexpected config environment key; review before recording')
    require(not any(word in k.upper() for k in env for word in ('SECRET', 'PASSWORD', 'API_KEY', 'TOKEN')),
            'Sensitive config environment key; refusing to record')
    require(not any(k.startswith(('STRATA_DUMP', 'STRATA_LOGPOS')) for k in env), 'Unexpected diagnostic env')
    env['HIP_VISIBLE_DEVICES'] = '0,1,2,3'
    return {'args': args, 'env': env, 'inputs': inputs, 'config_sha256': digest(CONFIG),
             'image': IMAGE, 'requested_clock_mhz': 1350,
             'clock_qualification': 'User approved existing 1350 MHz profile for matched A/B; no clock writes'}


def parse_done(line, ids, tokens, new):
    f = line.split()
    require(len(f) >= 16 and f[0] == 'DONE', 'Malformed DONE')
    require(int(f[1]) == len(tokens) == new and int(f[2]) == len(ids) and f[5] == 'length', 'Incomplete GEN')
    require(int(f[8]) == 0 and int(f[14]) == len(ids), 'Prompt was reused or not fully read')
    require(all(type(t) is int and 0 <= t < VOCAB for t in tokens), 'Invalid output token')
    pp, tg, accepted, offered = float(f[3]), float(f[4]), int(f[6]), int(f[7])
    require(pp >= 0 and tg > 0 and 0 <= accepted <= offered, 'Invalid timings/acceptance')
    return {'DONE': line.strip(), 'prompt_tokens': len(ids), 'output_ids': tokens, 'prompt_ms': pp,
            'decode_ms': tg, 'PP': len(ids) * 1000 / pp if pp else None, 'TG': new * 1000 / tg,
            'drafts_accepted': accepted, 'drafts_offered': offered,
            'MTP_acceptance': accepted / offered if offered else None}


def engine(root, out, variant, cfg, requests, quality):
    before = idle()
    name = 'strata-q5-audit-' + uuid.uuid4().hex[:12]
    env = dict(cfg['env'])
    if quality:
        env['STRATA_DUMP_FIRST_LOGITS'] = '/out/first'
    cmd = ['docker', 'run', '--rm', '-i', '--name', name, '--network', 'none',
           '--device', '/dev/kfd', '--device', '/dev/dri', '--ipc', 'host',
           '--ulimit', 'memlock=-1:-1', '--user', f'{os.getuid()}:{os.getgid()}',
           '-w', '/src', '-v', f'{PROD}:/src:ro', '-v', f'{root}:/audit:ro', '-v', f'{out}:/out',
           '-v', '/home/radneon/models-nvme:/models:ro', '-v', '/home/radneon/models:/sata:ro',
           '-v', '/home/radneon/models-nvme/shared-engram:/home/radneon/models-nvme/shared-engram:ro']
    for gid in sorted({p.stat().st_gid for p in [Path('/dev/kfd'), *Path('/dev/dri').glob('renderD*')]}):
        cmd.extend(('--group-add', str(gid)))
    for k, v in sorted(env.items()):
        cmd.extend(('-e', k + '=' + v))
    cmd.extend(('--entrypoint', '/audit/' + variant + '/strata', IMAGE, '--serve', *cfg['args']))
    out.mkdir()
    manifest = {'variant': variant, 'binary_sha256': digest(root / variant / 'strata'), 'settings': cfg,
                'argv': cmd, 'gpu_before': before, 'started_unix_s': time.time(), 'passed': False}
    save(out / 'manifest.json', manifest)
    save(out / 'requests.json', requests)
    p = None
    start = time.monotonic()
    lines = queue.Queue()
    results, info = [], {}
    try:
        with (out / 'stderr').open('x') as err, (out / 'stdout').open('x') as raw, \
                (out / 'gpu.jsonl').open('x') as samples:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                 text=True, bufsize=1)

            def pump():
                try:
                    for line in p.stdout:
                        raw.write(line)
                        raw.flush()
                        lines.put(line)
                finally:
                    lines.put(None)

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            last_sample = 0

            def next_line():
                nonlocal last_sample
                while True:
                    require(time.monotonic() - start < 21600, 'Six-hour per-engine deadline exceeded')
                    require(not (out.parent / 'cancel').exists(), 'Parent cancellation marker')
                    if time.monotonic() - last_sample >= 5:
                        samples.write(json.dumps({'elapsed_s': time.monotonic() - start, 'gpus': gpu_state()}) + '\n')
                        samples.flush()
                        last_sample = time.monotonic()
                    try:
                        line = lines.get(timeout=1)
                    except queue.Empty:
                        continue
                    require(line is not None, 'Native engine exited early')
                    require(not line.startswith(('ERR ', 'FATAL')), line.strip())
                    return line

            while True:
                line = next_line()
                if line.startswith('INFO '):
                    info.update(kv.split('=', 1) for kv in line.split()[1:] if '=' in kv)
                if line.startswith('READY '):
                    require(line.split()[1] == '262144', 'Unexpected READY context')
                    for k, want in (('context', '262144'), ('kv', 'int8'), ('spec', '6'), ('mtp_max', '2'), ('lookup', '0')):
                        require(info.get(k) == want, 'Unexpected INFO ' + k)
                    require('locked in RAM' in (out / 'stderr').read_text(), 'PLE RAM lock not confirmed')
                    manifest.update(engine_info=info, load_wall_s=time.monotonic() - start)
                    save(out / 'manifest.json', manifest)
                    break
            for idx, request in enumerate(requests):
                tokens, ids, new = [], request['ids'], request['new']
                sample = 'temperature=0 seed=12345' if quality else 'temperature=1 top_p=0.95 top_k=20 seed=12345'
                head = f'GEN {new} {sample} aux_cpus=1'
                t0 = time.monotonic()
                p.stdin.write(head + ' ' + ','.join(map(str, ids)) + '\n')
                p.stdin.flush()
                while True:
                    line = next_line()
                    if line.startswith('RESUME '):
                        require(int(line.split()[1]) == 0, 'Unexpected prompt reuse')
                    if line.startswith('T '):
                        tokens.append(int(line.split()[1]))
                    if line.startswith('DONE '):
                        result = parse_done(line, ids, tokens, new)
                        break
                result.update(index=idx, label=request['label'], header=head, request_wall_s=time.monotonic() - t0)
                if quality:
                    require('chunk=0' in result['DONE'].split(), 'Quality prefix unexpectedly took batched prefill')
                    f = out / f'first.{idx}'
                    require(f.stat().st_size == 4 * VOCAB, 'Missing/short full-vocabulary logit row')
                    result['logits_sha256'] = digest(f)
                results.append(result)
                save(out / 'results.json', results)
                print(json.dumps({'case': out.name, 'request': idx, 'label': request['label'], 'completed': True}), flush=True)
            p.stdin.write('QUIT\n')
            p.stdin.flush()
            p.stdin.close()
            require(p.wait(timeout=600) == 0, 'Native engine exit failed')
            reader.join(timeout=10)
            require(not reader.is_alive(), 'Stdout reader did not finish')
        manifest['passed'] = True
    except BaseException as exc:
        manifest['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        # Own UUID container only. Production restart is exclusively the parent's job.
        try:
            subprocess.run(['docker', 'rm', '--force', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120, check=False)
            if p is not None and p.poll() is None:
                p.wait(timeout=120)
            remaining = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'], text=True)
            require(not remaining.strip(), 'Isolated container remains; parent must inspect')
            # Sysfs utilization is sampled and can outlive the exited container briefly.
            for _ in range(30):
                if all(int(x['gpu_busy_percent']) == 0 for x in gpu_state()):
                    break
                time.sleep(1)
            manifest['gpu_after'] = idle()
            manifest['cleanup_passed'] = True
        except Exception as exc:
            manifest['cleanup_error'] = str(exc)
            if 'error' not in manifest:
                raise
        finally:
            manifest.update(elapsed_s=time.monotonic() - start, native_exit=p.returncode if p else None)
            save(out / 'manifest.json', manifest)
    return results


def compare(directory):
    import numpy as np  # Already used by tools/quant_quality.py pcompare; no GPU needed.
    a, b = directory / 'baseline', directory / 'candidate'
    ma, mb = [json.loads((p / 'manifest.json').read_text()) for p in (a, b)]
    require(all(m.get('passed') and m.get('cleanup_passed') for m in (ma, mb)), 'Incomplete quality run')
    require(ma['settings'] == mb['settings'], 'Unmatched settings')
    require((a / 'requests.json').read_bytes() == (b / 'requests.json').read_bytes(), 'Unmatched teacher-forced IDs')
    requests = json.loads((a / 'requests.json').read_text())
    results = [json.loads((p / 'results.json').read_text()) for p in (a, b)]
    rows, first_numeric, first_top1 = [], None, None
    for i, request in enumerate(requests):
        pair = []
        for p, records in zip((a, b), results):
            f = p / f'first.{i}'
            require(digest(f) == records[i]['logits_sha256'], 'Logit artifact changed')
            x = np.fromfile(f, dtype='<f4').astype(np.float64)
            require(len(x) == VOCAB and np.isfinite(x).all(), 'Nonfinite/truncated full logits')
            pair.append(x)
        x, y = pair
        lx, ly = x - x.max(), y - y.max()
        lx -= np.log(np.exp(lx).sum())
        ly -= np.log(np.exp(ly).sum())
        kl = float(np.sum(np.exp(lx) * (lx - ly)))
        require(kl >= -1e-12, 'Numerically invalid KL')
        row = {'index': i, 'label': request['label'], 'prefix_tokens': len(request['ids']),
               'target_id': request['target_id'], 'baseline_top1': int(x.argmax()), 'candidate_top1': int(y.argmax()),
               'kl_baseline_candidate': max(0.0, kl), 'max_abs_logit_delta': float(np.abs(x - y).max())}
        if first_numeric is None and not np.array_equal(x, y):
            first_numeric = dict(row)
        if first_top1 is None and row['baseline_top1'] != row['candidate_top1']:
            first_top1 = dict(row)
            top = sorted(set(np.argsort(x)[-5:].tolist() + np.argsort(y)[-5:].tolist()))
            first_top1['logits'] = [{'id': t, 'baseline': float(x[t]), 'candidate': float(y[t])} for t in top]
        rows.append(row)
    kls = sorted(r['kl_baseline_candidate'] for r in rows)
    require(bool(kls), 'No matched rows')
    summary = {'positions': len(rows), 'vocab': VOCAB, 'kl_direction': 'P_baseline || P_candidate',
               'kl_mean': math.fsum(kls) / len(kls), 'kl_p99': kls[int(.99 * (len(kls) - 1))],
               'p99_definition': 'sorted[floor(0.99*(N-1))], matching quant_quality.py',
               'top1_agreement': sum(r['baseline_top1'] == r['candidate_top1'] for r in rows) / len(rows),
               'first_nonidentical_logits': first_numeric, 'first_top1_disagreement': first_top1,
               'scope': 'all-vocabulary logits after fixed prefixes; no sampled token is fed back'}
    save(directory / 'quality-rows.json', rows)
    save(directory / 'quality-summary.json', summary)
    print(json.dumps(summary, indent=2))


def run(a):
    quality = a.mode == 'quality'
    a.root, a.out, a.evidence = a.root.resolve(), a.out.resolve(), a.evidence.resolve()
    require(a.evidence.is_file(), 'Need the parent numerical reproduction artifact')
    cfg = settings(a.tokens if quality else None)
    if quality:
        require(a.tokens is not None and a.tokens >= 2, '--tokens >= 2 is required for quality')
        corpus = json.loads(a.ids.read_text())
        requests = []
        for s in corpus['segments']:
            require(len(s['ids']) >= a.tokens, 'Requested quality prefix exceeds corpus segment')
            for n in range(1, a.tokens):
                requests.append({'label': f"{s['name']}:{n-1}", 'ids': s['ids'][:n], 'new': 1, 'target_id': s['ids'][n]})
        input_path = a.ids
    else:
        require(digest(a.fixture) == FIXTURE_SHA, 'fixture64K512 SHA256 mismatch')
        ids = json.loads(a.fixture.read_text())
        require(len(ids) == 65536, 'Need exactly 65536 prompt tokens')
        requests = [{'label': 'fixture64K512', 'ids': ids, 'new': 512}]
        input_path = a.fixture
    require(all(type(t) is int and 0 <= t < VOCAB for r in requests for t in r['ids']), 'Invalid fixture IDs')
    a.out.mkdir(parents=True, exist_ok=False)
    identity = {'mode': a.mode, 'settings': cfg, 'input_path': str(input_path), 'input_sha256': digest(input_path),
                'reproduction_evidence': str(a.evidence), 'reproduction_sha256': digest(a.evidence),
                'binary_sha256': {v: digest(a.root / v / 'strata') for v in ('baseline', 'candidate')},
                'clock_tables': clock_tables(),
                'metric_scope': 'PP=n/native prompt_ms; last prompt token in decode; TG=new/native decode_ms; load separate; OS cache not flushed'}
    save(a.out / 'identity.json', identity)
    order = [('baseline', 'baseline'), ('candidate', 'candidate')] if quality else [
        ('A1', 'baseline'), ('B1', 'candidate'), ('B2', 'candidate'), ('A2', 'baseline')]
    runs = []
    for tag, variant in order:
        require(settings(a.tokens if quality else None) == cfg, 'Settings or input metadata changed during series')
        require(clock_tables() == identity['clock_tables'],
                'DPM tables changed during series')
        require(digest(a.root / variant / 'strata') == identity['binary_sha256'][variant], 'Binary changed during series')
        result = engine(a.root, a.out / tag, variant, cfg, requests, quality)
        runs.append({'tag': tag, 'variant': variant, 'results': result})
        save(a.out / 'series.json', runs)
    if not quality:
        summary = {'requested_clock_mhz': cfg['requested_clock_mhz'], 'clock_qualification': cfg['clock_qualification'], 'variants': {}}
        for v in ('baseline', 'candidate'):
            rr = [r['results'][0] for r in runs if r['variant'] == v]
            summary['variants'][v] = {k: sum(r[k] for r in rr) / len(rr) for k in ('PP', 'TG')}
            offered, accepted = sum(r['drafts_offered'] for r in rr), sum(r['drafts_accepted'] for r in rr)
            summary['variants'][v].update(MTP_acceptance=accepted / offered if offered else None,
                                         repeated_output_identical=rr[0]['output_ids'] == rr[1]['output_ids'])
        x, y = [runs[i]['results'][0]['output_ids'] for i in (0, 1)]
        summary['A1_B1_first_free_generation_difference'] = next(
            ({'zero_based_output_position': i, 'baseline_id': u, 'candidate_id': v} for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
        summary['free_generation_scope'] = 'Same prompt and seed; subsequent contexts diverge after first differing output. Not teacher-forced KL.'
        save(a.out / 'performance-summary.json', summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    for mode in ('quality', 'perf'):
        q = sub.add_parser(mode)
        q.add_argument('--root', type=Path, default=ROOT)
        q.add_argument('--out', type=Path, required=True)
        q.add_argument('--evidence', type=Path, required=True)
        q.add_argument('--ids', type=Path, default=ROOT / 'src/data/quant_quality/v1/set.ids.json')
        q.add_argument('--fixture', type=Path, default=FIXTURE)
        q.add_argument('--tokens', type=int)
    q = sub.add_parser('compare')
    q.add_argument('directory', type=Path)
    a = p.parse_args()
    if a.mode == 'compare':
        compare(a.directory)
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        run(a)


if __name__ == '__main__':
    main()
