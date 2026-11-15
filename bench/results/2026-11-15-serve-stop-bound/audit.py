#!/usr/bin/env python3
"""Native serve boundary probe: one resident engine, synthetic fixtures, immutable run directory.

Sweep sampled budgets once, plus one greedy control; continue the selected budget
using exact emitted IDs. Missing diagnostics are not evidence of a native failure.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
LIMITS = (1, 2, 3, 4, 5, 7, 8)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='new directory; existing paths refused')
    ap.add_argument('--scenario', choices=('sweep', 'case'), default='sweep')
    ap.add_argument('--max-new', type=int, default=4, help='selected case / sweep continuation budget')
    ap.add_argument('--continue-new', type=int, default=3)
    ap.add_argument('--route', choices=('live', 'park', 'both'), default='both')
    ap.add_argument('--sampling', choices=('sampled', 'greedy'), default='sampled')
    ap.add_argument('--seed', type=int, default=424242)
    ap.add_argument('--eos-ids', help='comma-separated token IDs; replaces the config EOS list')
    ap.add_argument('--pipeline-windows', '--pipeline1', type=int, choices=(1, 2),
                    nargs='?', const=1, help='override native pipeline-windows (1 exercises serial)')
    ap.add_argument('--serial-switch', action='store_true',
                    help='debug pw=1 per request, retaining pipeline2 startup allocation/cache')
    ap.add_argument('--theta', type=float, help='debug switch threshold; 2 suppresses speculative B')
    ap.add_argument('--replay', type=Path, help='run one saved request JSON in this fresh process')
    ap.add_argument('--expect', choices=('pass', 'fail', 'observe'), default='pass',
                    help='fail needs a measured invariant violation; observe preserves exploratory runs')
    ap.add_argument('--run', action='store_true', help='actually load the engine; otherwise dry run')
    a = ap.parse_args()
    if min(a.max_new, a.continue_new, a.seed) < 1 or a.seed >= 2**64:
        ap.error('budgets and seed must be positive; seed must fit uint64')
    if a.scenario == 'sweep' and not a.replay and a.max_new not in LIMITS:
        ap.error('sweep max-new must select one of 1,2,3,4,5,7,8')
    if a.theta is not None and (not math.isfinite(a.theta) or a.theta < 0):
        ap.error('theta must be finite and nonnegative')
    if a.serial_switch and a.pipeline_windows is not None:
        ap.error('serial-switch must preserve the configured startup pipeline allocation')
    eos = None
    if a.eos_ids is not None:
        try:
            eos = [int(x) for x in a.eos_ids.split(',')]
            if not eos or any(x < 0 or x >= 2**31 for x in eos):
                raise ValueError
        except ValueError:
            ap.error('eos-ids must be comma-separated nonnegative int32 IDs')
    if not a.run:
        print(f'Dry run: {a.scenario}, {a.sampling}, route={a.route}; no engine loaded.')
        return 0

    # Import the existing tokenizer/engine only after --help and dry-run have returned.
    from conversation_cache_parity import load_tokenizer
    from serve.frontend import ChatTemplate
    from serve.server import StrataEngine, child_env

    cfg = json.loads(a.config.read_text())
    args = list(cfg['args'])
    for key, value in (('--pipeline-windows', a.pipeline_windows), ('--eos-ids', a.eos_ids)):
        if value is not None:
            while key in args:
                i = args.index(key)
                del args[i:i + 2]
            args += [key, str(value)]
    a.output = a.output.resolve()
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)

    def save(name, value):
        (a.output / name).write_text(json.dumps(value, indent=2) + '\n')

    env = child_env(cfg)
    env.update(STRATA_PIPELINE_DEBUG='1', STRATA_PIPELINE_LOG='1')
    if a.theta is not None or a.serial_switch:
        switch = a.output / 'pipeline-switch.txt'
        switch.write_text(('pw=1 ' if a.serial_switch else '') +
                          (f'theta={a.theta}' if a.theta is not None else '') + '\n')
        env['STRATA_PIPELINE_SWITCH'] = str(switch)
    sampled = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'seed': a.seed}
    greedy = {'temperature': 0.0, 'top_p': 1.0, 'top_k': 20, 'seed': a.seed}
    sampling = sampled if a.sampling == 'sampled' else greedy
    replay = json.loads(a.replay.read_text()) if a.replay else None
    if replay is None:
        p = Path(cfg['tokenizer'])
        tok, tpl = load_tokenizer(p), ChatTemplate(p / 'chat_template.jinja')
        texts = ['Write integers from 1 to 200, separated by comma and space. Output only the list.',
                 'Name three colors, one per line. Do not write any numbers.']
        rendered = [tpl.render([{'role': 'user', 'content': text}], enable_thinking=False) for text in texts]
        rendered[0] += '1, 2, 3, 4, 5, 6, 7, 8, '
        A, B = [tok.encode(text, parse_special=True) for text in rendered]
        save('fixtures.json', {'texts': texts, 'rendered': rendered, 'A': A, 'B': B})
    save('manifest.json', {
        'engine': str(a.engine), 'engine_sha256': sha256(a.engine), 'config': str(a.config),
        'config_sha256': sha256(a.config), 'runner_sha256': sha256(Path(__file__)),
        'args': args, 'cwd': cfg.get('cwd'), 'tokenizer': cfg['tokenizer'],
        'options': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'tuning': {k: env.get(k) for k in ('STRATA_IQ_MT_MIN', 'STRATA_PRIMARY_COMMIT_OVERLAP',
                   'STRATA_EXP_MODE', 'STRATA_HIP_PROMPT_F16', 'STRATA_PIPED_CONCURRENT',
                   'STRATA_PIPE_SHAPE_PREP', 'STRATA_PIPELINE_DEBUG', 'STRATA_PIPELINE_LOG')},
    })
    records, violations, coverage = [], [], []
    engine, error = None, None
    log = a.output / 'engine.log'

    def generate(name, ids, count, sp, route=None):
        record = {'name': name, 'prompt_ids': list(ids), 'max_new': count, 'sampling': dict(sp),
                  'ids': [], 'route_requested': route, 'log_start': log.stat().st_size}
        engine.last = {}  # never mistake a previous DONE for the current failed request
        try:
            for token in engine.generate(ids, count, sp, threading.Event()):
                if token is not None:
                    record['ids'].append(token)
        except BaseException as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            record.update(last=dict(engine.last), resume=engine.reused, log_end=log.stat().st_size)
            with log.open('rb') as f:
                f.seek(record['log_start'])
                text = f.read(record['log_end'] - record['log_start']).decode('utf-8', 'replace')
            record['trace'] = [s for s in text.splitlines() if any(k in s for k in
                               ('STOP_', 'STATE_HASH', 'strata pipeline:', 'conversation cache:'))]
            # Parse only the optional length; preserve all stage/bound/undo fields verbatim.
            record['state_lengths'] = [int(x) for x in re.findall(r'(?:STOP_STATE|STATE_HASH)\b[^\n]*?\bL=(\d+)', text)]
            records.append(record)
            save(f'{len(records):03d}-{name}.json', record)
        out, last = record['ids'], record['last']
        problems = []
        if last.get('generated') != len(out) or not 0 < len(out) <= count:
            problems.append('native returned count / DONE / budget disagree')
        if last.get('finish') not in ('length', 'stop'):
            problems.append('unexpected native finish')
        if last.get('finish') == 'length' and len(out) != count:
            problems.append('length completion did not reach budget')
        hits = [i for i, t in enumerate(out) if eos is not None and t in eos]
        if hits and (hits != [len(out) - 1] or last.get('finish') != 'stop'):
            problems.append('output continued past first configured EOS')
        if eos is not None and (name.startswith('A-') or name == 'replay') and not hits:
            coverage.append(f'{name}: configured EOS not emitted; stop position remains unproved')
        expected = len(ids) + len(out) - 1
        if any(n != expected for n in record['state_lengths']):
            problems.append(f'committed L differs from expected {expected}')
        if route:
            restored = 'conversation cache: restored ' in text
            if route == 'park' and not restored:
                coverage.append(f'{name}: no parked restore observed')
            elif route == 'live' and restored:
                coverage.append(f'{name}: restored instead of direct-live')
            if not {'reused', 'prompt_read'} <= last.keys():
                coverage.append(f'{name}: missing reuse/read metadata')
            elif (record['resume'], last['reused'], last['prompt_read']) != (len(ids) - 1, len(ids) - 1, 1):
                # A missed park can be admission pressure, not this native stop bug.
                (coverage if route == 'park' and not restored else violations).append(
                    f'{name}: expected RESUME=reused={len(ids) - 1}, read=1; '
                    f"got {record['resume']}/{last['reused']}/{last['prompt_read']}")
        violations.extend(f'{name}: {p}' for p in problems)
        return out

    try:
        engine = StrataEngine(str(a.engine.resolve()), args, cwd=cfg.get('cwd'), log=str(log), env=env)
        save('engine-info.json', dict(engine.info))
        if replay:
            generate('replay', replay['prompt_ids'], replay['max_new'], replay['sampling'])
        else:
            for limit in LIMITS if a.scenario == 'sweep' else (a.max_new,):
                head = generate(f'A-{limit}', A, limit, sampling)
                if limit != a.max_new:
                    continue
                continuation = A + head
                if a.route in ('live', 'both'):
                    continuation += generate(f'live-{limit}', continuation, a.continue_new, sampling, 'live')
                if a.route in ('park', 'both'):
                    generate(f'B-{limit}', B, 1, greedy)
                    generate(f'park-{limit}', continuation, a.continue_new, sampling, 'park')
            if a.scenario == 'sweep' and a.sampling == 'sampled':
                generate('greedy-control', A, a.max_new, greedy)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                error = error or f'close: {type(exc).__name__}: {exc}'
        save('results.json', {'requests': records, 'violations': violations, 'coverage_gaps': coverage,
                              'error': error, 'note': 'Trace review is required for boundary/EOS/B/replay coverage.'})
    ok = error is None and (a.expect == 'observe' or
                           (bool(violations) if a.expect == 'fail' else not violations and not coverage))
    print(json.dumps({'expect': a.expect, 'expectation_met': ok, 'violations': violations,
                      'coverage_gaps': coverage, 'error': error, 'output': str(a.output)}, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
