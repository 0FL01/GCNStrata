#!/usr/bin/env python3
"""Prepare an isolated-output, host-only c183 capture build. Default: validate only.

Only the supervising/root task may deliberately invoke --execute. No GPU, API,
network, published branch, qualified binary, or baseline snapshot is modified.
Execution uses the existing build tree but snapshots/restores its seven source
files and executable, and fails closed on changed selector-object or kernel-archive bytes.
The two archived kernel inputs MUST exist at the pinned baseline paths; an
identical mutable archive is not a substitute for the archived provenance.
"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import time

R = Path('/home/radneon/strata')
W = Path('/home/radneon/strata-rnd-hc-f16')
BASE = R / 'logs/perf-campaign-20261009/qsa-topk-fit17-model'
O = R / 'logs/perf-campaign-20261009/qsa-200k-capture-build'
PATCH_NAME = 'qsa-200k-capture.patch'
IMAGE = 'sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
BASE_SHA = 'c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049'
BASE_BUILD_SHA = 'ab206a42b4984015f21a9e72b2aac402292b35fb995b8395631ef5256f9f9104'
SELECT_SHA = '06e9036483a1c743f274f6548ae0399341e4c2570d3a0ae69ce2e92790f14deb'
ARCHIVE_SHA = '135bfd9e51c94d9596db56d69db956f0397ff088097b28950002b98c3b97af9b'
ARCHIVE_RECEIPT_SHA = 'a5cb9b5a9de446c5c3363eee36baac1c3fe477c90986ebbe2f7140a9966af5f4'
SELECT_SIZE, ARCHIVE_SIZE = 404880, 9869884
VERIFY = 'src/core/verify.cpp'
SELECT = 'src/kernels/cuda/qsa_select.cu'
SIX = {
    'include/strata/core/verify.hpp': '5a959164cbaed0feee14463ea036c0002816993a21b2f987d6dc57566fceef13',
    'include/strata/kernels/iq_kernels.hpp': '00e41a0b6d841cf6c532e012b113f5f0e4e9aa6b9e0d59a077b0e0bce2f1897a',
    VERIFY: 'e0562925ea41f430193df5c326c7ddec743a59b1e43bf5d2545040475053c292',
    'src/kernels/cuda/iq_kernels.cu': '3aefd6d98a44fc5d5954be4122debf310d2515afd016773f021ebc0b33deebbb',
    'src/program/generate.cpp': 'e7a5a754af2155c779aad5f2232ba1c0052ebbd2a4795d40885eff571b62221c',
    'src/kernels/cuda/qsa_decode_attn.cu': '278f6080871552da95f18ea5e5af6a3e2a4208bf4d330368a6f054d895e91faf',
}
SOURCES = dict(SIX, **{SELECT: '8d836acd211b7e493cd50bd680d128489f3f25f94ea26725afbab2402ef899bc'})
OBJECT_REL = 'build-906/CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_select.cu.o'
ARCHIVE_REL = 'build-906/libstrata_kernels.a'
BINARY_REL = 'build-906/strata'
ARTIFACTS = {BINARY_REL: BASE_SHA, OBJECT_REL: SELECT_SHA, ARCHIVE_REL: ARCHIVE_SHA}
BASE_OBJECT = BASE / 'qsa_select.cu.o'
BASE_ARCHIVE = BASE / 'libstrata_kernels.a'
NAME = 'strata-qsa-200k-capture-build'
LOCKS = ('/tmp/strata-qsa-fit17-qualification.lock', '/tmp/strata-direct-ff34-ce.lock')
PURPOSE = 'qsa-200k-selector-capture'


def need(ok, label):
    if not ok:
        raise RuntimeError(label)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def pinned_fields():
    return dict(base_binary_sha256=BASE_SHA, base_build_json_sha256=BASE_BUILD_SHA,
                base_source_sha256=SOURCES, select_object_sha256=SELECT_SHA,
                kernel_archive_sha256=ARCHIVE_SHA,
                base_archive_snapshot_json_sha256=ARCHIVE_RECEIPT_SHA, image=IMAGE,
                patch_file=PATCH_NAME, output_dir=str(O))


def validate_spec(s):
    """Pure validation: no filesystem, subprocess, Docker, build, or GPU access."""
    need(isinstance(s, dict), 'package must be an object')
    for key in ('patch_sha256', 'patched_verify_sha256'):
        need(digest(s.get(key)), 'required digest: ' + key)
    need(s['patched_verify_sha256'] != SOURCES[VERIFY], 'capture source must differ from baseline')
    need(type(s.get('schema_version', 1)) is int and s.get('schema_version', 1) == 1, 'package schema version')
    for key, value in pinned_fields().items():
        need(key not in s or s[key] == value, 'pinned package field: ' + key)
    need('patch_path' not in s or s['patch_path'] == PATCH_NAME, 'package patch_path must be adjacent fixed filename')
    return True


def validate_receipt(d, s, binary_sha256=None, package_sha256=None):
    """Pure fail-closed build-receipt validation, importable by the paired runner."""
    validate_spec(s)
    need(isinstance(d, dict), 'build receipt must be an object')
    expected = dict(pinned_fields(), schema_version=1, purpose=PURPOSE,
                    patch_sha256=s['patch_sha256'], patched_verify_sha256=s['patched_verify_sha256'],
                    gpu_devices=[], network='none', exit_code=0, api_touched=False, promoted=False,
                    compiler_confirmed_stopped=True, sources_restored=True, sources_invalidated=True,
                    build_artifacts_restored=True, select_object_unchanged=True,
                    kernel_archive_unchanged=True,
                    select_object_path=str(O / 'qsa_select.cu.o'),
                    kernel_archive_path=str(O / 'libstrata_kernels.a'),
                    binary_path=str(O / 'strata'))
    for key, value in expected.items():
        need(key in d and d[key] == value and type(d[key]) is type(value), 'invalid build receipt: ' + key)
    need(type(d.get('kernel_rebuild_detected')) is bool, 'kernel rebuild status missing')
    need(isinstance(d.get('patch_path'), str) and Path(d['patch_path']).is_absolute()
         and Path(d['patch_path']).name == PATCH_NAME, 'resolved bundle patch path missing')
    need(d.get('compiled_sources') == dict(SOURCES, **{VERIFY: s['patched_verify_sha256']}),
         'compiled source set must differ only in verify.cpp')
    need(digest(d.get('binary_sha256')) and d['binary_sha256'] != BASE_SHA, 'capture binary identity')
    need(digest(d.get('package_sha256')), 'package digest missing')
    need(not d.get('error_type') and not d.get('restore_failed_files'), 'build/restore error')
    if binary_sha256 is not None:
        need(d['binary_sha256'] == binary_sha256, 'capture binary digest mismatch')
    if package_sha256 is not None:
        need(d['package_sha256'] == package_sha256, 'package digest mismatch')
    return True


def validate_patch_text(text):
    """Reject additional file sections, binary/rename/mode edits and path tricks."""
    old = [line for line in text.splitlines() if line.startswith('--- ')]
    new = [line for line in text.splitlines() if line.startswith('+++ ')]
    need(old == ['--- a/' + VERIFY] and new == ['+++ b/' + VERIFY], 'patch only verify.cpp')
    diffs = [line for line in text.splitlines() if line.startswith('diff --git ')]
    need(diffs in ([], ['diff --git a/' + VERIFY + ' b/' + VERIFY]), 'patch file sections')
    forbidden = ('GIT binary patch', 'Binary files ', 'rename from ', 'rename to ',
                 'copy from ', 'copy to ', 'new file mode ', 'deleted file mode ', 'old mode ', 'new mode ')
    need(not any(line.startswith(forbidden) for line in text.splitlines()), 'patch must only edit text')


def kernel_action(line):
    """Record compiler/archive invocations; byte identity is the acceptance gate."""
    return bool(re.search(r'Building\b.*(?:strata_kernels\.dir|qsa_select\.cu)', line, re.I)
                or re.search(r'Linking\b.*libstrata_kernels\.a', line, re.I)
                or re.search(r'(?:hipcc|clang\+\+|nvcc)(?:\s|$).*(?:qsa_select\.cu|strata_kernels\.dir)', line)
                or re.search(r'(?:^|\s)(?:\S*/)?(?:ar|llvm-ar|ranlib)\s.*libstrata_kernels\.a', line))


def fingerprint(path):
    st = path.stat()
    return dict(sha256=sha(path), size=st.st_size, mtime_ns=st.st_mtime_ns,
                ctime_ns=st.st_ctime_ns, inode=st.st_ino)


def call(argv, **kwargs):
    return subprocess.check_output(argv, timeout=30, **kwargs)


def verify_baseline():
    need(sha(BASE / 'strata') == BASE_SHA, 'archived c183 binary mismatch')
    need(sha(BASE / 'build.json') == BASE_BUILD_SHA, 'archived c183 build.json mismatch')
    need(json.loads((BASE / 'build.json').read_text()).get('compiled_sources') == SOURCES,
         'archived compiled_sources mismatch')
    for f, expected in SOURCES.items():
        need(sha(BASE / 'compiled-source' / f) == expected, 'archived source mismatch: ' + f)
    need(sha(BASE / 'archive-snapshot.json') == ARCHIVE_RECEIPT_SHA, 'archived kernel receipt mismatch')
    need(BASE_OBJECT.stat().st_size == SELECT_SIZE and BASE_ARCHIVE.stat().st_size == ARCHIVE_SIZE,
         'archived artifact size mismatch')
    for path, expected in ((BASE_OBJECT, SELECT_SHA), (BASE_ARCHIVE, ARCHIVE_SHA)):
        need(path.is_file() and not path.is_symlink(), 'required regular archived snapshot absent: ' + str(path))
        need(sha(path) == expected, 'archived artifact mismatch: ' + str(path))
    for f, expected in ARTIFACTS.items():
        need(sha(W / f) == expected, 'live c183 build artifact mismatch: ' + f)
        need(not (W / f).is_symlink(), 'live artifact must be a regular file: ' + f)


def build_command():
    return ['docker', 'run', '--rm', '--network', 'none', '--name', NAME,
            '--entrypoint', '/bin/sh', '-v', str(W) + ':/src',
            '-v', str(R / 'third_party/llama.cpp') + ':/src/third_party/llama.cpp:ro',
            '-w', '/src', IMAGE, '-c', 'cmake --build build-906 --target strata -j4 --verbose']


def monitor_build(p, log, d, timeout=1800):
    deadline, pending = time.monotonic() + timeout, b''
    with selectors.DefaultSelector() as sel:
        sel.register(p.stdout, selectors.EVENT_READ)
        while sel.get_map():
            need(time.monotonic() < deadline, 'build timeout')
            for key, _ in sel.select(timeout=0.2):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                log.write(chunk); log.flush()
                pending += chunk
                lines = pending.split(b'\n'); pending = lines.pop()
                for raw in lines + [pending]:
                    line = raw.decode('utf-8', errors='replace')
                    if kernel_action(line):
                        d['kernel_rebuild_detected'] = True
                        seen = d.setdefault('kernel_rebuild_log_lines', [])
                        if line[:4000] not in seen and len(seen) < 100:
                            seen.append(line[:4000])
    d['exit_code'] = p.wait(timeout=max(1, deadline - time.monotonic()))
    need(d['exit_code'] == 0, 'capture build failed')


def stop_compiler(p):
    """Never restore source bytes unless the owned Docker compiler is proven stopped."""
    ok = True
    try:
        active = bool(call(['docker', 'ps', '-q', '--filter', 'name=^/' + NAME + '$']).strip())
    except (OSError, subprocess.SubprocessError):
        return False
    if active or (p is not None and p.poll() is None):
        try:
            subprocess.run(['docker', 'stop', '--time', '30', NAME], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        except (OSError, subprocess.SubprocessError):
            try:
                subprocess.run(['docker', 'kill', NAME], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except (OSError, subprocess.SubprocessError):
                ok = False
        if p is not None:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(); ok = False
    try:
        ok = not call(['docker', 'ps', '-q', '--filter', 'name=^/' + NAME + '$']).strip() and ok
    except (OSError, subprocess.SubprocessError):
        ok = False
    return ok


def execute(s, package_sha256, patch_path):
    validate_spec(s)
    need(digest(package_sha256), 'actual package digest required')
    need(not call(['docker', 'ps', '-q']).strip(), 'another container active; no API/build may be running')
    verify_baseline()
    patch = Path(patch_path).resolve()
    need(patch.name == PATCH_NAME and patch.is_file(), 'fixed adjacent bundle patch required')
    need(sha(patch) == s['patch_sha256'], 'patch hash mismatch')
    validate_patch_text(patch.read_text())
    need(not O.exists(), 'fresh output directory required: ' + str(O))
    O.mkdir(parents=True)
    files = list(SOURCES) + list(ARTIFACTS)
    before = {}
    # Snapshot EVERY source/artifact before even the first build-tree mutation.
    for f in files:
        need((W / f).is_file() and not (W / f).is_symlink(), 'source/artifact must be regular: ' + f)
        dest = O / 'before' / f; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(W / f, dest); before[f] = fingerprint(W / f)
        need(sha(dest) == before[f]['sha256'], 'snapshot verification: ' + f)
    d = dict(pinned_fields(), schema_version=1, purpose=PURPOSE, started_utc=stamp(),
             patch_sha256=s['patch_sha256'], patched_verify_sha256=s['patched_verify_sha256'],
             package_sha256=package_sha256, patch_path=str(patch), before=before, gpu_devices=[], network='none',
             api_touched=False, promoted=False, exit_code=None, sources_restored=False,
             build_artifacts_restored=False, kernel_rebuild_detected=False,
             select_object_unchanged=False, kernel_archive_unchanged=False)
    p, error = None, None
    def interrupted(signum, frame):
        raise InterruptedError('build stop requested')
    old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        staged = O / 'compiled-source'
        for f in SOURCES:
            dest = staged / f; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BASE / 'compiled-source' / f, dest)
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=staged, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=staged, check=True)
        d['compiled_sources'] = {f: sha(staged / f) for f in SOURCES}
        need(d['compiled_sources'] == dict(SOURCES, **{VERIFY: s['patched_verify_sha256']}),
             'only verify.cpp may differ from exact seven-source baseline')
        # Leave exact sources and their mtimes alone. Changed baseline files retain
        # their archived mtimes; do not touch headers or any GPU source. Force the
        # two host translation units only when their actual bytes changed.
        for f in SOURCES:
            if sha(W / f) != d['compiled_sources'][f]:
                shutil.copy2(staged / f, W / f)
                if f in (VERIFY, 'src/program/generate.cpp'):
                    (W / f).touch()
        need({f: sha(W / f) for f in SOURCES} == d['compiled_sources'], 'overlay source verification')
        need(fingerprint(W / OBJECT_REL) == before[OBJECT_REL], 'selector object changed before build')
        need(fingerprint(W / ARCHIVE_REL) == before[ARCHIVE_REL], 'archive changed before build')
        d['argv'] = build_command()
        with (O / 'build.log').open('xb') as log:
            p = subprocess.Popen(d['argv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            monitor_build(p, log, d)
        need({f: sha(W / f) for f in SOURCES} == d['compiled_sources'], 'sources changed during build')
        after_object, after_archive = fingerprint(W / OBJECT_REL), fingerprint(W / ARCHIVE_REL)
        d['kernel_artifact_metadata_changed'] = (after_object != before[OBJECT_REL] or after_archive != before[ARCHIVE_REL])
        d['select_object_unchanged'] = after_object['sha256'] == SELECT_SHA
        d['kernel_archive_unchanged'] = after_archive['sha256'] == ARCHIVE_SHA
        d['kernel_artifacts_used'] = dict(selector_object=after_object, kernel_archive=after_archive)
        need(d['select_object_unchanged'] and d['kernel_archive_unchanged'], 'USED kernel artifact bytes changed; candidate rejected, no substitution allowed')
        for f, name in ((BINARY_REL, 'strata'), (OBJECT_REL, 'qsa_select.cu.o'), (ARCHIVE_REL, 'libstrata_kernels.a')):
            shutil.copy2(W / f, O / name)
        d.update(binary_sha256=sha(O / 'strata'), binary_path=str(O / 'strata'),
                 select_object_path=str(O / 'qsa_select.cu.o'), kernel_archive_path=str(O / 'libstrata_kernels.a'))
        need(d['binary_sha256'] != BASE_SHA, 'capture binary must differ from baseline')
        need(sha(O / 'qsa_select.cu.o') == SELECT_SHA and sha(O / 'libstrata_kernels.a') == ARCHIVE_SHA,
             'output artifacts mismatch')
    except BaseException as exc:
        error = exc; d['error_type'] = type(exc).__name__; d['error'] = str(exc)
    finally:
        for sig in old:
            signal.signal(sig, signal.SIG_IGN)
        d['compiler_confirmed_stopped'] = stop_compiler(p)
        failures = []
        if not d['compiler_confirmed_stopped']:
            failures = files[:]
        else:
            for f in files:
                try:
                    # Avoid replacing untouched originals, including protected objects.
                    try:
                        now = fingerprint(W / f)
                    except FileNotFoundError:
                        now = None
                    if now != before[f]:
                        shutil.copy2(O / 'before' / f, W / f)
                    need(sha(W / f) == before[f]['sha256'], 'restored source/artifact digest')
                    if f in SOURCES:
                        # The transient host objects may contain capture instrumentation.
                        # Invalidate every restored source so future builds cannot reuse it.
                        (W / f).touch()
                    else:
                        need((W / f).stat().st_mtime_ns == before[f]['mtime_ns'], 'restored artifact mtime')
                except BaseException:
                    failures.append(f)
        d.update(sources_restored=not any(f in SOURCES for f in failures),
                 sources_invalidated=not any(f in SOURCES for f in failures),
                 build_artifacts_restored=not any(f in ARTIFACTS for f in failures),
                 restore_failed_files=failures, finished_utc=stamp())
        # Baseline read-only identity check also guards accidental destination mistakes.
        try:
            need(sha(BASE / 'strata') == BASE_SHA and sha(BASE / 'build.json') == BASE_BUILD_SHA,
                 'baseline identities changed')
            need(sha(BASE / 'archive-snapshot.json') == ARCHIVE_RECEIPT_SHA, 'archived receipt changed')
            need(all(sha(BASE / 'compiled-source' / f) == h for f, h in SOURCES.items()),
                 'archived sources changed')
            need(sha(BASE_OBJECT) == SELECT_SHA and sha(BASE_ARCHIVE) == ARCHIVE_SHA,
                 'archived kernel identities changed')
        except BaseException as exc:
            if error is None:
                error = exc; d['error_type'] = type(exc).__name__; d['error'] = str(exc)
        write(O / 'build.json', d)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    need(d['compiler_confirmed_stopped'], 'compiler not proven stopped; snapshots retained, restoration withheld')
    need(d['sources_restored'] and d['build_artifacts_restored'], 'restoration failed; see build.json')
    if error is not None:
        raise error
    validate_receipt(d, s, package_sha256=package_sha256)
    print(json.dumps(dict(executed=True, build_receipt=str(O / 'build.json'), binary_sha256=d['binary_sha256'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--execute', action='store_true', help='supervising root task only; never default')
    args = parser.parse_args()
    s = json.loads(args.package.read_text()); validate_spec(s)
    if not args.execute:
        print(json.dumps(dict(spec_valid=True, executed=False, remote_state_checked=False)))
        return
    with contextlib.ExitStack() as stack:
        for path in LOCKS:
            lock = stack.enter_context(open(path, 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(s, sha(args.package), args.package.resolve().parent / PATCH_NAME)


if __name__ == '__main__':
    main()
