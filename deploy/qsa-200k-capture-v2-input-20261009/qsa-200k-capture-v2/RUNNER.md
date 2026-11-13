# Paired 200K capture runner

Preparation only. No build, container, API, MI50/GPU or real native run was performed while creating/testing these files.

## Default: offline validation

From the complete finalized package directory:

    python3 run_paired_200k.py --package package.json

This checks pinned dependency bytes and the original qualified fixture/flags/native arguments. It does not inspect Docker, GPUs, host processes or remote paths.

## Deliberate root-task execution

Only after the isolated diagnostic build has passed and its SHA256 has been independently read:

    python3 run_paired_200k.py --package package.json --execute --binary-sha256 NEW_DIAGNOSTIC_SHA256

No defaults or guessed diagnostic hash are accepted. This runner never invokes a build or API. It refuses a preexisting `qsa-200k-capture-run` output directory. Both shared qualification/direct locks are held for the whole pair.

One fresh c183 control process is followed by one fresh diagnostic process, each receiving exactly one 200000-token fixture request for 256 outputs, temperature 0, top_p 1, top_k 1, seed 12345. A failed control stops before the diagnostic. There is no retry, reordering or flag exploration. FIT17=1, TREG=2; WIDE51, OLD and profiling flags are absent. The only additional diagnostic flag is `STRATA_QSA_SELECT_DUMP=/captures/window`.

Both arms use the runner's effective UID/GID and existing numeric supplementary groups, recorded in each manifest; the original `video` entry is retained. The original repository and model bind mounts remain read-only. Only the diagnostic gets a writable host bind, its fresh private capture directory. Container-local ephemeral storage is unchanged from the qualified helper.

Before any request, the runner verifies the build's successful stopped/restored receipt, package hash, exact baseline and diagnostic binaries, unchanged selector object/archive bytes, compiled source snapshots, reference and fixture hashes. The two captures are passed to the strict 200K validator with request/fixture/baseline provenance.

## Evidence and stop rules

The run root is 0700; all files are 0600. Raw stderr, DONE, complete output IDs and original helper results are stored beneath `private/`. All 256 IDs per arm and the complete qualified engine topology are equality-gated against the fixed reference. Captures remain private.

`summary.json` contains bounded numeric/status evidence, every native and suffix counter, and numeric diagnostic-minus-control deltas. Any counter or suffix total divergence produces `PAUSE_COUNTER_DIVERGENCE` and a nonzero exit. The runner does not repeat a run to obtain cleaner counters. Timing is diagnostic only; no performance claim or throughput comparison is made.

Ctrl-C/SIGTERM triggers cleanup of the two uniquely named owned containers. No unrelated process/container is stopped. The API is never started or restored.

## CPU verification

    python3 -m unittest test_paired_runner -v

The tests mock native/container creation while exercising the actual copied protocol loop. They check the exact 1+1 bound, one request per process, flags and mounts, full IDs, private raw forensics even with malformed DONE, strict old-64K rejection, preexisting-capture/control-quality aborts, counter pause propagation, dependency identity, and offline default behavior.
