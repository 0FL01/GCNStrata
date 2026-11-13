# Bounded WIDE51 real-200K selector replay

Prepared and CPU-tested on the controller only. This package has not run on MI50 or accessed a GPU. No real 200K capture hashes are present or invented. The root task alone may transfer and deliberately execute it after a qualified capture pair exists.

## Exact experiment

- Two directly linked binaries use the identical, unchanged, header-driven `qsa_topk_fixture_replay.cpp` consumer. Baseline `replay.fit17` links the reviewed public FIT17 object; candidate `replay.wide51` links the reviewed WIDE51 object. No support archive, private selector replacement or trace-derived route claim.
- Both arms set REG=2, FIT17=1 and TOPK_TRACE=1. TOPK_TRACE is required by the unchanged consumer but ignored by the public selector. WIDE51 is absent for baseline and exactly `1` for candidate. OLD, FIT17_TRACE, VERIFY_PROFILE, capture dumping and inherited device masks are cleared. Each process explicitly sees physical ordinals `0,1` and selects its own captured device.
- First, four sequential correctness processes: baseline/candidate on GPU0 stage0/layer23, then baseline/candidate on GPU1 stage1/layer47. Batch=1, samples=1. Their unavoidable consumer event timings remain raw logs only and are excluded from latency results.
- Only after all four complete, one ABBA per GPU, with the respective actual stage capture. Baseline, candidate, candidate, baseline; fresh process per arm; batch=64, samples=7; four warmup graph launches in the unchanged consumer. Exactly eight latency processes. GPU1 is still measured if GPU0 is slower. No search, retry, extra sweep, crossover capture or extra repetition.
- Every process checks direct production=reference=captured IDs, output padding and end canaries, graph=captured IDs before and after timing, and unchanged score/step input bytes. CPU exact IDs are separately verified before any GPU action. This proves public replay equals the private c183 capture rather than assuming public/private equivalence.

## Prerequisites and fail-closed gates

The packaged, hash-pinned receipts require the manual/resource PASS and the completed 42-process/1212-case boundary gate. Replay build receipt and both binaries require independently supplied hashes; both selector object hashes and consumer source are pinned to the reviewed bytes.

The capture summary must report PASS_CAPTURE_ONLY, full 256-ID reference equality, equal native/suffix counters, and successful cleanup. Both supplied capture files must match its recorded filename/SHA256/byte count. The original unchanged validate_200k.py and analyze_qsa_dump.py then validate both actual files, CPU-sort/radix-check exact IDs, enforce paired metadata/steps, pos>=200000, actual observed T in 2..6, max_t=6, capacity51202, and exact stage/layer identities. No fixed T=4 assumption, 64K substitute, renamed old file, header-only relabeling or synthetic measurement is accepted.

The exact captured T is whatever this one native window contains. The two captures do not claim to cover every T=2..6 or all model layers.

Safety uses the original byte-identical qsa_probe_safety.py and private_parser_support.py: pinned container image, fixed high/190W, existing 10C thermal margins, all temperatures <=55C before every process, live thermal polling, per-process 180s timeout, owned-container cleanup, and the single /tmp/strata-direct-ff34-ce.lock. API and all containers must already be stopped. Known native inference/replay/compiler processes also block execution. Artifacts and exclusivity are rechecked immediately after cooldown. This runner never starts/stops the API, changes power settings or stops unrelated containers. Root must continue to ensure no other GPU job bypasses the common lock.

The output must be new. Failure aborts immediately and cannot resume/retry this schedule. Final receipts are hash-manifested. Output directories/files are private 0700/0600 under execute's 077 umask. Raw logs contain selector measurements and metadata, not captured score arrays or token IDs.

## Root execution command

Copy this package intact to an absolute MI50 path chosen by root and verify SHA256SUMS. The replay build has already passed and its 20 exported receipt entries were independently verified on the controller:

- build.receipt.json: ca60d4183c202bb08aef585cae6779ef5ab6d5cf6471aa0c0a28ebe91ce5c97a
- replay.fit17: ee4505a1ff161f468284d00281919e2a4c75dcc0ad2ef5bc5d90b1c8caa26efc
- replay.wide51: 1ba24426536c09fa17163b993115939456a54a6527d1fad5de0a9c54a89f18e3

Set RUN to the copied package, BUILD to /home/radneon/strata/logs/perf-campaign-20261009/qsa-wide51-replay-build, CAPTURE_SUMMARY to the qualified capture run's actual summary.json, REAL0/REAL1 to its actual stage files, and OUTPUT to a fresh replay output directory. Independently read and assign CAPTURE_SUMMARY_SHA, REAL0_SHA, REAL1_SHA from the finalized files only after capture PASS. They are deliberately unavailable in this package.

    python3 "$RUN/run_real200k_wide51.py" \
      --build "$BUILD" \
      --build-sha ca60d4183c202bb08aef585cae6779ef5ab6d5cf6471aa0c0a28ebe91ce5c97a \
      --baseline-sha ee4505a1ff161f468284d00281919e2a4c75dcc0ad2ef5bc5d90b1c8caa26efc \
      --candidate-sha 1ba24426536c09fa17163b993115939456a54a6527d1fad5de0a9c54a89f18e3 \
      --resource-build "$RUN/prerequisites/resource-build.receipt.json" \
      --manual-review "$RUN/prerequisites/manual-review.json" \
      --resource-gate "$RUN/prerequisites/resource-gate.json" \
      --boundary-summary "$RUN/prerequisites/boundary-summary.json" \
      --capture-summary "$CAPTURE_SUMMARY" --capture-summary-sha "$CAPTURE_SUMMARY_SHA" \
      --real0 "$REAL0" --real0-sha "$REAL0_SHA" \
      --real1 "$REAL1" --real1-sha "$REAL1_SHA" \
      --output "$OUTPUT" --leave-api-stopped --execute

Omitting --execute only validates CLI paths/hash syntax and performs no input, Docker or GPU inspection. Actual provenance/CPU correctness gates always run before execute reaches Docker/GPU checks. Python -O is prohibited.

## Output interpretation

Per-GPU output reports all 28 raw latency samples (14 baseline and 14 candidate), four run medians, the mean of baseline run medians, mean of candidate run medians, percent reduction, and whether both candidate medians are below both baseline medians. There is no invented speed threshold, statistical significance claim, model throughput claim, PP/TG extrapolation, publishing or promotion. A negative result remains a completed measurement, provided correctness and cleanup pass.

All route labels are explicitly source-expected with route_observed=false. This is warm-cache isolated selector latency on one captured last-QSA layer of each stage, including the guarded no-op launches. It excludes scorer, model, PP and TG timing.

## CPU verification

    python3 -m unittest test_real200k_runner test_validate_200k -v

20 tests passed. Tests use synthetic CPU-only fixtures/logs and mocked external operations. Coverage includes dynamic T=2..6/header metadata, exact output parser, wrong/missing/duplicate parity, NaN/negative/malformed timing, old64K and relabeled-header rejection, structurally-valid-but-wrong CPU IDs, capture/build provenance, own physical-device mapping, exact 4+8 schedule, all correctness before ABBA, failure/no-retry/cleanup, negative candidate continuation, no-execute default, immutable file/symlink gates and the common lock. No test evidence is presented as actual GPU or real native capture evidence.
