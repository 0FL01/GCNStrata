# Offline FIT17 resource-gated runtime screen

Prepared for root execution only. This package has not run on MI50, built a GPU binary, or made a timing/model claim. It adapts the preserved `run_middle_test.py`; `qsa_probe_safety.py` is byte-identical to the verified helper SHA256 `749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45`.

## Flow

1. Validate explicit root resource receipt, reviewed source/patch, harness/transition source, preprocessor proof, both root-supplied binary hashes, and all fixture hashes.
2. Require API and every other container already stopped. Hold `/tmp/strata-direct-ff34-ce.lock` exclusively. No API identity/health checks, API stop/start, or settings writes.
3. Validate both physical GPUs, high performance mode, unchanged 190 W caps, cooldown and thermal margins through the original helper. Fixed `HIP_VISIBLE_DEVICES=0,1`; ordinal mapping is the already-qualified mapping from the prior campaign.
4. Run transition parity for both GPUs, T1/4/6/8/9, baseline0 and each resource-approved candidate. With both modes approved this is 30 fresh processes. One driver graph per T traverses every case. Any parity, provenance, safety, cancellation, or process failure aborts and cleans up.
5. Run separate fresh-process GPU0 ABBA groups for mode0-vs1 and mode0-vs2, but only approved modes. A result below30% mean-of-two-median latency reduction or without strict max(B)<min(A) separation is saved as `negative_candidate`; it does not abort or prevent the next approved candidate being tested.
6. Only GPU0-passing candidates proceed to their own fresh-process GPU1 ABBA groups. GPU1 results report separation and reduction independently. No automatic winner, model run, or promotion.
7. Stop only named probe containers, verify no container remains, verify unchanged high/190 W settings, and leave API intentionally stopped. Thermal telemetry and stdout/stderr persist per process. Create `OUTPUT/cancel` to request cancellation, or signal the wrapper.

With both candidates passing the screen: 30 transition + 8 GPU0 + 8 GPU1 probe processes. If both are negative: 30 + 8. Baselines are rerun per comparison, never reused across candidate ABBA groups. All runs use the frozen qualified ROCm image, `--network none`, read-only build/fixture mounts, and the original launch/cooldown/safety implementation.

## Input layout

Build directory (absolute, under `/home/radneon/strata`):

- `qsa_select.fit17.cu`: SHA `21efb8501aa41ce938daa497c5b3196e275e39a4fadd9138da2cb4e8cf2aded1`
- `qsa-topk-fit17-probe.patch`: SHA `a119128d75a0bbadc3af8b40ee1f60a9bdfe190a0e14903bfbcac3877212a81e`
- `qsa_topk_fixture_replay.cpp`: vetted current harness SHA `0eac4ceb11f783e14977f35afdbe0bf6c39fbddaa0fe8c39675e4a0f346a667e`
- `qsa_reg17_transition.cpp`: SHA `65738c8661973c6c14c9a7910e29b28c3a75c4e49ac3fd4208be9e7903c17f35`
- `qsa_topk_fit17_probe` and `qsa_reg17_transition`: exact root-supplied binary SHA values, verified before every process
- `preprocessor-proof.json`: `platform_shim`, `wrong_intrinsics_header_absent`, and `half_wave_shift_verified` must all be true

Transition fixture directory contains the regenerated five reg17 binaries and `reg17-transition-manifest.json` SHA `5dee4fb41ba853d3bc94e6d4401c4d8c980b87d1df5efedc8059360ab258a231`. Generate and verify using the separately supplied reg17 script bundle first.

Both real fixtures remain exactly 295268 bytes:
- GPU0: `06e572c257f2187e67787f6c813b3d43c0a68584455adf1cff6bed5757b56a1b`
- GPU1: `a2ccc7241c927f4d7995705dc5a0ad95e6d2f03a8638f093b833156ffe8e2fac`

## Root-authored resource receipt

Do not fabricate a PASS or include modes that failed the independent ISA/resource gate. Required JSON fields:

    {
      "passed": true,
      "source_sha256": "21efb8501aa41ce938daa497c5b3196e275e39a4fadd9138da2cb4e8cf2aded1",
      "patch_sha256": "a119128d75a0bbadc3af8b40ee1f60a9bdfe190a0e14903bfbcac3877212a81e",
      "qsa_topk_fit17_probe_sha256": "ROOT_VERIFIED_REAL_BINARY_SHA",
      "qsa_reg17_transition_sha256": "ROOT_VERIFIED_TRANSITION_BINARY_SHA",
      "allowed_modes": [1, 2]
    }

`allowed_modes` must be a unique nonempty subset of [1,2]. Omit a candidate failing the resource gate. Mode0 baseline is always included. Mode1 is scalar17; mode2 is fixed-two-group histogram17. Both candidate modes use max device n_bid+1<=17408; mode0 is baseline guarded66. For T<=8 the wrapper requires exactly one matching FIT17 mode/histogram/arms trace; T9 instead requires the unchanged reference256 fallback trace and no candidate trace.

## Invocation (root only)

First syntax/parser/control-flow tests are CPU-only:

    python3 test_fit17_wrapper.py

Then fill the absolute paths and binary SHAs from actual build receipts:

    python3 run_fit17_test.py \
      --build "$BUILD" --real0 "$REAL0" --real1 "$REAL1" \
      --transition-fixtures "$FIXTURES" --gate-receipt "$RESOURCE_RECEIPT" \
      --expected-source-sha 21efb8501aa41ce938daa497c5b3196e275e39a4fadd9138da2cb4e8cf2aded1 \
      --expected-patch-sha a119128d75a0bbadc3af8b40ee1f60a9bdfe190a0e14903bfbcac3877212a81e \
      --expected-binary-sha "$REAL_BINARY_SHA" --expected-transition-sha "$TRANSITION_BINARY_SHA" \
      --output "$NEW_OUTPUT" --leave-api-stopped --execute

Without `--execute`, only argument syntax is checked. It is not a resource or fixture validation PASS. `python -O` is forbidden. Output must not exist. A clean completed screen may have no GPU0-passing candidates; inspect `negative_candidate_modes` and the raw measurements rather than interpreting exit0 as performance PASS. Errors exit nonzero; summary preserves failure/cleanup state.

## CPU test provenance

Bundled `test_logs/` are actual old reg33 logs copied from controller `qsa-topk-middle-gpu-receipts-20261009/real-abba` and `transition-runs-v2`. Tests first parse them unchanged with the preserved old wrapper. New FIT17 trace lines and boundary-label conversions are explicitly synthetic parser examples, not new GPU results. Four unittest cases cover actual real logs, actual T4/T9 transition logs, malformed/duplicate/incorrect trace rejection, negative ABBA behavior, and mocked end-to-end ordering in which candidate1 fails30%, candidate2 passes, and only candidate2 reaches GPU1. No docker or GPU operations occur in these tests.
