# Pinned public FIT17 qualification runner

Root-run only; no GPU work was done while preparing this package. Actual reviewed resource/ISA approval and six exact binary hashes are mandatory. The pinned public input commit is `ca3533157d066a13b128f3daa61f56411dcf3155`; input manifest SHA is `00a102313d8823d3415e102eb713172704d3a571bad143cda45dca45f405a0a5`. The builds must use public headers/source only and no support archive.

## Bounded flow

- Eight public parity selftests: both physical GPUs, original public baseline with FIT17 unset, candidate with FIT17 unset, candidate0, candidate1. Each selftest covers 18 public cases, including counted fallback. The public CLI has no device argument; it receives one physical GPU through HIP_VISIBLE_DEVICES and uses local ordinal0.
- Thirty captured transition runs: both GPUs, T1/4/6/8/9, original baseline0/candidate0/candidate1. Each driver has one graph with constant addresses/capacities across all cases, exact CPU/reference IDs, padding/end canaries, and unchanged inputs.
- Eight real-capture correctness runs: both GPUs and all four baseline/default-off/explicit modes. Each checks exact captured/reference/graph IDs and input/canary equality. Its mandatory minimal event sampling is discarded, not treated as a timing result.
- Only after all correctness and actual resource/ISA gates: fresh candidate-binary ABBA mode0/1 on GPU0 AND GPU1, four processes each, batch64/samples7. Dedicated public FIT17 trace is OFF in timing. No candidate trace is inferred from device metadata or private TOPK_TRACE; the latter remains1 solely because the unchanged capture driver requires it and public selector ignores it.
- All 54 processes are sequential under the shared engineering lock, frozen image, network-none, read-only mounts, original safety helper's high/190W/thermal/cooldown requirements, and named-container cleanup. API must already be off and stays off. No API health/start/stop operations and no settings changes. Prior verified HIP ordinal mapping is assumed; physical PCI set is rechecked by the helper.

This is reproduction of the selected scalar17 public port. The historical private discovery screen failed30%; that remains a historical negative and is not redefined. The public runner records the diagnostic30% outcome but NEVER skips GPU1 because of it. Both public ABBA pairs are always reported. Positive separation on both devices is a descriptive replication check. There is no new private-vs-public numeric threshold, no model claim, and no automatic promotion. Runtime errors, parity errors, provenance failures, and safety failures still abort with cleanup.

## Inputs and resource receipt

`--build` should be `/home/radneon/strata/logs/perf-campaign-20261009/qsa-topk-fit17-public-build`. It must contain build.json, build.exit_code=0, INPUTS.sha256, sources, preprocessor/resource/ISA artifacts and six binaries: public-parity.base/fit17, capture.base/fit17, transition.base/fit17.

Root/reviewer supplies an ACTUAL gate JSON, then root passes its SHA as --gate-sha. Required fields:

    {
      "passed": true,
      "resource_isa_passed": true,
      "allowed_modes": [0, 1],
      "public_commit": "ca3533157d066a13b128f3daa61f56411dcf3155",
      "support_archive_used": false,
      "build_json_sha256": "ACTUAL_SHA",
      "binaries": {
        "public-parity.base": "ACTUAL_SHA", "public-parity.fit17": "ACTUAL_SHA",
        "capture.base": "ACTUAL_SHA", "capture.fit17": "ACTUAL_SHA",
        "transition.base": "ACTUAL_SHA", "transition.fit17": "ACTUAL_SHA"
      },
      "resource_files": {
        "base.resources.log": "ACTUAL_SHA", "fit17.resources.log": "ACTUAL_SHA",
        "preprocessor-proof.json": "ACTUAL_SHA",
        "ACTUAL_BASE_ISA_FILENAME.s": "ACTUAL_SHA",
        "ACTUAL_FIT17_ISA_FILENAME.s": "ACTUAL_SHA"
      }
    }

Actual ISA filenames must begin qsa_select.base / qsa_select.fit17. Every listed binary/resource hash must agree with both build.json and the files on the GPU host. Do not fabricate approvals or populate expected hashes from unchecked files.

Use the original five reg17 transition fixtures and manifest SHA `5dee4fb41ba853d3bc94e6d4401c4d8c980b87d1df5efedc8059360ab258a231`. Both real fixtures remain 295268 bytes with the exact hashes embedded in the preserved support module. All binary/fixture hashes are checked again immediately before their runs.

## Root invocation

    python3 test_public_runner.py
    python3 run_public_fit17.py \
      --build "$BUILD" --gate "$ACTUAL_RESOURCE_GATE" --gate-sha "$GATE_SHA" \
      --fixtures "$REG17_FIXTURES" --real0 "$REAL_GPU0_FIXTURE" --real1 "$REAL_GPU1_FIXTURE" \
      --output "$NEW_OUTPUT_DIRECTORY" --leave-api-stopped --execute

All paths must be absolute. Without --execute only argument syntax is checked. Never use python -O. Output directory must not exist. The existing helper's cancellation marker `OUTPUT/cancel`, SIGINT, and SIGTERM request cleanup. Inspect summary.json, individual stdout/stderr, and telemetry. Exit0 means the prescribed screen completed safely; it is not a claim of a30% gain or any model gain.

## CPU validation

Four tests cover an actual prior capture stdout, synthetic public topology examples, all transition parser modes/Ts, and a fully mocked54-process run proving both-GPU ABBA continues when GPU0 is only10% faster and every timing run disables FIT17_TRACE. These are parser/control-flow tests, not GPU receipts. `private_parser_support.py` is the preserved prior offline wrapper used only for vetted launch, constants, safety import, and ABBA arithmetic. Its private route parser and execute function are never called. `qsa_probe_safety.py` remains exact SHA749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45.
