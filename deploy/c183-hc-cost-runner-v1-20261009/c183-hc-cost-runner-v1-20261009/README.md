# HC host-clock cost runner delta

Prepared from reviewed c183-hc-counts-runner-v1-20261009. No MI50, model, HIP build, or GPU execution during preparation. This package is not runtime authorization.

run_cost.py retains exact v4 provenance and baseline c183 identity, actual binary/build/inner/device/review hash admission, fresh full 55-module device-byte and metadata verification, shared nonblocking lock, API-off requirement, thermal/cooldown checks, and original native wrapper. Exact model arguments and environment are unchanged, including STRATA_HC_COUNTS_ONLY_OBSERVER=1. One cold 12288-token prompt requests one token, then QUIT. Load/prefill watchdog is 300 seconds and shutdown watchdog 60 seconds. Raw private stdout/stderr and failure termination receipts survive parser rejection.

The new marker is hc-cost-<actual binary SHA256>.once, separate from the untouched prior hc-counts marker; one use includes failures, with no automatic retry. Output uses cost.json and summary.cost. Strict parse_cost.py also requires parse_counts.py and verifies all 576 timing observations, original counts, expected 11/9 W-bad populations and stable visible lifetime.

Per stage, conditional 64K repeat screen = 14 * measured full-repeat sum + measured tail-repeat sum. First hits are separate. The deliberately generous stage sum below 1003.739604 ms parks without cache patch or model qualification; equality or above allows attribution/lifetime investigation only. Whole guard wall includes predecessor work, is not wholly removable, and is neither a model bound nor a gain estimate.

Run CPU mocks: python3 -m unittest -v test_runner test_failure_evidence test_cost_execution

Root must supply actual --binary, --baseline, --build-receipt, --inner-receipt, --device-gate, --review-receipt, --output and every corresponding SHA argument. Review bindings include run_cost.py and input-pins.json hashes. Default performs offline admission only. --execute --leave-api-stopped is permitted only after root reviews the actual build receipts, full device gate and binary. No marker is consumed by preparation or offline admission.
