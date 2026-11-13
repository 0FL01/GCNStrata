# One bounded current-c183 profile refresh, v2

Root-only execution. Preparation is controller/CPU only. No model build, source branch change, API action, power change, publication, automatic retry, or performance gain claim.

## Scope

Exactly two fresh native processes: control then profile; one identical65,536-token request and1,024 outputs each. Existing B1 flags/native arguments remain byte-for-byte values. Profile adds only PREFILL_TIMING, DECODE_TIMING, SPLIT_TIMING and VERIFY_PROFILE=1. All are presence-only; =0 would still enable them. Greedy seed12345, spec6/mtp4/suffix3,19078 slots/9900 primary,27/21 layers, int8 KV/resident32768, no prompt reuse, adaptation0 and PLE RAM lock.

Actual gfx906 host build defines STRATA_HIP_GFX906 but not STRATA_USE_HIP. Thus the unoverridden main shared fork defaults ON. VERIFY_PROFILE forces it OFF. The diagnostic is an altered-scheduling stream-interval envelope, not untouched-build exclusive kernel attribution. Native time changes combine instrumentation, forced serialization, adaptive scheduling and single-pair run variability. No optimization gain or pure instrumentation-overhead estimate follows. Do not add overlapping PP timelines, GPU stage intervals, or nested CPU/verify timings into a model budget.

The previous shared-stream-off experiment is closed/negative. This package does not reopen it as a candidate. Optional native counters remain null when the DONE16 protocol omits them; stderr diagnostic windows are separately named. All IDs/counters are saved. IDs and INFO topology must match; adaptive counters are compared descriptively. Fresh control IDs must match the existing historical c183 B1 receipt. No matching result is obtained by retrying a failed pair.

## Input gates

- Frozen binary: /home/radneon/strata/logs/perf-campaign-20261009/recovered-c183-input/strata, SHA c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049.
- Packaged prompt-65536.ids.json contains65,536 IDs,340708 bytes and exact original file SHA63ce3d4042027a00b2193feab143f1d9443b5c31a863a962ec9543ccb54d9695. Unchanged bytes were recovered from controller /home/opencode/ai/strata-donor-review/receipts/quant-comparison-64k-20261006/prompt-65536.ids.json. No regeneration was performed.
- Pinned image bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646.
- Original native_support.py and qsa_probe_safety.py remain byte-identical. The isolated native_arm.py copy removes the requirement for an enabled FIT17 host trace because actual B1 flags omit it. V2 additionally streams full private stdout before parsing, retains partial parsed IDs and termination/error status on every exit, records reader/capture completeness, and preserves evidence even if native cleanup fails. The exact reviewed function diff is native_arm.patch. All native flags, arguments, sampling, request count and safety settings remain unchanged.
- Outer cleanup attempts both owned container names independently, retains each error, then checks empty-container and power state. A first cleanup error cannot skip the second name.
- Query source remains byte-identical to v1: ccfbaac7f3542cc3391aa4f6a9db34c6edd6a28a4531bc547bf68b6baf09d2fe. V1 is preserved separately and is superseded for model execution.

## Timestamp query and provenance

Root compiles query_wallclock.cpp inside the pinned image using /opt/rocm/bin/hipcc and runs it once with HIP_VISIBLE_DEVICES=0,1 under the same shared lock, fixed high190W settings and exclusive host. No model or kernels are used. It prints runtime return codes, both physical PCI IDs, architecture and hipDeviceAttributeWallClockRate in kilohertz. Both must be25000 for the producer's40ns/tick conversion.

Known producer provenance:

- verify_kernels.cu SHA1f7a650452510af9892b9636534c8086fd80578ddee1733022e926f3285ae15c; source1207 uses wall_clock64()*40.
- W/build-906/CMakeFiles/strata_kernels.dir/src/kernels/cuda/verify_kernels.cu.o SHA50a3e527914aa51e6c85a6b6aa358625b4398bd1a9fb9ab06742c735194486c5,237736 bytes.
- W/build-906/libstrata_kernels.a SHA135bfd9e51c94d9596db56d69db956f0397ff088097b28950002b98c3b97af9b matches c183 archive receipt. Extracting verify_kernels.cu.o from that archive yields the same object SHA.
- W=/home/radneon/strata-rnd-hc-f16. Root still reviews actual gpu_stamp_kernel ISA conversion before setting producer_isa_verified=true. Do not manufacture a passed receipt from source text alone.

Root supplies a timestamp receipt JSON containing: binary_sha256, kernel_archive_sha256, timestamp_source_sha256, timestamp_object_sha256 (the exact hashes above); archive_member_matches_object=true; producer_isa_verified=true; producer_ns_per_tick=40; query_source_sha256 (actual packaged helper hash); and clock_query (the complete parsed query JSON). Pin this receipt by CLI SHA. Include any ISA evidence paths/hashes as extra fields so the full receipt remains reviewable. A failed/unverified clock check blocks the native pair.

## Root invocation

Run CPU tests first: python3 -m unittest test_profile test_failure_evidence. Then transfer this directory to a fresh R/logs package directory and verify SHA256SUMS. Keep source-only inputs, scripts, and receipts private. Tests cover pre-DONE ERR, EOF, malformed token line, timeout, cancellation and native cleanup failure, as well as attempting both outer cleanup names after an initial failure.

python3 run_profile.py --binary /home/radneon/strata/logs/perf-campaign-20261009/recovered-c183-input/strata --fixture ABSOLUTE_EXACT_FIXTURE --timestamp-receipt ABSOLUTE_ACTUAL_TIMESTAMP_RECEIPT --timestamp-receipt-sha ACTUAL_RECEIPT_SHA --output NEW_DIRECTORY_UNDER_R --leave-api-stopped --execute

Omit --execute for syntax validation only. Do not use python -O. Original shared lock /tmp/strata-direct-ff34-ce.lock is mandatory. All containers must already be stopped. Root separately reserves host against unrelated native GPU work. Image STRATA defaults are checked against approved baseline keys; unexpected inherited timing/route flags abort. Binary/fixture hashes are rechecked per arm. Existing safety enforces both GPUs, high190W, critical-minus10°C and80/90°C limits, cooldown≤55°C, RAM≥64GiB,30-minute process timeout, cancellation marker and signals. Only named owned containers are cleaned up. No health request or API restart occurs.

Outputs include summary, each native result/manifest/telemetry, private full stderr and streaming stdout, exact raw DONE when received, all parsed IDs plus partial-ID and termination/error receipts on every exit, diagnostic raw phase/stage reports, A→B time changes and control time/101 budgets for a 1% throughput target. Raw stdout is flushed before each line enters the parser; it is fsynced once the reader ends. A failed or incomplete capture aborts. Stage profile allocation can fail silently inside the engine; two stage reports with matching positive window counts are mandatory. Failure is preserved and terminates the experiment.
