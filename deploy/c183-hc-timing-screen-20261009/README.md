# One-shot synthetic HC timing screen

This is a bounded microbenchmark screen using the same two already-built HC ELFs, after independently audited exhaustive raw GPU parity. It is not a whole-model gain, bound, qualification, or deployment claim. Root alone builds this host controller against HIP and runs GPU0. Preparation compiles only plain CPU tests; it does not compile kernels, alter the engine, run a model, or change power settings.

## Prerequisites

The manifest and wrapper pin baseline ELF SHA256 5abe1289b549a9fa4ad7da00398fd81179a6d3b414bd2b3c44b749d5e315f377, candidate ELF SHA256 3f0c5ecb285232b1e338bf0aa51ebe729eefaa547388c14ebbd488167105ecf9, the independent resource gate, and exhaustive-audit.json SHA256 a12bd1f1953115c1c9a9252f35f75bfadc6c6db5e3e20298a0debf389afc04a7. The latter establishes 131,108 actual raw GPU pairs / 262,216 launches, both zero exits, all guards, exact outputs and all stats, and cleanup. Prior tiny wrapper serialization failure remains a historical wrapper failure; the corrected, tested serializer is reused here.

## Five descriptors

Descriptor 0: W, n=3,276,800, scale4096, stats offset0, optimistic maximum callcount1728.
Descriptor 1: X, n=41,943,040, scale512, stats offset5, maximum callcount810.
Descriptor 2: X, n=41,932,800, scale512, stats offset5, maximum callcount54.
Descriptor 3: X, n=1,310,720, scale512, stats offset5, maximum callcount810.
Descriptor 4: X, n=1,310,400, scale512, stats offset5, maximum callcount54.

Every launch uses ceil(n/256) blocks of256 threads and the exact original ABI: baseline has runtime float scale as its fifth argument; candidate has four arguments. All device buffers are allocated once before preflight/timing; the maximum descriptor allocation is reused. The scrub buffer is separate. Host vectors and input generation occur outside timed intervals.

## Three synthetic patterns

Pattern0, accepted: periodic signed exact values +0.25,-0.25,+0.5,-0.5,+1,-1,+2,-2,+4,-4,+8,-8. Expected active flags1,2,3 are000.
Pattern1, sparse reject: the same accepted sequence, replacing each4096th element starting at0, cycling BF16 words0x0001,0x7f7f,0x7f81,0x8000. Expected aggregate flags are111.
Pattern2, dense reject: every element cycles those four reject words. Expected flags are111.

These are synthetic stress distributions, not distributions inferred from model tensors. Rejected cases do not imply that the production host would take the same later path or invoke the converter at every optimistic count.

## Full correctness barrier

All15 descriptor/pattern pairs get baseline and candidate conversion before any timing. Additionally, the scrub descriptor (n=16,777,216, scale512, stats offset0, accepted pattern) gets both arms. This is exactly32 preflight launches.

Each arm starts with zeroed16-word stats, guarded by a23-word record with canaries at every exterior word. Input and output have n+2 halfwords with exterior canaries. Preflight copies back and compares every output bit, every input word against its deterministic generator, all stats words and guards, and required flags. Full raw input/output/stats for both arms are saved before checking equality. Every timed sample resets stats and records all23 returned words outside its event interval. Each completed timing cell repeats full output/input/stats/guard validation. Failures stop immediately and retain existing raw files; post-cell mismatches also save full failing arrays.

## Exactly752 kernel launches

For each pattern, then each regime, then each descriptor, run three ABBA quartets. A means baseline, B means candidate. There are30 cells,12 measured launches per cell,360 measured launches total. No statistical retry, adaptive sample count, rerun to win, or extra unrecorded warmup is permitted.

Warm regime0: one untimed same-arm conversion precedes every measured sample. There are180 warm predecessors. Its completion is synchronized, stats are reset, then the timed sample begins.
Scrub-conditioned regime1: one untimed baseline conversion on the independent accepted scrub buffer precedes every measured sample. Its input/output footprint totals64MiB. There are180 scrub predecessors. Scrub completion is synchronized, measured stats reset, then the timed sample begins. This is explicitly scrub-conditioned, not guaranteed cold. Using baseline as the common predecessor may affect instruction-cache state; this is a controlled conditioning recipe, not an isolation claim.

Total:32 preflight +180 warm predecessors +180 scrub predecessors +360 measured =752. Completion and independent parser checks require these exact counts. Scrub buffers are checked in the two-arm preflight and again after all timing; the final check adds no launches.

## Timing boundaries and caveats

HIP events are created once. Per-sample prepared argument arrays, stats copies, predecessor execution, synchronization and all setup occur before the start-event submission. The timed section submits start event, one module launch, and stop event. Success return-code logging and JSON occur only after stop-event submission; all three return codes are checked and retained. Failed start submission aborts before the launch. Stop-event synchronization, duration extraction, stats readback and logging occur afterward.

The events measure device-stream elapsed time for one launch. Single-launch event brackets can include host submission gaps; they are not a pure instruction-latency instrument. All individual event durations, paired ABBA results and ranges are retained. Immediately after elapsed-time extraction, an event_result row preserves the exact float bits and return code before duration or stats validation; the CPU parser verifies bit agreement with successful sample rows. Scrub-final failures save full readback before throwing. No whole-model inference is justified from these numbers.

## Predeclared screening estimator

For each quartet, baseline mean=(Afirst+Alast)/2 and candidate mean=(Bfirst+Bsecond)/2. Record both means, their signed difference and baseline/candidate ratio. For each descriptor/pattern/regime cell, the estimator is the median of its three paired quartet differences. Also report each arm's median quartet mean, raw values and range.

For each of the six complete pattern/regime scenarios independently, multiply its five descriptor deltas by1728,810,54,810,54 and sum. Report both the signed weighted sum and the deliberately optimistic sum with negative descriptor deltas clipped tozero before weighting. Never combine the best descriptor cells from different scenarios.

The screening gate is the largest whole-scenario optimistic sum compared with1003.739604ms. Strictly greater means INVESTIGATION_ONLY; at or below means PARK_WITHOUT_MODEL_QUALIFICATION. Counts are optimistic maximum counts, not measured call frequencies for these patterns. This sum is a screening estimate, never a whole-model bound or gain. Stop after the one valid screen or any failure.

## Safety and invocation

Root builds only screen.cpp against the existing HIP runtime, records compiler command/version/exit and driver SHA256, outside this immutable package. Example in the approved build environment: hipcc -std=c++17 -O2 /path/to/package/screen.cpp -o /absolute/output/hc_screen. Do not compile any device source or alter either ELF.

python3 run_screen.py --driver /absolute/output/hc_screen --driver-sha ROOT_VERIFIED_SHA256 --output /absolute/fresh/output --leave-api-stopped --execute

Omit --execute for hash/config validation only. The pinned wrapper and unchanged helpers require GPU0 at physical PCI0000:05:00.0, gfx906, high mode,190W cap, cooldown<=55C, conservative thermal ceilings at critical-minus10C or lower, the existing exclusive lock, and all containers/API off. No model mounts/networking/API actions/settings changes. Native default SIGALRM provides a300second hard process watchdog; wrapper checks deadline too. OS/driver teardown is separately bounded cleanup, not a hardware-preemption guarantee. Cancellation and every failure preserve receipts and leave API stopped. There are no automatic retries.

## Evidence and CPU tests

Each of32 preflight files is little-endian: seven uint32 words [0x48535431,descriptor,pattern,arm,n,scale,offset], then n+2 uint16 input words, n+2 uint16 output words, then23 uint32 stats words. The independent CPU parser regenerates input bytes, checks every output bit between arms, guards and exact flags. It rejects malformed JSON, duplicate keys, invalid event values, wrong order/counts and bad return codes. The wrapper hashes every raw binary and stdout/stderr even on failure. Full preflight data totals roughly2.3GB before compression; preserve it. After the terminal run, root compresses the complete receipt directory for transport. Compression is outside timing and is not performed by this runner.

python3 -m unittest -v test_screen.py compiles/runs the real CPU argument/fixture/serializer/watchdog helper, checks actual binary fixtures with the independent parser, corrupts evidence to test rejection, verifies estimator signs/clipping/gate and exact schedule, and mocks nonzero/parser-failing runs to check raw evidence and cleanup preservation. These are controller tests, not HIP compilation or GPU timing results. For a terminal real run: python3 analyze.py /absolute/output rechecks and prints the full report without GPU use.
