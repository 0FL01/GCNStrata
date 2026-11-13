# Bounded real200K selector capture

Prepared controller-only. No C++ compile, MI50 access, GPU execution, API operation, published-branch edit, or runtime qualification by the preparer.

## Identity and scope

This package captures scores, steps and actual selected IDs from the exact c183 integration baseline. It does not use public5fc5 as a substitute. The only diagnostic production-source edit is src/core/verify.cpp; verify.hpp, Verifier layout, all kernel/public interfaces and QSDUMP01 are unchanged. The old real64K capture executable3321 is not reused.

The seven c183 source hashes, original binary c183, original selector object06e903, archived kernels archive135bfd and their source/build/archive receipts are pinned in build_capture.py. Root verified the archived object/archive location and sizes before preparation. package.json binds the new patch/source and every runtime helper. INPUTS.sha256 binds the delivered package files.

## Root-only sequence

1. Verify INPUTS.sha256 and inspect package.json. Run the Python CPU/source tests. These do not invoke Docker, compile C++, or execute GPU work.
2. Default dry validation: python3 build_capture.py package.json
3. Root may explicitly build: python3 build_capture.py package.json --execute
4. Review new build.json, source hashes, used selector/archive hashes, compiler log and successful source/artifact restoration. Candidate hash is intentionally not predicted. A selector/archive byte mismatch stops qualification; no archive substitution is allowed. Identical-byte recompilation is accepted and recorded.
5. Default runner validation: python3 run_paired_200k.py --package package.json
6. Root may explicitly run the one bounded pair: python3 run_paired_200k.py --package package.json --execute --binary-sha256 EXACT_NEW_BUILD_HASH
7. Inspect qsa-200k-capture-run/summary.json and private evidence. PASS_CAPTURE_ONLY is input-capture qualification, not WIDE51 parity or speed qualification.

Build output is /home/radneon/strata/logs/perf-campaign-20261009/qsa-200k-capture-build. Run output is sibling qsa-200k-capture-run. Both must be fresh; wrappers do not silently reuse, erase or retry an earlier attempt. The API remains off.

## Native request and parity

Exactly one fresh c183 control process/request followed by one fresh diagnostic process/request. Reuse the qualified prompt-200000.ids.json (SHA736636e97ede94cc37e32447eefde4217a38bd17c81a7349345be98060e766c4), 256 outputs, greedy temperature0/top_p1/top_k1/seed12345 and exact native args/flags/topology. FIT17=1, TOPK_REG=2; WIDE51, TOPK_OLD and VERIFY_PROFILE absent.

All256 output IDs must equal the qualified reference and each other. Preserve full native DONE/output IDs/stderr privately. Save every available native/suffix counter and paired delta. Any delta pauses for analysis, without an automatic rerun; native timings are diagnostic only. Build/run containers use no network. Native control and diagnostic use identical host UID/GID; GPU/model access must work for that identity, without loosening permissions. The only writable host bind for native capture is its fresh private directory; container-internal temporary files remain possible.

## Capture safety

The run-local lambda executes after existing cs_ and copy_ completion, before next_->run. The copy completion status is checked when dumping is enabled. Existing HIP shim API cudaStreamIsCapturing/hipStreamIsCapturing must report None on cs_, copy_ and sh_cs_ before any D2H. No new backend API, device buffer, graph node, GPU kernel, profiler stamp or production-buffer write.

Atomic process-local states are indexed by devices0/1: unused0 -> claimed1 -> complete2. A failed attempt remains claimed and is not retried. Eligibility is actual T2..6, max_t6, pos0>=200000 and final row<=200263, unsplit serial G1. Preserve capacity51202, cap2051 and source byte strides204808/8204/16. Capture one stage0/layer23 and one stage1/layer47 fixture. The actual T/pos are observed, never forced. A missing eligible window is missing evidence.

The writer holds an O_DIRECTORY/O_NOFOLLOW directory fd, requires owned0700 mode, creates via openat(O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600), and cleans only an incomplete matching inode through that held directory. The fresh directory belongs exclusively to this wrapper/run; no other same-UID actor may mutate it concurrently. On fstat failure after exclusive creation, the diagnostic fails and retains the unverified empty file instead of risking deletion of a replacement. Completed receipts are never removed or overwritten.

## Consumer and required next gates

analyze_qsa_dump.py and qsa_topk_fixture_replay.cpp remain byte-identical to the vetted originals. validate_200k.py adds the experiment identity checks before parsing: exact paired stages/geometry/steps, real wide-window bounds, bounded file sizes and strict rejection of old64K fixtures, including relabelled headers with old steps. CPU exact-ID sorting and weighted radix use the original parser/analysis. Raw score bits, IDs and model protocol are private; summary contains metadata/hashes/counts only.

No native off/on200K result, new diagnostic compile, real200K fixture, replay parity or WIDE51 runtime speed/parity is established by this package. Subsequent replay must retain capacity-strided scores, actual nq/steps, canaries, input-byte checks, and no active_blocks argument. First require captured=baseline=reference IDs in direct and graph execution; then separately qualify the new WIDE51 implementation. The consumer's own >=65536 generic check is not a 200K experiment gate: run validate_200k first.
