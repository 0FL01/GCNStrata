# Frozen radix-stop source and CPU review

Date: 2026-10-09 UTC. Controller: nc-lab. Verdict: no source correctness defect found within the existing valid-input contract. This is not GPU or performance qualification.

## Frozen inputs

- Public baseline: `5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a`
- Public candidate: `587c02f9a76916605be595bbf8cdf4ce72ac04edbade4c80686772e72903064c`
- Public patch: `694531ccfcfe3da6017540af121bb774bdf0d0896b0112a6e4dcd67a0491d38b`
- Private candidate: `0ba4909d7c3661e16f8003d70ce9a4045bc5f05dcce1ae5897d1a0fd2d742946`

Both candidate files reproduce exactly from the pinned baselines and existing source transform. Their bytes, patches, and prior CPU receipts were unchanged during this review. No GPU command, MI50 access, candidate edit, commit, push, or publication occurred.

## Exactness argument

The contract is R=4, nonnegative n_kv, n_bid=floor(n_kv/4), one final tail of weight n_kv mod 4, valid score/output storage, and dispatch capacity covering every row. Full blocks weigh four cells; at most one tail can have positive weight less than four. Arbitrary malformed steps are outside this contract and are not newly validated by the candidate.

1. After the third byte pass, prefix P fixes the high 24 bits. All positive keys in the selected bucket are in [P,P|255]. The saved `above` counts cells strictly above that interval.
2. A positive bucket mass no greater than four implies exactly one positive-weight block: two full blocks weigh eight; a full plus a positive tail weighs at least five; two partial tails do not exist. A zero-weight tail is explicitly ignored.
3. Counting keys above P|255 as `gt` and keys in the interval as `eq` therefore reproduces the exact full-key threshold partition. The one positive-weight owner writes its full uint32 key into shared s_digit. Existing eq-budget and prefix-scan logic allocate the same number of cells.
4. Output uses the recovered full key, not P or P|255. Contiguous per-thread block ownership plus the unchanged exclusive scans and emission loops preserve ascending IDs and lowest-index tie resolution.
5. If the selected bucket contains multiple positive blocks, the fourth radix pass remains. Tied full blocks and full-plus-partial collisions conservatively fall back. The unchanged all-selected branch emits identity IDs. Zero-width output was separately exercised; its threshold is immaterial.

Source locations in qsa_select.radix-stop.cu: order_key 23-28; scan 576-600; guarded register kernel 602-749; stop decision 680-686; interval count and unique writer 702-721; key read after eq scan 723-727; opt-in host branch 1437-1450.

## Synchronization and tails

All pre-barrier exits are CTA-uniform. The digit/stop flag is written by thread zero and published with __syncthreads. A second barrier finishes all digit reads before a next-pass write or shared full-key reuse. The break condition is CTA-uniform. In the recovery branch only one positive-weight block writes the recovered key. The first barrier inside block_excl_scan occurs after that write; all three scan barriers finish before the key is read. No writer remains after recovery. This establishes the source happens-before argument, but CPU execution does not test HIP lowering, wave behavior, or hardware races.

Zero-weight tails with maximal, minimal, or threshold-equal keys cannot contribute to the histogram, recovery writer, counts, or output. NaNs map to key zero; signed zeros share the same key; infinities and subnormals follow the unchanged order_key mapping. The new branch reasons only about uint32 keys and cell weights.

The largest valid register geometry has at most 67,584 blocks and at most 270,335 active cells including its final tail, so the tested histogram and scan counts remain comfortably within signed int range.

## Verified evidence

- Existing source receipt: 100,014 raw-float key cases, 3,136 adversarial source-snippet cases, 16 captured rows from two 64K and two 200K files. All IDs matched their captured baseline. Thirteen of the sixteen rows take the modeled three-pass path; three retain four passes. This is fixture replay on CPU, not a new GPU run.
- New independent harness: 361,968 exact-output cases: 260,816 exhaustive small key/tail/width combinations, 100,000 randomized cases, 1,152 ownership-boundary cases up through 67,584 blocks. All meaningful non-bypass positive-width thresholds also equal the four-pass baseline.
- The new harness checks 1,000,014 raw float bit patterns. It includes 98,032 recoveries, 242,561 fallbacks, 21,375 all-selected bypasses, and 21,068 zero-width inputs; zero-width cases overlap the other categories.
- The entire new harness was repeated under AddressSanitizer and UndefinedBehaviorSanitizer: exit zero, same case counts and no sanitizer diagnostic.
- Existing extracted host-dispatch receipt: 282 cases across gfx906/HIP-other/CUDA; 152 CPU device-guard predicates; six rejected invalid template combinations.
- Additional host parsing and boundary receipt: 108 passing cases. Any presence of STRATA_TOPK_OLD, including empty or 0, preserves the old route. Existing REG uses atoi(value)==2, while FIT17 and RADIX_STOP require exactly the string 1. Static flags must be set before first use; tests use fresh processes.

The earlier verbal report of 387,832 cases had no recoverable script or receipt in the audit directory or relevant shared outputs. It is not counted as verified evidence. The durable independent run above replaces that unverified count.

## Dispatch and remaining gates

The new launch modes are <17,true,3> and <66,true,4>. Guard mapping preserves mode 1/2 predicates, including the actual maximum n_bid across up to eight device rows. The new branch requires gfx906, the extended decode route, FIT17=1, RADIX_STOP=1, 0<nq<=8, no positive active-count route, no old-route override, and capacity above 33,792 but no greater than 67,584 blocks. Short windows still use the existing reference arm. The flag does not change non-gfx906 routes. Setting RADIX_STOP alone does not enable the candidate.

Original template modes and launch targets remain in source. This review does not certify unchanged old-symbol ISA: source-level s_digit signedness and eliminated constexpr code still need the separate compiler/resource gate. The resource package must use these exact frozen hashes.

Still required before any GPU qualification or speed claim: exact-candidate device compilation and resource/ISA checks; hardware exact-ID replay and sanitizer/race checks where supported; graph growth/restore and guard ownership checks; paired performance measurements under the existing acceptance criteria. Histogram aggregation, thread scheduling, and block scans are sequentially modeled in the CPU harness. This report makes no throughput, latency, occupancy, or production-readiness claim.

## Artifacts and rerun

- `source-review-summary.json`: compact final receipt, sanitizer status, and evidence hashes
- `independent-source-review.json`: independent CPU coverage and exact inputs
- `review_radix_stop_cpu.py`, `independent-cpu.cpp`: source-verbatim snippet validation and independent oracle harness
- `independent-cpu-test.log`, `independent-cpu-build.log`: ordinary build/run evidence
- `independent-cpu-sanitized-*`: sanitized executable, logs and exit status
- `review_dispatch_edges.py`, `independent-dispatch-edges.json`: extra host parsing tests
- `cpu-source-proof.json`, `cpu-dispatch-proof.json`: prior evidence retained byte-for-byte
- `test_source_cpu.py`, `test_dispatch.py`: prior test sources; captured-fixture rerun requires the original private capture dependencies on nc-lab
- `REVIEW.sha256`: all review-directory file hashes except the manifest itself

Reproduce the independent CPU run with `python3 review_radix_stop_cpu.py`; reproduce supplemental dispatch checks with `python3 review_dispatch_edges.py`. Both require the frozen inputs and existing extracted-host binaries in this directory. They write review outputs only. Sanitized repeat: compile independent-cpu.cpp using g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer, then run the result. CPU builds intentionally leave CUDA unroll pragmas unused; warnings are recorded. The compact harness also records one harmless misleading-indentation warning; its unconditional checks were executed and verified under sanitizers.
