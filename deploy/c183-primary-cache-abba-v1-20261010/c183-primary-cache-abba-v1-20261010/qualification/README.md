# Primary W-negative-cache v2 qualification only

Prepared on controller. No MI50 execution, GPU/model access, build, or transfer performed by this package preparation.

One fresh process, one cold frozen 12,288-token prefix, one output; GEN then QUIT. Exact c183 environment/native arguments plus STRATA_HC_WNEG_PRIMARY=1 and STRATA_HC_WNEG_QUALIFY=1. The latter is runtime-only and must be unset for any later timing of this same binary.

Both stored raw baselines independently agree: output 248046; primary HC_F16_TRACE attempts=324/taken=99; peer attempts=252/taken=104. baseline-proof.json pins both archives and raw stdout/stderr. The negative key set is independently reconstructed from each archive: zero-based source ordinal, N/K, mask4, exactly2 hits for each of11 primary negative weights. Exactly one primary scope and no peer scope are admitted. 302 physical guards are inferred as324 candidates minus22 hits from the reviewed sole guard path, not an independent runtime measurement. No performance claim or speed threshold exists here.

The reused native/safety/device-comparison files are byte-identical to the sealed cost runner. API stays off; high190W, shared lock, cooldown/thermal checks, 300s load/prefill and60s shutdown watchdogs, full raw/stdout/stderr/termination/cleanup receipts. Single-use marker is keyed by candidate binary, O_EXCL, and is never removed on failure. No automatic retry.

Admission requires the actual candidate binary, actual successful host and inner build receipts, complete55-module device equality freshly recomputed from both binaries, actual two-TU linked TLS evidence, exact v2 source hashes and source review, plus an independent admission review bound to all actual hashes and this runner/input-pins hashes. Source seal alone cannot authorize execution. Default CLI is offline admission only; --execute and --leave-api-stopped are additionally required. Parent must authorize the later actual one-shot launch.

Run CPU-only tests: python3 -m unittest -v test_qualification test_admission test_failure_evidence

Scope restriction: reviewed generate flow only, one nonconcurrent public run, fixed resident W throughout the request. No arbitrary mutating callbacks, reset, delete, reload or rebind races are covered. Peer execution remains baseline; the cache never survives across requests. PASS is only frozen-prefix one-token output parity and exact real-hit-population qualification, not broad model correctness or performance validation.
