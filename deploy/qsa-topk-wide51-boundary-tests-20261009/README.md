# Isolated WIDE51 boundary correctness package

This package was prepared on the controller only. It contains no GPU runner,
does not start any GPU work, does not build the engine, and changes no
published source. Synthetic rows are correctness fixtures, not real-capture or
performance evidence. No speed claim is made.

## Scope and provenance

- Candidate selector: `qsa_select.wide51.cu`, SHA-256
  `98f30cd9fb2fd80619a8caf70c6c1d74bb5c7c14d50ca312b36158cd9518ae08`.
- FIT17 baseline: `qsa_select.fit17.cu`, SHA-256
  `5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a`.
- Original vetted transition driver: SHA-256
  `65738c8661973c6c14c9a7910e29b28c3a75c4e49ac3fd4208be9e7903c17f35`.
- Unchanged weighted-radix parser: SHA-256
  `53ad4c4d185d761aa35656e463d34b07a9c7bc91deeafb0e36e2b8bc3ea3daaa`.

The parent reported the separate automated/manual resource and ISA gate PASS.
This package does not repeat or replace that review. Adapted driver HIP
compilation, component linkage and all GPU execution remain NOT_RUN here.
The parent will compile this isolated external driver against already built
selector objects. No full engine build is needed.

## Bounded future process plan

`manifest.json` is data for review, not an executable runner. It specifies 21
candidate processes per device, or 42 across two separately authorized devices:

1. Fifteen primary processes: capacities 52223, 52224 and 52225, each with
   nq 1, 4, 6, 8 and 9; FIT17=1 and WIDE51=1.
2. Six controls at capacity 52224, nq 4: WIDE51 unset, 0, 2 and 01 with FIT17=1;
   FIT17 unset and 0 with WIDE51=1.

The literal string `01` is deliberately invalid. It catches an accidental
integer parser accepting non-exact values. Additional invalid strings are
covered offline, without multiplying GPU processes. JSON null means remove
the environment variable, not set it to the string `null` or an empty value.

All processes require REG=2. TOPK_OLD, TOPK_TRACE and FIT17_TRACE must be absent.
Trace value 0 is not accepted by this driver: absence is the unambiguous state.
The old FIT17 trace prints a reg66 topology and is not evidence of WIDE51.

Each process instantiates one graph and fixes its nq, allocation capacity and
all addresses. Only score/step payloads and reset output buffers change outside
capture. Exactly one selector call, with no active-count argument, is captured.
Every capacity/nq/environment combination uses a fresh OS process because the
selector caches host configuration. Never change flags or capacities inside an
existing process to reuse its graph.

There are 15 binary files, 426 distinct fixture cases and 2496 packed rows.
The six controls reuse one binary fixture, giving 606 graph replays per device
plus one untimed warmup in each of its 21 processes. A full baseline-process
matrix is not included in that count; each candidate case already checks the
separately invoked reference kernel and CPU oracle.

## Boundary coverage and valid inputs

- Short boundary: 24575, 24576, 24577, including mixed windows.
- FIT17: all four tails immediately below and above the 17408-active-block
  bound, n_kv 69628 through 69635; mixed windows at 69632.
- Wide contexts: 200703/200704 and 204799/204800, with mixed windows around
  200704 and 204800. These cover the 49/50 and 50/51 per-thread transitions.
- Allocation edges: n_kv `4*capacity-4` and `4*capacity-1`, both in contract,
  including zero-weight spare and partial-tail rows. No over-capacity row is
  generated. Every row satisfies `n_bid=floor(n_kv/4)` and
  `n_bid+1 <= max_blocks`.
- Growth and restoration: short to FIT17 to wide to FIT17 to short, followed
  by more mixed growth and shrinkage using the same graph.
- Nonmonotonic windows: earlier or penultimate long/capacity-edge rows with a
  last row in FIT17; and an earlier capacity-edge row with a short final row.
- nq=9 exercises the unchanged capacity-based reference fallback.
- Ties, signed zeros, positive/negative NaNs, infinities, negative values,
  subnormal bit patterns, and deterministic arbitrary float bit patterns.

At capacity 52225 the candidate must choose the guarded reg66 wide launch even
when current active rows fit 51. Capacity is the immutable host reach in this
uncounted path. At capacities 52223 and 52224, exact FIT17=1 and WIDE51=1 enable
the reg51 wide launch. A short final row selects guarded reference for the whole
window. Otherwise all rows participate in the FIT17 guard; an earlier long row
must defeat the middle arm. Disabled FIT17 keeps the old guarded reg66 topology.

## Evidence limits

Driver output `source_expected_route` is only a model of the pinned candidate's
dispatch. It is NOT a device counter, a measured route, or proof of the linked
object. `route_observed=0` is explicit in every route-bearing line. If the driver
is additionally linked to the FIT17 baseline, those labels remain the
candidate's expected routes and must not be attributed to the baseline.

Route evidence must combine verified selector object/source identity, the exact
process environment, immutable fixture capacity, absence of a count argument,
and the reviewed device guards. Old trace text is never route evidence. The
manifest's launch counts are source-derived expectations as well.

The adapted driver preserves the vetted allocation/cleanup, warmup, graph
capture/replay, CPU/reference comparisons, per-row output suffix checks, end
canaries and input-preservation logic byte-for-byte. Its only functional changes
are allowed fixture capacities, an explicit row-capacity check, accepting unset
or invalid option values for controls, disabling misleading trace requirements,
and accurate source-model labels for the candidate.

## Offline verification and reproduction

From this directory, controller-only commands are:

```sh
python3 -B test_wide51_offline.py
python3 -B generate_wide51_transitions.py
python3 -B test_wide51_offline.py --fixtures
sha256sum -c SHA256SUMS
```

The generator verifies all pinned input hashes and checks every unique generated
row with the unchanged weighted-radix parser. Full offline tests independently
reread all binary files, validate their packed dimensions, steps, hashes and IDs,
and rerun weighted-radix validation for every unique row. Shared rows are deduped
only by SHA-256 over their complete step/score/ID payload. Every fixture row is
still parsed and checked for its own capacity contract.

`offline-test-results.json`, `generation.log` and `offline-test.log` record the
actual controller results. `SHA256SUMS` pins the final package files. The separate
real 200K capture remains outside this synthetic package and is required for
later real-input qualification or any authorized performance work.
