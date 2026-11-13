# Bounded WIDE51 correctness runner

Prepared and tested on the controller only. No GPU process was started. This
runner contains no performance runs, timings, API health/start/stop operations,
new build steps, source publication, or settings changes.

## Exact authorized matrix

The fixture manifest is pinned to SHA-256
`82e1d2be8927458a1f8040afca379f858e060933cef6955a07881108fd151773`.
Its 21 processes run first on physical GPU ordinal 0 and then on ordinal 1,
sequentially: exactly 42 processes and 1212 graph case replays. No baseline
matrix or additional combinations are added. Each process checks graph output,
separate reference output, CPU IDs, output padding, end canaries and input bytes.

The 15 primary processes cover capacities 52223/52224/52225 and nq 1/4/6/8/9.
Six bounded controls at capacity 52224/nq4 cover WIDE51 unset/0/2/01 and FIT17
unset/0. Every process is fresh, keeping its own graph addresses and capacity
constant. Capacity 52225 must retain guarded reg66 for wide rows even if the
current rows fit 51. No row violates its allocation capacity.

REG is exactly 2. OLD, TOPK_TRACE and FIT17_TRACE are removed from the child
environment. Both feature variables are removed first, then set only if the
manifest gives a string; JSON null means unset. Exact string 01 stays 01.

## Mandatory provenance

Four explicit expected SHA-256 arguments are required:

- Newly compiled transition binary: root supplies `--binary-sha` after checking
  its isolated linkage to the already reviewed candidate selector object.
- Manual review: `eaa6954d12312b3c8048056b3b999df7ce78d335dbaa4f036eca92f07939853a`.
- Automated resource gate: `e25b59127276328bcfdc768fd1051db758f11507aacb4e651001c3cbdad1c13f`.
- Fixture manifest: `82e1d2be8927458a1f8040afca379f858e060933cef6955a07881108fd151773`.

The latter three are also pinned inside the runner. It validates manual PASS,
its candidate/baseline source identities, the manual-to-resource hash binding,
scalar pre-barrier exits, 13 unchanged kernels and the reviewed resource gate.
It checks all fixture hashes/sizes before touching GPU tools, then rechecks
binary, receipt, manifest and current fixture hashes before every process.

The explicit binary hash binds exactly the binary root approves. It does not
independently prove which selector object was linked; root must verify the
component build receipt/map before supplying that approved hash. The driver
source SHA should remain
`97efe48e2c76432143d99aeecdd584957bd36ca615749233bbefb5732a2882f2`.

## Preserved safety and launch behavior

The following helpers are copied byte-for-byte from the vetted public FIT17
runner and verified on import:

- `private_parser_support.py`: `95579778a7162c1c13ac730eca3c04d47e5dbd61d7304d1b1f0afef99b07d374`.
- `qsa_probe_safety.py`: `749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45`.

Only their launch, safety, file/hash and cleanup helpers are used. Their old
experiments, API-related functions, route parsers and timing code are never
called. The runner preserves:

- One nonblocking lock for the entire run: `/tmp/strata-direct-ff34-ce.lock`.
- Existing two-device PCI check and established HIP ordinal mapping.
- Existing high mode, 190W cap, thermal margins and <=55C cooldown checks.
- The frozen ROCm image, network-none and read-only binary/fixture mounts.
- One named container at a time, 180-second per-process limit and existing
  telemetry polling/cleanup behavior.
- Cancellation through OUTPUT/cancel, SIGINT or SIGTERM.
- All containers/API must already be stopped; they must remain stopped afterward.
- A fresh 0700 output directory and final unchanged-power/settings verification.

No performance measurement is taken. Telemetry elapsed seconds and UTC timestamps
are safety/audit information only.

## Output parser and audit files

Every stdout line must match the prescribed ordered CASE_PASS records followed
by exactly one completion line. Missing, duplicate, extra, reordered or malformed
records fail. Each line must affirm CPU/reference/graph equality, padding,
canaries and unchanged inputs. Completion must report the exact nq, capacity,
case count, flags, single graph and constant addresses/capacities.

`source_expected_route` is a source-model expectation, not a measured device arm.
Every record must say `route_observed=0`; summaries retain `route_observed=false`.
Legacy FIT17 topology trace is rejected, never used as WIDE51 evidence. Any
performance line is rejected as unexpected stdout.

Each process retains TAG.stdout, TAG.stderr and TAG.driver.json, including exit
code, command, telemetry and cleanup evidence. summary.json is updated before
and after each process and on failures. Failure stops the sequence and performs
cleanup; it never silently continues or retries GPU processes. Exit zero requires
all 42 processes and final cleanup to pass.

## Parent/root invocation

Run the CPU tests first. Paths must be absolute; the binary must reside under
the existing Strata root. FIXTURES is the extracted boundary-package root that
contains manifest.json and its fixtures subdirectory.

```sh
python3 -B test_wide51_runner.py
python3 -B run_wide51.py \
  --binary "$REVIEWED_TRANSITION_BINARY" --binary-sha "$ACTUAL_BINARY_SHA" \
  --manual-review "$MANUAL_REVIEW_JSON" \
  --manual-sha eaa6954d12312b3c8048056b3b999df7ce78d335dbaa4f036eca92f07939853a \
  --resource-gate "$RESOURCE_GATE_JSON" \
  --resource-sha e25b59127276328bcfdc768fd1051db758f11507aacb4e651001c3cbdad1c13f \
  --fixtures "$FIXTURES" \
  --manifest-sha 82e1d2be8927458a1f8040afca379f858e060933cef6955a07881108fd151773 \
  --output "$FRESH_OUTPUT" --leave-api-stopped --execute
```

Without --execute, only argument syntax is checked; no device or container tools
are used and no lock is acquired. Never run with Python optimization (-O).

## Offline tests

Nine small unittest cases cover all 21 parser configurations; corruption,
duplicates, wrong route-observation claims and old trace rejection; matrix bounds
and capacity-52225 fallback; exact unset/string command construction; explicit
receipt/binary hashes; mocked sequential 42-process completion; early failure,
exit-code preservation and cleanup; preflight failure before GPU access; and
the no-execute/shared-lock control flow. All external operations are mocked in
these tests. test_inputs contains small copies of the reviewed JSON receipts and
manifest, not the binary fixtures. These tests are not GPU qualification receipts.
