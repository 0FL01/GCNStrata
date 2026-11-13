# Frozen exact radix-stop resource probe

This package adds only the default-off, exact-string-`1` option
`STRATA_GFX906_TOPK_RADIX_STOP=1` to the current qualified FIT17 selector.
The frozen public source and canonical patch are unchanged:

- Baseline SHA256: `5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a`
- Candidate SHA256: `7036ef4eb84a145bbeeab46d0a735548cd719a656c5dcf6ec2a3ab0e3e089964`
- Patch SHA256: `39a29aa4c63c31dd2b8f7bf8714512dd5b8508186d128f8462919c1492710a23`

New specializations are exactly `block_topk_reg_kernel<17,true,3>` and
`block_topk_reg_kernel<66,true,4>`. Modes 3/4 map to the existing guards 1/2.
They may stop after byte three only when boundary-bucket positive cell mass is
1..R. A single positive-weight block then recovers the exact full key during
the existing gt/eq interval scan. Existing barriers publish that write.
This package is not a GPU correctness or speed claim.

## Root-only execution handoff

Only the coordinating root agent is to transfer and launch this package on the
selected MI50 build host. Preparation workers must not connect to MI50 or run
the build. The word root here means the coordinating agent, not Linux UID 0.
Root must obtain any currently required transfer approval before copying this
bundle or its outputs to MI50; the package itself grants no transfer permission.

After transferring the tar and its sibling checksum, verify the archive, extract
it into a new input directory, verify `INPUTS.sha256`, and invoke manually:

    bash INPUT/build-host.sh INPUT /home/radneon/strata/logs/perf-campaign-20261009/qsa-radix-stop-resource-only

The output directory must not already exist. The wrapper is byte-identical to
the previous wide51 wrapper: same `/tmp/strata-direct-ff34-ce.lock` engineering
lock, pinned image, read-only input mount, isolated output mount, no GPU device
mounts, and Docker `--network none`.

Pinned image:

    sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646

The inner script verifies there are no `/dev/kfd` or `/dev/dri` device nodes,
checks clang23, and reuses the original compiler flags, public headers, include
order and platform HIP shim. It builds only the baseline and candidate selector
translation units and lightweight CPU snippet tests. It never builds an engine,
links/runs a selector executable, loads a GPU runtime, or runs a GPU image.

## Hard resource gate

- Exactly 13 baseline kernels, all with identical actual function machine-code
  bytes, complete metadata (including arguments), and compiler resource remarks
- Exactly two candidate additions, modes 17/3 and 66/4; no hidden extra functions
- Compare 17/3 to 17/1, and 66/4 to 66/2
- No increase in scratch, SGPR/VGPR allocation, or SGPR/VGPR spill counts
- No compiler occupancy regression
- LDS 32,908 bytes, wave64, 1024-thread bound, and no dynamic stack
- Descriptor/remark resource agreement, required data present, valid actual ELF

Any failure is retained as a failure. Do not relax thresholds, edit source,
silently accept parser failures, or continue to GPU work. A resource pass still
requires manual review of both new ISA bodies and their guard, stop, single
writer, barrier and key-recovery control flow. Actual compile and manual ISA
review have not been performed when this preparation package is frozen.

## Pure CPU preparation checks

    python3 test-source-cpu.py --output /tmp/radix-source-new
    python3 test-dispatch.py --output /tmp/radix-dispatch-new
    python3 test-resource-gate.py

The first two commands require new output directories and a standard CPU C++
compiler; use `--cxx PATH` to choose one. The gate suite is Python stdlib only
and uses generated synthetic ELF/metadata/remark/disassembly fixtures. No GPU
or network is needed. `validation/` contains frozen preparation receipts.
The primary gate deliberately adapts the prior WIDE51 pinned-clang23 parsers.
Fourteen test methods cover 136 positive/negative CLI fixture checks. Both
metadata and remark parsers also accept the prior qualified 13-kernel receipts.

The source test compiles the actual extracted order-key, weighted radix choice,
count/key-writer, and emission sections, with 100,014 raw float-bit cases and
3,136 adversarial geometry/ownership cases. CPU modeling covers histogram
aggregation, scan prefixes and barrier happens-before, so this cannot validate
actual device barriers or runtime behavior. Dispatch tests compile the actual
host function with launch recording for gfx906 HIP, other HIP and CUDA paths;
they exercise exact flag strings, capacities, counted calls, FIT17/REG/OLD
interactions, guard identity, and invalid template assertions.

`provenance/cpu-source-proof.json` preserves the prior controller evidence:
100,014 float cases, 3,136 source cases, and 16 real 64K/200K capture rows.
Its SHA256 is `fd70b55229c2fbb3ccfbd18da3415ddc6dbbe6d650866821fe4452fdb2b0b3bf`.
Raw captures and a capture re-run are not included in the portable package.
`provenance/cpu-dispatch-proof.json` preserves the separate prior dispatch
receipt. These receipts predate this resource compilation and are source-only.

## Build receipts and failure handling

Outputs include both actual gfx906 device ELF files, assembly, disassembly,
metadata, symbol listings, resource remarks, preprocessor-shim proof, fresh CPU
test receipts, a resource-gate JSON report, and both new ISA-body extracts.
`build.receipt.json` explicitly records that GPU/full-engine work did not run.
`resource-isa-receipts.tar.gz` preserves gate pass or failure evidence.
The outer wrapper always preserves compiler logs, exit code and `OUTPUTS.sha256`
even if compilation aborts before the gate. Reuse no previous output directory.
No publishing, GPU qualification, graph replay or performance measurement is
authorized or performed by this package.
