# Isolated gfx906 PER51 resource experiment

This package adds one experimental, exact-string-1, default-off option:
STRATA_GFX906_TOPK_WIDE51=1.

The baseline is the already-qualified current FIT17 selector, byte-for-byte SHA256
5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a.
The baseline is not the older pre-FIT17 selector.

The candidate changes exactly two source locations:
1. Permit PER51 for WINDOW_GUARD=2 while retaining the existing LONG_ONLY and guard assertions.
2. Inside the existing enabled FIT17 route, replace only the wide66 launch with guarded wide51 when the new option is exactly "1" and reach <= 1024*51 (52,224 blocks).

TK_PER_MAX remains 66. Device guards and arithmetic are byte-identical. There is no extra launch: eligible calls still launch guarded ref256, reg17, and one guarded wide kernel. Default-off, FIT17-off, counted calls, OLD, nq > 8, CUDA, non-gfx906 and larger capacities keep their prior dispatch. Existing FIT17 trace text is deliberately unchanged and is not evidence of which wide variant launched.

## Safety contract

In this route, extended_decode implies counted=false, so reach=max_blocks is the immutable allocation capacity. Every valid row must satisfy n_bid + 1 <= max_blocks. Thus capacity <= 52,224 proves ceil((n_bid+1)/1024) <= 51 for every valid row, including the planned 200K allocation of 51,202 blocks. This does not add support for invalid rows beyond the allocation.

This package does not change any published branch, existing source tree, installed engine, API state or GPU runtime. The package contains no GPU command, GPU qualification harness, support archive or full-engine build.

## Root invocation on the build host

Copy/extract this package into a new INPUT directory, then run:

    bash INPUT/build-host.sh INPUT /home/radneon/strata/logs/perf-campaign-20261009/qsa-wide51-resource-only

The output directory must not already exist. The wrapper takes the same shared engineering lock and uses the pinned image:

    sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646

Docker has no network and no GPU mounts. The inner script rejects /dev/kfd or /dev/dri, verifies clang23, and uses the exact original COMMON flags and include order, including the public platform HIP shim. It compiles only the current-FIT17 and candidate selector translation units. Lightweight extracted-source CPU tests are also compiled; no selector executable, engine or GPU test is linked or executed.

## Outputs and gate

- Actual saved gfx906 device ELF images and assembly for both selector TUs
- Disassembly, ELF metadata, symbol lists, compiler resource remarks and version
- Full device preprocess output and platform-shim/half-wave-ballot proof
- CPU extracted-source dispatch and guard checks
- resource-gate.json and wide51.guard2.isa.txt
- build.receipt.json, OUTPUTS.sha256 and resource-isa-receipts.tar.gz

The automatic gate requires every existing kernel's actual machine-code bytes, metadata and compiler resource remarks to remain identical. Exactly one guarded PER51 kernel may be added. It requires wave64, the unchanged 1024-thread bound and 32,908-byte LDS layout, strictly smaller scratch and VGPR spill counts, and no SGPR/register-allocation/occupancy regression versus guarded wide66.

A gate failure exits nonzero and is a hard stop before any GPU work. A pass is only a resource/unchanged-ISA result: root must inspect the new guard/control flow before separately authorized GPU qualification. Smaller PER, fewer spills or lower scratch are not a speed claim.

## Source tests already run on the controller

The tests compile the actual extracted host dispatch function with launches replaced by recording calls, separately for gfx906 HIP, non-gfx906 HIP and CUDA-preprocessor paths. Each env case runs in a new process because the production flags are cached statics. The source transform is checked against the complete baseline and canonical patch.

Coverage includes capacities 52,223 / 52,224 / 52,225, nq 1/4/6/8/9, allocation 51,202, legacy and large-capacity edges, exact enabled/disabled env strings, counted, OLD, FIT17-off, REG-off and no-query paths. The actual extracted device guards run as CPU predicates with the actual 24,576-cell short threshold and step indices. Compile-time negative tests preserve invalid guard/PER and LONG_ONLY rejection. An exhaustive integer ceiling check covers every valid row count through the eligible capacity boundaries.

See validation/controller-source-proof.json. These are source-contract checks, not GPU correctness or performance tests.

## Qualification limits

The real 200K score/steps input has not been captured. The existing 64K capture exercises reg17; it cannot qualify wide51 performance. This experiment is intentionally one PER51 resource build, not a PER sweep. GPU parity, graph replay and realistic wide51 performance qualification remain separate later work.

provenance/ contains copied original build scripts, public header provenance, and qualified text receipts for comparison. It does not contain the old device-image archive. The controller copy of the older public receipts archive and standalone base device image were found truncated; fresh images produced by this package are required.

