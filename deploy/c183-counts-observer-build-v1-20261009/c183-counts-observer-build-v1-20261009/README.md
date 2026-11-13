# Exact host-only counts observer build

Root-only opt-in CPU compilation/link package. No model/GPU execution. Minimal extension of reviewed v4 isolation: retain original sources and verified repaired v4 result tree read-only; fresh scratch closure copied from v4; overlay only reviewed gemm.cu, prefill.cpp and observer header; compile those two units with exact original argv/output spellings; replace only gemm.cu.o and prefill.cpp.o in scratch libstrata_prefill.a; relink observer binary. No changes to original or v4 trees.

## Strict input gates
V4 host receipt SHA5696899be076d084e8b1896efffed5ad1981c484b79913a5486fc729f864ad87 and inner receipt SHA1b483e54874fdc44b8497209068daff93eb46b798e4a1d96038f7c728b4fda27 must match exactly; both PASS, exit0, cleanup PASS, inner linked hash exactc183 and actual baseline binary exactc183. Every fresh copied closure file matches the pinned host artifact manifest. Repaired engine archive SHA6999f03f57473776a99b72bf4ebe8766a9c6c751f74affd010a2fccf36412da9 stays unchanged throughout.

Observer sources are the previously reviewed counts observer tar SHAa4490090e3ac141fc13a6a1bfa0c54b6500707cad9a6640e9398eaeb0bf94998, with three individual pinned output hashes in build_inside.py. New header requires a complete tiny prefill include-directory overlay: copy all original regular headers (bounded16 files,1MiB each), add observer header, mount readonly at both /src include location and /out alias. Original header hashes checked before/after; no new include flags. Source file overlays preserve original paths. Original unmodified source and metadata pins are checked independently through /original.

Every prefill archive member is uniquely listed, extracted and hashed before/after. Names/order unchanged; only two specified members differ and each equals its newly compiled object. Every other linker input including repaired engine remains identical. Exact command argv, tool hashes, stdout/stderr, return codes, source/member/object/binary hashes and failure receipts retained.

## Complete device gate
Entire raw .hip_fatbin and .hipFatBinSegment byte arrays must match baseline exactly. Device-bearing section inventory is fixed and exhaustive for this image. ELF section-table bounds and uniqueness are validated. Every bundle is parsed in order with only verified zero alignment padding; unexpected structures fail. Exactly55 complete gfx906 ELF payloads are required. Per-module inventory records every section (including notes/resource metadata) and symbol metadata; complete ELF payloads and all module inventories must match. Wrapper magic/version/reserved, zero raw pointer slots and exactly one ELF relative relocation per wrapper must bind one-to-one to the corresponding bundle. Only the validated host virtual-address relocation difference is permitted; no payload byte normalization, CUID override or relaxed hash gates. The baseline gemm full-device payload remains protected as part of complete raw byte equality.

## Isolation and invocation
Pinned image sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646; no network/GPU; pull never; readonly rootfs; explicit host UID1000 matching0700 output;4 CPUs/12GiB/512pids; original sources/package/v4 inputs read-only. Outer1000s bound; each compile300s, link180s, each archive command30s; original closure copy4GiB/120s. No automatic retry.

After independent source review, root invokes on MI50 with a fresh absolute output outside original tree, v4 results and package:
python3 run_reproduce.py --output /absolute/fresh/counts-observer-build-output --execute

Do not run the generated binary. A later separately reviewed runner owns any model experiment.

Success schemas: host-receipt.json passed=true,status=PASS_COUNTS_OBSERVER_BUILD_DEVICE_EQUALITY,cleanup.passed=true; reproduction/receipt.json passed=true,status=PASS_HOST_ONLY_OBSERVER_BUILD_AND_DEVICE_EQUALITY; reproduction/device-equality.json passed=true and complete_device_payload_bytes_equal=true,complete_module_kernel_metadata_equal=true. Inner receipt pins device-equality hash and observer_binary_sha256. Parent must independently review actual build evidence before runtime admission.

CPU tests: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_reproduce.py. Controller test additionally validates full55-module extraction against frozen baseline and rejects malformed ELF mutation. Tests invoke no HIP compiler or Docker.
