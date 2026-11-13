# v3 exact-original-argv provenance reproduction

Root-reviewed, opt-in CPU build only. v1/v2 and their failure receipts remain unchanged. This is the targeted retry for the output spelling included in Clang HIP CUID hashing; every object and frozen binary byte-hash gate remains strict. No observer or model/GPU execution.

## Minimal isolation change
The original tree /home/radneon/strata-rnd-hc-f16 is mounted read-only at /src and independently at /original. A fresh output-owned scratch-build directory is nested at /src/build-906, writable. The compiler working directory remains /src/build-906. Exact original recipes are expanded without changing any argv token: -o, -MF and -MT retain their original relative spellings. Relink output is the original spelling strata within the scratch build.

Only the explicit relative link inputs, two original standalone objects and three flags/recipe/link metadata files are copied. Absolute /opt/rocm inputs remain in the pinned image. No entire build copy, symlinks or hardlinks are used. Every fresh file copy is hash-verified against its original before/after copying and checked for distinct inode identity and link count1. The closure is bounded to4GiB with120s copying deadline. Host receipt records the file list, hashes and byte counts. Original closure hashes are checked after execution. Inner preflight checks scratch contents against /original; original archive contents, all original metadata and original link inputs remain verified independently. Recompiles overwrite only the scratch standalone objects; the original copied archive remains untouched for the exact original-input relink.

## Unchanged gates and bounds
Pinned image sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646. Explicit host uid:gid with real/effective equality and0700 output ownership. Read-only rootfs, cap-drop ALL, no-new-privileges, no GPU devices, network none, pull never,4 CPUs,12GiB memory,512 pids. Original source/include are read-only. Source/header hashes and exact original metadata, unique archive member hashes, rebuilt gemm and prefill object hashes and frozen c183 binary hash must all match. No stripping, CUID override, extra compilation flags or normalization.

Each compile300s, relink180s; outer container1000s and bounded owned-container cleanup remain. Stop immediately on mismatch. All partial scratch outputs, stdout/stderr, command argv/return codes and receipts remain. Earliest inner failure emits JSON to captured stderr plus best-effort /out/inner-failure.json. Parent performs any MI50 execution after review; preparation tests invoke no Docker or HIP compiler.

## Invocation
From the transferred sealed package, choose a fresh absolute output outside source and package:

python3 run_reproduce.py --output /absolute/fresh/provenance-v3-output --execute

Without --execute, only metadata/package validation occurs. Do not reuse earlier output paths. Inspect host-receipt.json, reproduction/receipt.json and scratch-build outputs after termination.

CPU tests: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_reproduce.py

## Source rationale
Compiler version receipt reports ROCm commit46fcb339fb61119b337f973c7ca9e710a319fdd0 plus a local PATCHED revision. At the official commit, CUIDOptions hashes the input plus all rendered non-input arguments, including -o; ASTContext hashes CUID into HIP suffixes. This supports restoring the complete original argv; it does not establish that all v2 differences are explained or that v3 will pass.
https://github.com/ROCm/llvm-project/blob/46fcb339fb61119b337f973c7ca9e710a319fdd0/clang/lib/Driver/Driver.cpp#L138
https://github.com/ROCm/llvm-project/blob/46fcb339fb61119b337f973c7ca9e710a319fdd0/clang/lib/AST/ASTContext.cpp#L15413
