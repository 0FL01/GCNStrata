# v4 exact baseline verifier recovery

Minimal extension of reviewed v3. CPU-only compile/link; root executes after independent review. Never run strata. v1/v2/v3 package and run receipts remain untouched.

Diagnosis: original restored c183 executable coexists with capture-era engine archive. Compile only the exact archived baseline verifier, replacing only verify.cpp.o in the scratch engine archive, then demand exact full c183 SHA256. Existing gemm/prefill source, object and archive checks remain, without recompiling those already proven objects.

Immutable baseline file overlays preserve original /src paths: source verify.base.cpp from qsa-200k-capture-v2-20261009; headers from strata-primary-commit-overlap match archived compile receipt hashes exactly. Three pinned copies in baseline/ mount read-only over original paths. Other source files remain original read-only. Original source/build tree independently mounts /original read-only. Only fresh bounded minimal build copies mount writable at original /src/build-906 path, preserving exact original argv and compiler working directory. Engine compile argv expands exported actual CMake recipe and exact pinned flag token groups. No new flags, normalized outputs or CUID overrides.

Every engine archive member is uniquely listed, extracted and hashed before/after repair. Order and names must be identical; only verify.cpp.o may differ. Original member must match capture SHA880188624ff973f9dd3fb99259a5d8ad9293a5dfef27aa101b91f01585a5616a. All other link inputs remain byte-identical. Final output must equal c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049 or stop with all failure evidence. The repaired verifier object has no independently known original object hash; final complete binary equality is the controlling provenance gate.

Pinned image sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646. Host UID must be1000 with matching real/effective identity, output owner and0700 mode; container explicit uid:gid matches. Network none, no GPU devices, pull never, read-only rootfs, cap-drop ALL, no-new-privileges,4 CPUs,12GiB memory,512 pids,1000s outer timeout. Each compile300s/link180s/archive extraction30s. Original closure copy bound4GiB/120s. No source/build mutation, model run, inference, observer, network or GPU access.

Invoke once on MI50 only after review from this sealed package:
python3 run_reproduce.py --output /absolute/fresh/provenance-v4-output --execute

Without --execute validation only. CPU unit tests invoke no Docker/HIP compiler. Inspect host-receipt.json and reproduction/receipt.json including all command/stdout/stderr/member hashes. Failure retains every scratch output. Preserve earlier run directories; use fresh output.
