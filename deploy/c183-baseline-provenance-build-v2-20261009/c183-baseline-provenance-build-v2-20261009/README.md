# Original c183 source provenance reproduction

Root-executable, opt-in CPU build script. Preparation did not compile HIP, contact MI50, run the engine, or execute a GPU. This package rebuilds only ORIGINAL gemm.cu and prefill.cpp, and relinks unchanged ORIGINAL objects/libraries. It does not build or apply the observer.

## Immutable inputs

The fixed MI50 source tree is /home/radneon/strata-rnd-hc-f16. It is mounted /src:ro. Package files are /probe:ro. The original build remains /src/build-906 and the working directory remains /src/build-906. No make/CMake regeneration, clean, archive rewrite or source modification is performed.

The locally available image is pinned to sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646. Docker is run with --pull=never, --network none, no GPU devices, read-only root filesystem, dropped capabilities, no-new-privileges, a private writable /tmp,4 CPU quota,12GiB memory and512 process limit. /dev/kfd and /dev/dri must be absent inside the container. The sole persistent writable mount is the fresh output directory, /out:rw.

Source hashes required before compilation:
- gemm.cu:6f73546e4b654a9cc81f4b333a09530c7425e13f6ce33ac16ecb88a05b07fabd
- prefill.cpp:1a8855dd4e1d6116f10060f2be2110014ecc2afb9e80c9164fc0689c8f175b42
- gemm.hpp:a46c76cad934f4c4114a0f984aee2a7145c8549fc7b1efee16b09d62aab2cfaf

The original standalone objects and their unique members in libstrata_prefill.a must both match:
- gemm.cu.o:9b0d9fb8cf20c97bb81e43b363e6a6b68d32af0d36234b90ed058ad4433ad048
- prefill.cpp.o:efff91107cf3339e5929ac510097168da5c51d64012a93ece3894aa64b5d215d

This archive-member check is necessary: matching a loose object alone does not establish that the original binary linked it. The final exact relink ties the checked archive to frozen c183.

## Exact build commands

exported-build-metadata.txt preserves the root's original flags.make, build.make and link.txt export. All six compile flag groups, the original object recipes and the original link token sequence are verified against the live read-only build metadata. The macro STRATA_VERSION retains its literal quoted string value.

gemm uses /opt/rocm/lib/llvm/bin/clang++ with the exact HIP defines/includes/flags, including -O3,-DNDEBUG,-std=c++20,--offload-arch=gfx906,-Wall,-Wextra and -x hip. prefill uses /opt/rocm/llvm/bin/clang++ with the exact CXX defines/includes/flags and original -MD/-MT semantics. The only command-path changes are -o output files and prefill's -MF dependency output, redirected into /out/reproduction/objects. Absolute source paths remain /src/src/prefill/...; the working directory remains the original one.

Each rebuilt original object must match its pinned SHA256 before the next step. Any mismatch stops immediately. Path-sensitive compiler metadata or nondeterminism is a reason for analysis, not a reason to weaken the hash gate or retry with different flags.

After both object matches, the exact original executable link command runs against the unchanged original generate.cpp.o and libraries in /src/build-906. Only its -o argument becomes /out/reproduction/strata.relinked. It does NOT link the newly rebuilt objects or rearchive libraries, because the goal is to reproduce frozen c183 from original linked inputs. Every unique link input is hashed before and after, and command order including repeated archives is preserved.

strata.relinked must exactly hash to c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049. The generated program is never executed. generate.cpp is not recompiled: its original object is unchanged and bound through exact binary reproduction. This does not claim that the current generate.cpp source itself was recompiled/proven by this gate.

## Bounds and evidence

Each compile has300seconds; link180seconds; version queries15seconds each; archive commands30seconds. A timed-out command's private process group is killed and waited. The outer Docker operation has a1000second deadline and cleanup has separate20/10second bounds. No retries are performed. Resource exhaustion or timeout is a terminal failure requiring analysis.

Commands, argv, working directory, flags, image, compiler/tool versions and hashes, input/member/output hashes, stdout/stderr, exit codes, timeout facts and cleanup are retained. Partial objects/binaries survive failure. Inner build evidence is in reproduction/receipt.json; host/container evidence is in host-receipt.json. All source/build input checks are repeated, and original link inputs must remain unchanged. An object or binary mismatch cannot PASS. Original source/build directories are read-only throughout.

## Root invocation after independent review

Transfer the sealed package to a separate location outside the original tree. Choose an absolute output path that does not exist and is outside both the original source/build tree and this package.

python3 run_reproduce.py --output /absolute/fresh/provenance-output

This checks hashes/metadata only; no container or output directory is created.

python3 run_reproduce.py --output /absolute/fresh/provenance-output --execute

Only root executes that opt-in build after review. Success means original gemm/prefill objects and frozen c183 binary reproduced exactly. It authorizes no observer build, GPU/model execution or source modification. If any gate fails, retain the complete directory and stop for analysis.

CPU tests: python3 -m unittest -v test_reproduce.py. Tests parse the actual exported flags/link, compare command token sequences including allowed output-only changes, enforce container isolation, and run tiny ordinary Python subprocesses to verify nonzero/timeout evidence and process-group cleanup. They do not invoke Docker, ROCm, HIP compilation or a model.

## v2 ownership correction
The first v1 run failed at /out/reproduction mkdir before compilation (PermissionError13). The observed error does not itself prove the image default UID caused it. v2 explicitly runs as the invoking host uid:gid, verifies the fresh output owner uid/gid and0700 mode, and retains all isolation and exact compile/hash gates. No chmod, extra capabilities, devices or network are added. The inner entry now emits a structured failure to captured stderr even if its first mkdir fails, and attempts a fallback /out/inner-failure.json; if the mount itself is unwritable stderr is authoritative. Original v1 package and receipts remain unchanged. Run against a NEW output directory.

Root read-only query subsequently confirmed host and output1000:1000, output0700 and pinned-image Config.User=null(defaultroot), supporting the ownership diagnosis. Query receipt spool1791583198646-6071.log. The wrapper requires real/effective host uid/gid agreement.
