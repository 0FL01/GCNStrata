# Fixed-scale HC exhaustive raw parity, correctness only

Root-only, opt-in GPU0 controller. No model, engine modification, device compilation, performance timing, deployment, or API/power-setting changes. Preparation compiled only a plain CPU C++ test with g++; it did not build the HIP driver or execute a GPU.

## Exact coverage

For scale4096 then scale512, visit all 65,536 BF16 encodings in ascending order, each with n=1 and both exact baseline and candidate arms: 262,144 scalar launches. Each record has separate zeroed 16-word stats and guarded input/output/stats. Each 256-pair batch is synchronized and copied before the next batch. Three device buffers are allocated once for the maximum batch footprint and reset by batch copies; there is no allocation, transfer or synchronization per scalar pattern. The baseline uses five pointer-to-argument entries including float scale; the candidate uses exactly four, omitting scale.

After each scale's scalar pass, test n=0,1,63,64,65,255,256,257,448 at stats offsets0 and5, both arms: 72 extra launches. Mixed input begins with +0,-0, subnormal endpoints, minimum normal, +/-1, largest finite values, infinities and signed NaN examples, then deterministic mixed payloads. Stats buffers have 23 words: pointer at word1+offset, 16 valid zero words, every exterior word a canary. Output/input each have n+2 halfwords, exterior guards. n=0 still launches one guarded block and must leave stats zero. Larger cases use ceil(n/256) blocks, 256threads/block.

Success requires exactly548batches,131108pairs,262216launches, all input words unchanged, exterior guards intact, and all baseline/candidate output and stats words equal. Every stats word is compared for every pattern; an aggregate OR is not used. Valid stats indices1,2,3 must each be0 or1 and every other valid stats word must remain0 in both arms. This is equivalence to the current exact baseline, including baseline signed-zero/NaN behavior, not a new mathematical oracle.

## Provenance and previous tiny distinction

baseline.elf SHA256 5abe1289b549a9fa4ad7da00398fd81179a6d3b414bd2b3c44b749d5e315f377; candidate.elf SHA256 3f0c5ecb285232b1e338bf0aa51ebe729eefaa547388c14ebbd488167105ecf9. Independent resource-gate SHA256 b4da67f44d01b0a9af96358ba897640d5e6906afff3658838010c03829e52659. Unchanged baseline-binding.json is retained.

Previous actual tiny native run passed14pairs/28launches and cleanup. Its original wrapper parser failed on three unescaped operation strings; that wrapper did not PASS. The independent raw recovery passed, SHA256 a2d839c4a1c260dca9c999b87830b3abe11f9ce43a021859ec9326362d7b0a7b; recovery archive SHA256 a953ec53c723d6d4f19c0e4703afbcb8c3512c395e4572be9ec3d4a63f06d7f3. This package copies the corrected hip_log.hpp serializer, and tests quotes, backslash, tab, newline, carriage return and a low control byte.

## Root build and execution

Compile only exhaustive.cpp against existing HIP headers/runtime, outside this immutable package. Do not compile candidate or baseline device source. Example in the approved ROCm build environment: hipcc -std=c++17 -O2 exhaustive.cpp -o /absolute/output/hc_exhaustive. Root must record actual compiler command/version/exit and driver hash. Source contains no device code.

python3 run_exhaustive.py --driver /absolute/output/hc_exhaustive --driver-sha ROOT_VERIFIED_SHA256 --output /absolute/fresh/output --leave-api-stopped --execute

Without --execute, validation only; no Docker/GPU operations. Runner verifies manifest, ELF identities, both independent resource gates, driver hash, API/all containers off, exact physical PCI0000:05:00.0, gfx906, high power mode and190W cap. Uses unchanged pinned safety helpers, shared exclusive /tmp/strata-direct-ff34-ce.lock, <=55C cooldown, thermal ceilings at critical-minus10C or lower. The native process installs default SIGALRM and arms a300second hard alarm at main entry, including initialization. This independent process watchdog terminates even if the wrapper is blocked. The wrapper also checks its300second deadline each loop and after process exit; a bounded2second Docker query plus0.2second sleep can delay wrapper detection, but cannot extend the native hard alarm. Docker startup and cleanup retain their separate existing bounds. No retries. SIGALRM termination can truncate the current snapshot; missing completion or nonzero exit can never PASS. Root alone executes this command.

## Raw format and failure evidence

Every batch writes initial.bin before upload, input-copied.bin after input readback, output-copied.bin after output readback, and complete.bin after all stats readback. Intermediate files are explicitly partial snapshots, not complete GPU evidence. Before starting a batch, stdout records its metadata and launches-before count. Every HIP call records JSON-escaped operation and return code. Failures stop immediately; previously flushed files survive. The wrapper preserves stdout/stderr, process exit, every raw binary hash, thermal samples, cleanup status, summary and errors even on malformed JSON or native nonzero exit.

Binary records are little-endian: seven uint32 words [0x48435031,kind,scale,start,pairs,n,offset], followed by2*pairs*(n+2) uint16 input words, the same count of uint16 output words, then2*pairs*23 uint32 stats words. Record order alternates baseline then candidate for each pattern. Kind0 is exhaustive scalar, kind1 mixed tail. Header metadata and exact file sizes are checked independently by recheck.py. Initial files are independently checked for zero stats and active output sentinels; complete files are parsed and checked on CPU after GPU exit, with individual SHA256 hashes in independent-recheck.json; root may run python3 recheck.py /absolute/output again without GPU.

CPU tests: python3 -m unittest -v test_exhaustive.py. Tests compile/run the shared hard-watchdog helper in a short SIGALRM-terminated subprocess, reject duplicate JSON keys and matching corruption in both arms, and test core.hpp argument packing/serialization and actual binary fixtures, reject mutations/truncation, exhaustively enumerate prescribed coverage, validate hashes, and mock native nonzero and malformed-JSON exits to establish evidence/cleanup preservation. These are host tests only and do not establish GPU ABI or parity. HIP driver compilation and exhaustive GPU execution remain not run.
