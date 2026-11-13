# c183 HC fixed-scale candidate: CPU/resource handoff

Status: isolated preparation only. No HIP compilation, GPU launch, model build, or model test has been performed. Actual c183 source and original generic kernel remain unchanged.

## Mechanism and scope

The active scaled exact-input converter uses runtime scales 4096 (weights) and 512 (activations). Its binary-bound gfx906 function contains a generic FP32 division sequence for the roundtrip check: 11 instructions / 72 bytes, from 0x7928 through 0x7968. The compiler already elects one active lane per wave for each of three Boolean guard atomic operations. Manual atomic aggregation is therefore closed.

The isolated file hc_fixed_scale.hip supplies exactly two fixed-scale kernels. It retains the original multiplication, FP16 conversion, /scale expression, bit comparison, and three atomicOr predicates. Only scale becomes constexpr. No approximate reciprocal or fast-math flag is proposed. candidate-active-expression.diff shows the arithmetic/guard delta; the whole HIP file also contains the two wrappers. The candidate ABI has four explicit arguments; the original loaded kernel has five including scale.

This is a prefill-only opportunity: current host eligibility restricts T to 4095/4096 and the two HC matrix shapes. It does not remove TG HC0/HC1 costs. Any later integration must retain alpha 2^-21, the 64-byte stats reset, 40-byte guard copy and synchronization, diagnostic fields, all original BF16 inputs, and unchanged fallback decisions.

## Provenance and finite proof

baseline-binding.json binds exact source, host object, archive, device ELF and 428-byte function to frozen c183. baseline-kernel-metadata.json was parsed from that exact ELF. Baseline metadata: 7 VGPR, 20 SGPR, zero spills, zero LDS, zero scratch, wave64; kernel argument segment 296 bytes (including hidden ABI arguments).

cpu_proof.py enumerated all 65,536 BF16 bit patterns at both scales. In the host IEEE model, all 65,282 non-NaN inputs per scale preserve forward half output and Boolean guard bits when roundtrip division is compared with exact inverse multiplication. The 254 NaN patterns per scale are enumerated but explicitly not CPU-proven. Independently, all 63,488 finite half payloads and both infinities per scale produce the same inverse bits. For finite half h, h/2^9 and h/2^12 are exactly representable in FP32; the combined nonzero magnitude range is 2^-36 through 127.9375. Signed-zero sign is preserved in this arithmetic model.

This proof does not establish actual GPU NaN payload behavior or compiled forward-conversion parity. The original ISA has a mixed FP16 FMA with a zero operand; constant specialization may change its signed-zero behavior. Even though the source keeps /scale, every actual GPU output half bit and guard bit must match the original c183 ELF before timing.

## Resource falsifier after separate build approval

Compile only these two candidate kernels in the pinned compiler/image, using the original device flags and without fast-math changes. Save exact compiler command and source/object/ELF hashes. Reject before GPU work unless both kernels have no scratch, LDS or spills, use at most 7 VGPR / 20 SGPR, remove the generic divide sequence, and preserve the compiler's elected-wave-lane atomic lowering. Record full candidate metadata/disassembly; fewer instructions alone is not evidence of speedup. Any different output-conversion lowering is a correctness risk, not automatic approval.

## Small exact GPU correctness falsifier after separate approval

Load the actual bound baseline ELF through HIP module API and the candidate ELF separately. GPU0 only initially. Before any exhaustive run, perform a fail-fast n=1 set at both scales with separate zeroed stats blocks: BF16 +0 (0x0000), -0 (0x8000), normal +1 (0x3f80), subnormal (0x0001), +infinity (0x7f80), quiet NaN (0x7fc1), and signaling NaN (0x7f81). Compare output half bits and all 16 guard words against the actual baseline ELF; stop immediately on any mismatch, including changed signed-zero forward conversion. Only after this set passes, compare all 65,536 BF16 input patterns at each scale, including every NaN, infinity, signed zero, subnormal and overflow case. Full-array comparison establishes all half output bits but an aggregate OR across all patterns would mask predicate differences.

To expose each guard independently, give each pattern its own zeroed 16-word stats block and issue n=1 baseline and candidate calls pointing at that pattern. This is a fixed 262,144 kernel-call budget for both scales and arms, batched with synchronization and a fixed timeout. Copy and compare all 16 words and half bits after each batch, preserving mismatch details. Distinct guards also check untouched words. Separately use mixed-pattern arrays with n=0,1,63,64,65,255,256,257 and a 192-lane tail, canaries around buffers, and both stats offsets 0 and 5. This exercises wave OR aggregation and tail masks. No correctness launch is used as a performance sample. If this scalar-launch approach proves too expensive, stop and review a smaller exact method rather than silently substituting aggregate OR proof.

## Honest value gate

There is no measured converter-time pool yet. The exact baseline has very low register use and retains the same global input/output traffic; division removal may be hidden by memory bandwidth. Current stderr counts 1,728 primary attempts and 531 taken paths, but attempts increment before scratch-capacity/overlap checks. Neither attempts nor taken counts establish actual conversion launches.

The maximum primary schedule is listed in screen-plan.json: W3276800 at most 1,728 times; X41943040 and X1310720 each at most 810 times; X41932800 and X1310400 each at most 54 times. This covers both scales and five descriptors, at 256 threads per block. It totals 43,032,988,800 maximum converted elements and 172.132 GB of read/write traffic that specialization does not remove. The current +1% PP throughput target needs 1,003.739604 ms saved per request, equivalent to 0.580868 ms per maximum HC attempt across its two converters.

After resource/parity admission and separate timing approval, a disposable GPU0 component screen may use those five descriptors, fixed small warmup/sample counts, warm and one cache-scrub regime, and predeclared synthetic accepted/mixed-reject/all-reject patterns. Record every sample and output check. Let each descriptor's paired baseline-minus-candidate median be delta_j. Compute sum(maximum_calls_j * max(0, delta_j)) for each regime/pattern, and use the most generous predeclared scenario only as an optimistic screening estimate. If even that estimate is below 1,003.739604 ms, park the candidate without a model run. If it reaches the target, that only permits further investigation of actual launch counts and a short-model test. It is not a measured whole-model upper bound, because maximum counts, synthetic data, cache state, overlap and host scheduling differ. Do not rerun to win or multiply peer-GPU time into primary wall-time savings.

No defensible static timing bound currently closes this candidate: opcode count is not a wall-time model. The next cheap decision, if authorized, is the two-kernel compile/resource check, not integration or a model experiment.
