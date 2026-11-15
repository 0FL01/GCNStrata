# Q5_0 down: confirmed mixed-activation offset error on gfx906

## Verdict

The production-source baseline **was missing** the arithmetic correction from
[upstream aa678f10](https://github.com/Niko1221/Strata/commit/aa678f10d0e51ac17d301c75cbb6b80783437e8b).
The error was reproduced on all four gfx906 GPUs and on real Q3/Q2 expert rows.
The minimal correction passes the numerical checks below and changes Q3's logits.
This is not evidence that Q3 is generally better/worse than Q2 or BF16, or that it
produces better Three.js projects.

## Provenance and dispatch

- Fork baseline: `d3a7502d1b871ae55de0fec8fed8baf7a8c6a0e4`.
  Remote lab: `645a3c17c42c6ddf406d110ee8eed0b8cf78e2d1` plus the same three-line
  capture-heartbeat change. `src/` and `include/` match.
- The running `/proc/505893/exe`, release, and an independent clean baseline build
  had identical SHA256: `d398fe48e8219336e4c169e1d1e35b3dd4cd6db404e1250cc34c1aa9505a8c33`.
- Candidate (without diagnostic instrumentation):
  `fbc52b5228c32709e86b009fe690db5c3edd9f83f0d2433fdc72fffc87f775b2`.
- Build image: `sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646`;
  HIP `7.14.60850-0000000`, patched AMD clang 23; Release/O3, `--offload-arch=gfx906`.
  `STRATA_HIP_GFX906`, `STRATA_ORCA_Q4KS_MMQ`, `STRATA_MMQ_KQUANTS`,
  `STRATA_GFX906_HC_BF16_CLONE`, `STRATA_GFX906_HC_BF16_N4`,
  `STRATA_GFX906_HC_F16_EXACT` enabled. Pinned clean llama.cpp:
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- `build.make` compiles `src/kernels/cuda/iq_kernels.cu` as HIP into
  `libstrata_kernels.a`, linked into `strata`. Local/remote source hashes matched.
  `aa678f10` was not available as a local Git object (`merge-base` exit128), so
  missing ancestry was not used as proof; the actual formula was inspected.
- In `STRATA_EXP_MODE=16`, Q5_0 down bypasses the specialized gfx906 down routes:
  `native_expert_grouped -> launch_down<6> -> native_down_kernel<6> -> row_dot<6>`.
  The affected `vec_dot_q5_0_q8_1` used `ds.y`, the original float activation sum.
  The correction uses the sum of the same integer Q8 codes as the dot product:
  `d5*d8*(sumi - 16*sumu)`. CPU fallback is a separate ggml vec-dot path.
- `moe_fused_iq.cu` does not supply this Q5_0 down path. Q5_0 MMQ is enabled in
  this build; MMQ/FP16 prompt dispatch is separate. No prefill arithmetic changed.

## Local GGUF confirmation

Headers were read from the actual local split files. Q3 Q5_0-down layers:
`0,1,2,3,4,5,6,13,16,19,20,22,24,25,26,27,37,39,40,47`.
Selected GU/down pairs: layer0 IQ2_S/Q5_0; layers4,13,20 IQ3_XXS/Q5_0;
layers16,47 IQ4_XS/Q5_0. Q2 has only layer4 IQ2_S/Q5_0.
GU shape `[2560,640,512]`, down `[640,2560,512]`. Expert7 is the real-row fixture.
Selected Q3 layers except47 are in shard2; layer47 is in shard3.

## Numerical reproduction and fix

The scalar oracle explicitly decodes signed Q5 codes and consumes the **GPU's
exact Q8_1 codes/scales**. It does not call either implementation's vec-dot.
The whole-expert float reference instead dequantizes the same GGUF weights and
uses float activations; it is not an original BF16-model reference.

Crafted nonzero-mean activation case, T=1:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Output of all-zero signed-weight row (expected 0) | -0.2734375 | 0 |
| Relative L1 vs same-code oracle | 0.332304771 | 0 |
| Relative L2 | 0.311926567 | 0 |
| Max absolute error | 1.3671875 | 0 |
| NaN/Inf | 0 | 0 |

Baseline output matches the old `ds.y` expression exactly. This separates the
implementation defect from ordinary activation quantization error.

Real Q3, expert7, T=3 (relative L1; rounded):

| Layer | CPU/float, unchanged | GPU/float before | GPU/float after | Same-hq oracle before | Same-hq oracle after |
|---|---:|---:|---:|---:|---:|
| 0 | .0133 | .0251 | .0116 | .0230853 | 5.95e-8 |
| 4 | .0131 | .0250 | .0114 | .0231296 | 6.07e-8 |
| 13 | .0127 | .0233 | .0113 | .0213097 | 5.99e-8 |
| 16 | .0128 | .0214 | .0109 | .0191229 | 5.99e-8 |
| 20 | .0135 | .0226 | .0118 | .0202445 | 5.97e-8 |
| 47 | .0148 | .0206 | .0126 | .0170365 | 6.00e-8 |

The original whole-expert 0.03 threshold alone passed these faulty GPU results.
Q5_0's GPU/float threshold is now 0.018, as upstream; the independent same-code
oracle uses 1e-5. These are different contracts, not a universal CPU/GPU limit.

All 27 candidate invocations passed: synthetic pairs, selected real Q3 and Q2
rows at T=1,2,3,4,6; grouped modes0,2,7,16; and the crafted dot on GPUs1,2,3 in
addition to GPU0. No NaN/Inf. The grouped test checks entry counts separately
from its six-token window; v1/v2 equality alone cannot detect their shared old bug.
GPU-vs-optimized-CPU whole-expert metrics use different activation quantizers;
the aligned-code oracle is a scalar CPU reference, not the optimized CPU kernel.

`numerical-results.json` retains L1/L2/maxabs/nonfinite values and raw-log hashes
for all cases. Initial real-GGUF attempts failed to resolve the absolute shared
engram symlink; `baseline-real` contains the corrected, measured reruns. Weights
were not modified. `parity.sh` records the commands; the container must mount
`/home/radneon/models-nvme/shared-engram` at the same absolute path, read-only.

## Teacher-forced Q3 impact

635 fixed prefixes (lengths1..127 in each of five frozen quant_quality segments),
all 248320 vocabulary logits, same Q3/IDs/settings, greedy GEN1, pipeline2,
INT8 KV and FP16. Generated tokens were **not fed back**. Full-logit finite/size
checks and zero-reuse checks passed. This is a bounded corpus, not a 256K audit.

The existing serial-only dump hook did not cover pipeline2. Identical audit-only
instrumentation was applied to both diagnostic builds, after final-stage
`pl_finish` and before commit/parity reuse. It asserts GEN1, T1, and the last
supplied prompt token, and records the final GPU device and row. All635 records
were verified per variant. This instrumentation is retained as
`pipeline-first-logits.patch`, **not applied to the shipped runtime**.

Diagnostic binary hashes:
- baseline `a4a2580b19c97b18ea5eadefb41b26d45b64eefc67858181fca9d36484b20067`;
- candidate `c15e00620d7a56905081573c634c66062f0fcb3e020d2ca01e0ad26c08fb05f9`.

Full `KL(P_baseline || P_candidate)` mean **0.02137260674**, p99 **0.1595997914**;
top1 agreement **93.07086614%**. First top1 disagreement: code prefix length30,
baseline token318 vs candidate31950. Their respective logits are recorded in
`quality-summary.json`; every scored position is in `quality-rows.json`.
Neither backend is a BF16 reference. This measures impact, not a quality score.

## Performance: matched Q3 65536/512

User approved keeping **1350 MHz / existing voltage profile**, superseding the
initial 1300 MHz request. No clock/voltage writes. All four GPUs; same SATA Q3,
NVMe pack, shared IQ4_NL, INT8 KV, MTP, seed12345, temperature1/top-p.95/top-k20,
aux_cpus1, no prompt/conversation reuse. Fresh engine per run; load excluded.

Initial ABBA completed, but A1 overlapped diagnostic-copy I/O at request start.
It is excluded. One extra isolated baseline A3 replaces it. Thus the clean
reported order is **B1,B2,A2,A3**, not a clean ABBA; order/cache drift remains a
limitation. No builds/copies overlapped these four requests; OS cache not flushed.

| Run | PP tok/s | TG tok/s | MTP accepted/offered |
|---|---:|---:|---:|
| B1 corrected | 1155.44 | 65.93 | 238/263 |
| B2 corrected | 1146.50 | 67.25 | 238/263 |
| A2 baseline | 1152.54 | 55.81 | 208/270 |
| A3 baseline | 1153.74 | 55.45 | 208/270 |

Means: baseline PP1153.14/TG55.63, candidate PP1150.97/TG66.59;
PP **-0.19%**, TG **+19.70%**. Acceptance **77.04% -> 90.49%**.
Within-variant output IDs repeat exactly. Between variants the first free-output
difference is zero-based35 (1396 vs1017). Later contexts differ, so the TG change
is **not isolated kernel speed**; output and MTP work changed. No PP gain claimed.
See `performance-summary.json` and identity files for inputs and settings.

## Reproduction and limits

Build matching baseline/candidate with the listed pinned image/dependency and
production CMake settings, targets `strata native_expert_parity native_grouped_parity`.
`parity.sh baseline|candidate` runs the numerical checks in an isolated container.
`audit.py` owns only UUID-named test containers and refuses to run beside production:

```sh
python3 audit.py quality --root /path/to/diagnostic --ids /path/to/set.ids.json \
  --tokens 128 --evidence /path/to/confirmed-reproduction.json --out /path/to/quality
python3 audit.py compare /path/to/quality
python3 audit.py perf --root /path/to/audit \
  --evidence /path/to/confirmed-reproduction.json --out /path/to/perf
```

Remote raw artifacts, including full logits, remain under
`/home/radneon/strata-rd/q5-audit-20261114/artifacts/`; compact results are committed here.
The original production Q2 configuration/binary was not replaced by this audit.
After all test containers exited, it was restored: loaded Q2_K_XL + IQ4_NL,
API auth enabled, Docker healthy, original `d398fe48...` binary hash confirmed.
CUDA/SYCL were not built or qualified; the change is to the shared CUDA/HIP source.
No prefill numerical-parity claim, all-expert sweep, long-context logit audit or
V8/Three.js browser-quality series was performed. No new browser dependency or
service was installed. The first free-generation divergence was not instrumented
layer-by-layer; internal same-hq expert comparisons and fixed-prefix logits are
separate evidence surfaces, not a trace of that exact 64K request token.
