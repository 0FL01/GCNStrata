# gfx906 kernel tuning on four MI50 cards (2026-11-14)

Search for prompt (PP) and generation (TG) speed on four Radeon Pro VII / MI50 (gfx906) cards by changing the engine
rather than its switches or the GPU clock. Follows `../2026-11-13-gfx906-four-gpu-tuning/` (same harness, same
65,536-token prompt, 512 generated tokens, fixed seed, one process per run, production stopped for each series).

Profile: engine 0.1.45 at `53e6c6a5`, Qwen3.8-Flash-Next UD-Q2_K_XL, four stages of 12 layers, `--pipeline-windows 2`,
`--spec 6 --mtp-max-t 2 --spec-min-p 0.5`, `--prefill 4096`. All 56 runs are in `results.json`.

## Result

| | PP tok/s | TG tok/s | output |
|---|---:|---:|---|
| production before, 1350 MHz | 1160-1173 | 58.2-62.5 | `d6ba5e48de` |
| new binary, same switches, 1350 MHz | 1185-1191 | 65.1-65.5 | `d7fe802db3` |
| **new binary + `STRATA_HIP_PROMPT_F16=1`, 1350 MHz (deployed)** | **1370-1437** | **70.1-72.3** | `4bc5acdaad` |

- PP: +22% (median of nine runs at 1350 MHz 1427 against 1164 in five). At 1400 MHz the same change measured 1449-1479
  against 1180-1196.
- TG: the cost of a decode window did not change (fresh window typically 34.8-35.3 ms before and after, with single runs at
  36.5-38 ms on both sides; the first card's expert phase 1.68 -> 1.33 ms). The higher TG figure comes from a different continuation: each numerics change gives
  another 512 tokens with another draft acceptance (208/271, 215/269, 231/268). It is not a kernel gain.
- Every variant repeated its own output exactly between runs.

## What changed

1. **`STRATA_HIP_PROMPT_F16=1`** (existing switch, off in the deployed config): the prompt path's 16-bit GEMMs run
   FP16 in and out through rocBLAS instead of BF16 in / FP32 out. Per 4096-token chunk on one card: 2830 -> 2186 ms;
   hc read 540 -> 245, gdn 380 -> 250, qsa proj 190 -> 137, router+shared 102 -> 61 ms. The method and the
   distribution check are in `../2026-10-04-rdna2-fp16-prompt/`; it was never measured on gfx906 before.
   - Range (`STRATA_F16_RANGE=1`, the 65,536-token prompt): activations up to 73, BF16 weights up to 8.4, GEMM outputs
     up to 344; nothing beyond 65504, nothing non-finite.
   - Needle recall (`tools/needle_bench.py`, 8k / 32k / 64k at depths 10 / 50 / 90): 9 of 9 before and after; the
     64k requests took 44-56 s before and 37-47 s after.
2. **IQ2_XXS gate/up in LDS on gfx906** (`kExpLds16 = true`, `src/kernels/cuda/iq_kernels.cu`). The comment said R2 "as
   measured on gfx906"; on this card `native_expert_bench` gives, bitwise equal, 16 experts x 1 entry 132.5 -> 78.8 us
   and 8 x 2 128.2 -> 56.5 us. Five layers of this model use IQ2_XXS, all on the first card.
3. **Q5_0 expert layers through MMQ** (build option `STRATA_ORCA_Q4KS_MMQ=ON`): one layer of 48 left the
   dequantize-and-BLAS route. About 1-2% of PP, inside the noise; it changes that layer's rounding.
4. `gr_norm_split_fast_kernel` is the gfx906 default (the deployed config already set `STRATA_GR_NORM_FAST=1`), and
   `STRATA_VERIFY_NODES` names the kernels in the gfx906 build.

## What was measured and not adopted

| Variant | Result |
|---|---|
| per-format expert mode table (IQ2_XXS / IQ3_XXS gate/up fused, mode 8) | bench: 8-13% per call, bitwise equal; engine: window 34.9 ms either way, PP 1336-1342 against 1425-1431 in two of three runs |
| `STRATA_GR_V3=1` unlocked for windows up to 6 tokens (64 KB of LDS) | read 46.7 us against 50.3 at T 1, 69.5 against 60.8 at T 2 |
| `STRATA_HC_PERSIST=1`, `STRATA_LFUSE=1`, `STRATA_HIP_SWIGLU_FUSED=1` | no change |
| `STRATA_DENSE_MMQ=1`, `STRATA_MMQ_GROUP_SORT=1`, `STRATA_TOPK_REG_MIN_BLOCKS=4096` | no change |
| `STRATA_RDNA2_SGEMM=all` | PP 1133 |
| `STRATA_ATTN_LANECELL=1`; `STRATA_ATTN_MERGE_V2=1` | window 34.8 ms against 33.9; PP 1290 |

`STRATA_EXP_MODE=16`, `STRATA_GR_NORM_FAST`, `STRATA_GDN_*`, `STRATA_GFX906_ATTN_QUERY_SWIZZLE`, `STRATA_HC_F16_EXACT` and
`STRATA_HC_BF16_N4` are already set in the deployed config, so repeating them changes nothing.

## Where the time is (1400 MHz)

A decode window on one stage (12 layers, 655 graph nodes, 7.8-10.1 ms; a fresh window is the sum of the four stages):
experts 21%, hyper-connections 21%, dense Q8_0 projections about 20%, router + shared expert 14%, QSA about 14%, the head
1.6 ms on the last stage.

- An empty kernel in a graph costs 1.7-2.3 us on this card: 655 nodes are 1.1-1.5 ms. Fusing launches cannot give much.
- Weight read rates: dense Q8_0 about 400 GB/s (memory-bound), IQ4_NL down 277, IQ3_XXS gate/up 227, IQ2_S 200,
  BF16 hyper-connections 170-230, Q2_0 down 134-166.
- The pipeline's last stage never runs a speculative window and the drafter's chain starts from a verified window, so
  a run of accepted windows yields two windows per (chain + 35) ms. Depth 3 would need a third verifier, snapshot and
  hand-off buffer per stage and a rewrite of the A / B / D loop; not attempted.

Prompt path after the change, per chunk: expert MMQ 781 ms + its gather 76 ms (39%), QSA attention 284 (13%), GDN 408
(19%), hc read 245 (11%). The expert MMQ kernels are the largest remaining block.

## Hardware notes

- 1400 MHz / 1000 mV ran for about six hours, then card `0b` reported 33 correctable and 15 uncorrectable ECC errors
  in its gfx block (TCC) at the start of a prompt, cold. All cards went to 1350 MHz / 1000 mV; no new errors in the
  following 21 runs.
- `sweep.py` now retries a missed sensor poll (one EIO from a card's SMU stopped a series) and takes `EXE=<binary>`.

## Files

- `results.json`: every run (variant, switches, PP, TG, drafts, output hash, fresh / speculative window times where
  `STRATA_DECODE_TIMING=1` was set).
- `sweep.py`: the harness as used.
