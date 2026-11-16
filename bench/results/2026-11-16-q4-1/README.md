# bartowski Q4_1 on four gfx906: the new base model

**Decision (2026-11-16): bartowski `Qwen3.8-Flash-Next-Q4_1` (revision `db3a185731e162be9ffa4d52d20d7509939e60e8`)
with the shared IQ4_NL PLE is the base model of this fork on the four-MI50 rig.** Further work builds on it. The Unsloth
UD-Q3_K_XL deployment is deferred: its weights, pack, config and release binary stay on disk and are not deleted.

Runtime: `main` 7cc69bd5 (Q4_1 dense MMVQ, dequant, native PLE rows and grouped-MMQ prompt experts).
Binary SHA256 `cc2d87d45289952398ac41341d520d6ab9b3f8b79c511b4adebdcb5314d08d2f`, the pinned image
`sha256:bccb7ee7…`. The deployed config is [`evidence/deploy-config.json`](evidence/deploy-config.json).

## Prompt speed: three stacked causes

Workload: one 65,536-token raw-ID prompt, 512 output tokens, int8 KV, 262,144 context, `--kv-resident 32768`, MTP
spec 6 / max T 2, `--pipeline-windows 2`, four gfx906 in a layer split, a new engine per run.

| Run | Setup | prompt tok/s | decode tok/s |
|---|---|---:|---:|
| Q3 production before | Unsloth UD-Q3_K_XL, `--prefill 4096` | 1148.3 | 61.3 |
| Q4_1 before | FP16 dequant prompt experts, 4096 | 647.0 / 648.0 | 69.2 / 68.9 |
| + grouped MMQ | 4096 | 580.6 | 65.5 |
| + `--prefill 8192` | | 881.0 / 870.0 | 61.4 / 61.6 |
| + `--aux-cpus 16-31` | **deployed setup** | **1478.9 / 1518.3** | **61.9 / 61.5** |
| same, second 64K request of a process | | 1566.5 | 73.9 |

1. **Expert products.** Without an MMQ instance Q4_1 experts were dequantized to FP16 and multiplied per expert
   (`STRATA_PREFILL_TIMING=1`, one stage, one 4096 chunk: gate/up ~2100 ms, down ~800 ms). Q3's IQ3_XXS experts on
   MMQ took ~550 / ~330 ms. With `q4_1` in the MMQ list Q4_1 takes ~350-400 / ~210 ms.
2. **Residency.** 70.3 GiB of Q4_1 experts leave 17,174 of 24,576 resident (Q3: 22,698). Each chunk streams 9-12 GB
   per stage from pinned RAM, so an 8192-token chunk halves the streamed bytes per token.
3. **One CPU for every helper thread.** The pool owns CPUs 1-15 and the host thread CPU 0 (16 cores, 32 threads).
   The serving compose sets `STRATA_HTTP_AUX_CPUS=1`, and `--aux-cpus auto` then finds a single spare CPU, 16. All
   prefill stage threads, the streamed-ring issuers and the driver threads share it. Per-copy events showed H2D at
   5.6 GB/s with one active stage and 2.8 / 1.7 / 1.3 GB/s per card with two, three and four. A standalone test
   from 56 GiB of registered 4 KiB pages reaches 21.7-23.6 GB/s for all four cards. `--aux-cpus 16-31` (the SMT
   siblings of the workers, idle during a prompt) gives 6.1-6.5 GB/s per card at once and wait-copy ~10 ms per
   8192 chunk. Ruled out: IOMMU translation, DRAM bandwidth (IMC ~5 GB/s), KV host writes and blit copies.

The earlier MMQ attempt (586 tok/s) was rejected because the faster compute only exposed cause 3. In that earlier
setup the copies were hidden behind the slow FP16 GEMM.

## Accuracy: MMQ path against the qualified FP16 path

The FP16 dequant is byte-checked against ggml (`qualify.sh`, layers 0/1/2/3/20/47). Same binary, path switched with
`STRATA_PREFILL_MMQ=0`. The reference for acceptable divergence is the same pair on Q3, whose MMQ path is in production.

| Check | Q3 MMQ vs FP16 | Q4_1 MMQ vs FP16 |
|---|---|---|
| greedy, 25 prompts (21 tasks + 6K/8K/16K/64K), identical outputs | 15/25 | 14/25 |
| diverging at the first token | 2 | 3 |
| final residual, positions 57K-65K: rel. L2 median / p90 | 0.187 / 0.74 | 0.185 / 0.67 |

The Q4_1 divergences are near-ties: JSON with or without line breaks (both pass), a blank line or type annotations in
code, and one wrong no-reasoning answer (C vs B, expected D). On the frozen 24-task set
(`quality_vs_llama_20261115`, dataset SHA 270fce9b, network-none grader, code budget 512): **Q4_1 MMQ 11/24**,
Q4_1 FP16 10/24 (the FP16 run hit the budget on C08). Q3 had 11/24 on Strata and on llama.cpp. No accuracy regression
from MMQ was found. This is a regression check, not a model-quality benchmark.

## Limits

- RAM: experts 70.3 GiB + shared PLE ~28.8 GiB. The minimum MemAvailable during a 64K request was ~8 GiB. The native
  Q4_1 PLE embedded in the GGUF (+3 GiB, reader supported and checked) is not used.
  Deployed with the CPU vision encoder (0.9 GiB), idle MemAvailable after the start smoke is 6.3 GiB
  (`evidence/deployment.json`). That is below the conversation cache's 8 GiB free floor, so parked conversations
  are limited. RAM headroom is the first thing to improve on this base.
- A fresh Q3 process often pays a cold first request (lazy code objects, capture). Q4_1 did not show it in these runs.
  The effect of `--aux-cpus 16-31` on the deferred Q3 config was not measured in the serving protocol.
- Decode speed was not tuned. Differences between runs follow the MTP trajectories.
- Vision (CPU mmproj) is configured as before. No image end-to-end test was run on Q4_1.

Files: `candidate.patch` / `tests.patch` / `qualify.sh` / `runtime.sha256` are the pre-MMQ qualification provenance.
Full lab data is in `/home/radneon/strata-rd/q4-1-20261116/` (`exp/` holds these runs).
