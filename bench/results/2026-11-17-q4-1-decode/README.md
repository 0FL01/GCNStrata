# bartowski Q4_1 decode on four gfx906: decode-once Q4_1 kernels and `--adapt-every 4`

Base: [2026-11-16-q4-1](../2026-11-16-q4-1/README.md) (release 7cc69bd5). Runtime: commit 4a4951f4, binary SHA256
`a4f4e40862be165d9eb9b9c1c8fc2e8952d07fcd756308b0f87488ef4aefb15b`, the same image and build options.
Config: `configs/bartowski-q4-1-4gpu-iq4-ple-adapt4-20261117.json` = the 2026-11-16 config with `--adapt-every 4`.

## What was slow

Recon on the deployed base: a decode window cost ~25-27 ms of GPU time for ~1.4-1.5 tokens. Q4_1 had none of the
"decode once, apply to every token" kernels the IQ formats use:

- the grouped decode experts ran `native_gu_kernel<3>` / `native_down_kernel<3>`: one entry at a time, the expert's
  weights re-read for every token of the window; the down rows (K = 40 calls) left 24 of 32 lanes idle;
- the dense Q4_1 projections ran `mmvq_kernel<3>`: the weights re-read per column.

## Change (gfx906, Q4_1 only)

- `Split<3>` / `kSplit<3>`: `vec_dot_q4_1_q8_1` as load (the weight words, `dm`) and apply (the same dp4a chain and
  float expression). Dense Q4_1 then takes `mmvq_multi_kernel`.
- Grouped Q4_1 experts, `STRATA_LEGACY_EXP=<gate/up><down>`, default 31: gate/up on `s27_gu_kernel` (2 rows per
  warp, the activations in LDS), down on the 8-lane `native_down_multi_kernel`. `STRATA_LEGACY_EXP=0` restores the
  per-entry kernels.

Every output is bitwise the old kernels' output. Each lane sums its calls in the same order, followed by the same
warp tree.

| Check | Result |
|---|---|
| `native_expert_bench`, real layer-1 rows, G 4/8/10/16, T 1-4, mixed groups (pad 20) | bitwise equal, all |
| `iq_multi_parity`: Q4_1 `iq_mmvq` ncols 1-8, 11; grouped Q4_1/Q4_1 | bitwise equal, 0 failures |
| engine, static profile (`--adapt-every 0`), greedy, 4 prompts x 3000 tokens | texts identical to 7cc69bd5 |
| frozen 24-task set + 6K/8K/16K/64K prompts (greedy, 256 / code 512 tokens) | 26/26 and 8/8 token sequences identical: 11/24 |

Kernel times (MI50, one call per layer, 10 experts, `native_expert_bench`):

| Tokens per expert | before | after |
|---|---:|---:|
| 1 | 82.7 us | ~69 us |
| 2 | 139.6 us | ~86 us |
| 3 | 199 us | ~102 us |
| mixed, 14 groups, 1-3 entries | 213.9 us | 114.4 us |

Dense Q4_1 8192 x 2560 (`iq_multi_parity --bench`): 1 column 29.1 -> 27.9 us, 2 columns 50.1 -> 42.6, 4 columns
96.5 -> 72.4.

## Engine

HTTP server like production, one request at a time, 4 prompts (code, explanation, 11.5K-token review, 4K-token edit)
x 3000 tokens, seed 12345, four gfx906 in a layer split.

| Run | Config | TG tok/s per prompt | ms per window |
|---|---|---|---:|
| release 7cc69bd5, greedy (two runs, identical texts) | static | 60.1 / 59.3 / 59.5 / 61.5 | 25.67 |
| 4a4951f4, greedy (same texts) | static | **62.9 / 62.8 / 63.1 / 65.1** | **24.31** |
| old kernels (`STRATA_LEGACY_EXP=0`), sampled | adapt 4 | 55.4 / 55.5 / 51.2 / 59.6 | 25.81 |
| 4a4951f4, sampled | adapt 4 | **60.4 / 62.0 / 57.1** / 93.3 | **24.13** (first three) |
| recon base (2026-11-16 config, sampled) | static | 51.4 / 52.0 / 54.4 | ~27 |

The static greedy pair has the same token trajectory, so it measures the kernels alone: +5.6% TG and -5.3% per window.
Sampled runs follow different trajectories, so compare those by ms per window. The fourth sampled 4a4951f4 prompt
landed on an easy trajectory (1.91 tokens per window) and is not comparable.

`--adapt-every 4` (recon, +5.3% TG) lets the decode expert cache follow the traffic: hit rate 98-99% against 67-92% with
the static profile. Expert placement then depends on timing, so outputs are not reproducible run to run. That holds
even for greedy decoding: CPU and GPU expert products round differently.
Frozen tasks with it (greedy): 14/26 token sequences identical to the static run, which is the same scale as the
accepted MMQ vs FP16 pair of 2026-11-16 (14/25). Score 10/24: instructions 5/8 with the same tasks passing, math 0/8,
code 5/8. The code difference is C08, which ran to the 512-token budget, the same near-tie the FP16 run hit on
2026-11-16.

## Not kept

- HC down projection with every weight tile loaded ahead into registers and the activations read from L2
  (`gr_down_staged_kernel`'s sums, bitwise): no gain (T1 24.0 -> 23.7 us, slower at T 3-5). Removed.
- `fused_gr_bench` timed the plain HC read, not the engine's checked variant. `STRATA_BENCH_HC_CHECK=1` now runs
  `fused_gr_check` first. Staged read, cold weights: T1 51 us (norm 8.5, down 24, up 24.5), T2 62 us.
