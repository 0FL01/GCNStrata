# gfx906: four-GPU layer split, 64k prompt (2026-11-13)

Four Radeon Pro VII / MI50 (gfx906) cards, one conversation, the saved
65536-token prompt and 512 generated tokens. Binary built from `53e6c6a5`
(engine 0.1.45, SHA-256 in `results.json`), unchanged between arms. Fresh
engine per run, no prompt reuse, seed 12345, temperature 1, top_p 0.95,
top_k 20. Two runs per arm.

## Result

| Arm | PP tok/s | TG tok/s | MTP accepted/offered | Runs identical |
|---|---:|---:|---:|---|
| 2 GPUs, split 27, `--pipeline-windows 2` | 644.13 / 644.66 | 56.07 / 53.64 | 219/271 | yes |
| 4 GPUs, split auto, serial, unequal clocks | 1181.79 / 1170.06 | 52.05 / 48.04 | 241/260, 211/263 | **no** |
| 4 GPUs, split auto, serial, equal tables | 1148.10 / 1134.22 | 45.16 / 44.16 | 216/271 | yes |
| 4 GPUs, split auto, `--pipeline-windows 2`, equal tables | 1143.33 / 1098.45 | 60.28 / 56.13 | 208/271 | yes |

Median of the last arm against two GPUs: PP +74%, TG +6%. Its output is
also identical to the serial four-GPU arm. `--layer-split auto` chose
12,24,36 (twelve layers per card); all 24576 profiled expert pairs resident.

## What was wrong

The non-repeatable four-GPU output was not an engine defect. Cards 3 and 4
ran gfxclk levels up to 1700 MHz at a 925/950 mV ceiling and miscomputed;
cards 1 and 2 were capped at 1300 MHz at 925 mV. The same two cards alone
had already produced `DONE nonfinite` and divergence from output index 20.
With `STRATA_VERIFY_ALL_RESIDENT=0` the four-GPU run ended `DONE nonfinite`
at the first decode window (`results.json`, `resident-off-nonfinite`).
Writing the first pair's exact `pp_table` to the second pair made the output
repeatable.

The pipeline runs on four stages in this binary (`two verifiers per stage,
4 stages overlap across windows`). Forcing `--pipeline-windows 0` for more
than two stages costs about a quarter of decode speed here.

## Deployment

`CONFIG.production.json` is the deployed config: `gpu` 0-3, `layer_split`
auto, `--pipeline-windows 2`, `HIP_VISIBLE_DEVICES=0,1,2,3`. The service
came up healthy and answered an authenticated chat request.

## Limits

- One workload, two runs per arm, fan at 100% during measurement.
- The equal power tables do not survive a reboot. Four-GPU operation at
  stock tables (1700 MHz, 1162.5 mV) was not measured.
- Benchmark arms ran without `--vision`, prompt cache and conversation cache.
- `compare.py` is the host-specific helper that produced the runs; it
  depends on local controller scripts and is kept as a record, not a tool.
