# gfx906 four-GPU tuning sweep (2026-11-13)

Search for more prompt (PP) or generation (TG) speed on four Radeon Pro VII / MI50 (gfx906) cards with the
production profile of `../2026-11-13-gfx906-four-gpu/` (engine 0.1.45, commit `53e6c6a5`, four stages of 12
layers, `--pipeline-windows 2`, `--spec 6 --mtp-max-t 2 --spec-min-p 0.5`, `--prefill 4096`).

Workload: the saved 65,536-token prompt, 512 generated tokens, fixed seed, native protocol, one process per run.
`sweep.py` stops the production container once, runs one process per variant and restores production.
Every run below produced the same 512 output tokens.

**Result: nothing in this sweep is worth changing in production.** The engine switches measured here are within
run-to-run noise or slower, and the one real gain (a higher GPU clock) made the host unstable.

Run-to-run noise of the unchanged profile: PP 1142-1149 in three runs of this sweep at 1300 MHz (1098-1143 in the
earlier pair), TG 56-61 tok/s. Single runs that differ by less than about 5% say nothing.

## Engine switches, 1300 MHz / 925 mV

| Variant | PP tok/s | TG tok/s | drafts accepted/offered |
|---|---:|---:|---:|
| production profile | 1142, 1143, 1149 | 61.0, 60.8, 60.9 | 208/271 |
| same with the timing variables on | 1104 | 58.0 | 208/271 |
| `STRATA_MTP_STAGE0=1` (drafter on the first card) | 1081 | 56.1 | 208/271 |
| `--mtp-max-t 3` | 1131 | 55.0 | 275/405 |
| `--mtp-max-t 4` | 1137 | 51.6 | 306/501 |
| `--prefill 2048` | 1079 | 62.8 | 215/266 |
| `STRATA_SPLIT_MTP_BATCH=1` | 1061, 1138 | 57.3, 60.4 | 208/269 |
| `STRATA_SPLIT_MTP_BATCH=1 STRATA_KV_PREFETCH=1 STRATA_HANDOFF_PIPE=2` | 1168, 1145 | 60.4, 57.7 | 208/269 |
| `--pipeline-windows 0 --mtp-max-t 4` | 1146 | 45.3 | 312/483 |
| `--pipeline-windows 0 --mtp-max-t 6` | 1139 | 43.0 | 342/628 |
| `--pipeline-windows 0 --mtp-max-t 4`, `STRATA_SPLIT_OVERLAP=1` | 1133 | 46.7 | 312/483 |
| `--pipeline-windows 0 --mtp-max-t 6`, `STRATA_SPLIT_OVERLAP=1` | 1139 | 41.9 | 342/628 |
| `--pipeline-windows 0`, `STRATA_SPLIT_OVERLAP=1` | 1144 | 46.3 | 216/271 |

`--prefill 2048` changed the output tokens of that run (different chunk rounding); all others match.

## Where the time goes (profile run, 1300 MHz)

Decode, `STRATA_PIPELINE_DEBUG=1 STRATA_PIPELINE_LOG=1 STRATA_DECODE_TIMING=1`:

- 305 windows in 8086 ms, 1.68 tokens per window.
- 133 fresh windows at 36.35 ms (all four stages in turn), 172 speculative windows on the path at 18.90 ms
  (two stage times: the depth-2 pipeline's floor).
- 275 speculative windows were launched, 103 rolled back, 29 stayed below the gate. Of the scored chains with
  p_on >= 0.9, 104 of 107 held; below that about half.

So the pipelined windows already run at the floor of a two-deep pipeline; what limits TG is how often the
drafter's guess of the whole next window holds (56%), not a kernel or a hand-off.

Prompt, `STRATA_PREFILL_TIMING=1`, per stage and 4096-token chunk: GPU timeline about 2860 ms, wall about 3170 ms.
Largest parts: gemm gate/up 19%, hyper-connection read 12-17%, GDN 13%, QSA attention 10.5%, gemm down 10%.
Sixteen chunks over four stages take 19 chunk slots, which is the measured 1.77x over two cards; with
`sum(b + a C) + (S - 1) max(b + a C)` from docs/MULTI_GPU.md and b/a of several hundred tokens, 16 chunks is
already the best count for this prompt.

## GPU clock (not adopted)

All four cards, same table on each (they share one stock table), written only with no GPU client alive.

| Table | PP tok/s | TG tok/s | per card under load | outcome |
|---|---:|---:|---|---|
| 1300 MHz, 925 mV limit | 1142-1149 | 60.8-61.0 | about 125 W, 840 mV, 57 C | production |
| 1500 MHz, 1000 mV limit | 1202-1265 (7 runs) | 57.3-66.1 | about 160 W, 900-925 mV, 63 C | host lost on the 8th run |

At 1500 MHz, seven runs passed with identical output and no RAS counts. In the eighth, 35 s into the prompt, card
`0000:0b:00.0` stopped answering (root port `CmpltTO`, the engine's `HW Exception by GPU node-2 ... GPU Hang`), all
four root ports then reported `SDES` and every card was lost from the bus; the amdgpu driver faulted in
`amdgpu_device_halt_activities` during AER recovery and the host hung until it was power-cycled. Total GPU draw was
726-826 W in the last seconds, no temperature above 61 C. Whether the cause was the card's voltage margin at that
clock or the cards' supply is not known. The clock stays at 1300 MHz / 925 mV, now applied at boot.

A 1700 MHz / 1100 mV table was written once and the run was stopped by hand during model load; it was not measured.

## Upstream (Niko1221/Strata 0.1.59.1, 3881 commits past the merge-base)

Read for changes that could help this setup; nothing was ported.

- Already in this fork: pipelined windows on three and more stages, the drafter on the first card,
  `STRATA_SPLIT_OVERLAP`, `STRATA_HANDOFF_PIPE`, `--prefill-pipe`, `STRATA_KV_PREFETCH`.
- Not usable here: the all-cards split engine (wave32 / WMMA kernels), multi-CTA long-context selection
  (`!__HIPCC__`), routed-expert prompt streaming (one GPU), DFlash2 (not with pipelined windows), longer lookup
  windows (not on a layer split). `--pipeline-windows` is still clamped to 2.
- `aec327ee` (hyper-connection read, six launches to four) applies cleanly but is a no-op here: these cards take
  the staged read, not `gr_read`.

## Not tested

Prompts other than the one 64K fixture, short prompts, sampling settings other than the fixture's, a pipeline
deeper than two windows, 1400 MHz, long soak.

## Files

- `sweep.py` - the runner (`sweep.py name[:ENV=V,--flag=V,...] ...`; `TABLE=<pp_table>` writes that table to all
  four cards first). It depends on the controller modules under `deploy/` on the host.
- `results.json` - every run: variant, settings, clock table, PP, TG, draft counts, output hash.
