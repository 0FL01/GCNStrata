# gfx906 Inkling-2-Mini V8 qualification — November 27, 2026

This is a compact summary of the measured V8 release series, not a new independent rerun.
It records the final qualified configuration and the scope of the measurements.

## Hardware and configuration

- Four gfx906 GPUs, 16 GiB each, Linux, community ROCm/HIP 7.14; Xeon E5-2698B v3, 121 GiB usable RAM.
- Inkling-2-Mini Q4_0 target (bartowski), 15,064,977,088-byte GGUF, SHA-256
  `e8a583198523fe1119120061ed68fc3ce14dd1052ef53815014c867b307b9e37`.
- Q8_0 KV, configured context capacity 262,144; native Q4_0-mixed DFlash2 with F16 selector books;
  TP2 target on GPU0/1, co-resident DualTP prefill Stage B on GPU2/3.
- Single-stream inference. Four-GPU target head, context-dependent KV helpers, adaptive draft widths,
  pinned checkpoints, asynchronous mirror seeding and suffix-aware restore.
- Final engine `bd13ef75708dfa70cf27561d37ce682583db0870`; qualified ELF SHA-256
  `63c079c4b92afa476aadd09cc066abfefcd36e31e2028d57cf2804a26135c8da`.

## End-to-end measurements

The `agent5` fixture contains nine turns (warm, cold, three tool returns, append, detour, restore,
final tool return), 25,081 fresh prompt tokens and 1,553 generated tokens. Generation uses
temperature 1.0, top-k 0, top-p 1.0, min-p 0.0.

The full nine-turn totals below include the short warm turn. Do not compare them directly with
reports that omit it.

| Release / window | Full PP, s | Full TG, s | Client wall, s |
| --- | ---: | ---: | ---: |
| V7-R4 `023f973c`, W6b historical | 16.557 | 17.848 | 34.750 |
| V7-R4 `023f973c`, V8 W4 control | 16.524 | 17.980 | 34.852 |
| V8-R1 `c623a9a5`, W4 | 16.528 | 17.228 | 34.092 |
| V8-R2 `8baa23fb`, W9 control | 15.740 | 17.232 | 33.322 |
| V8-R3 `bd13ef75`, W9 | 15.707 | 17.232 | 33.297 |

W4 and W9 are paired A/B runs with two repetitions per arm. Cross-window comparisons are
historical, not a single controlled experiment. V8-R3's full AgentOps wall time is effectively
unchanged from the V8-R2 W9 control; the last release primarily improves restore and filtered sampling.

The final W9 `serve6` screen matched all eight output fingerprints. Its cold code phase processed
19,127 new tokens in 9.054 s. Its long phase processed 49,152 new tokens after reusing 16,384,
in 33.772 s. A fully cold 65,536-token result is not implied.

## W9 filtered API tests

Each decode case produced 128 output tokens after a 2K prompt.

| Sampling | V8-R2, tok/s | V8-R3, tok/s | Output SHA prefix |
| --- | ---: | ---: | --- |
| top-k 0, top-p 0.95 | 22.7 | 46.3 | c1cb175966567506 |
| top-k 0, min-p 0.05 | 29.4 | 133.4 | 0ff49126d138467f |
| top-k 20, top-p 0.95 | 112.7 | 120.5 | c64aecc6ea4d34e9 |

Each pair matches its own output fingerprint. The cases have distinct trajectories and draft
acceptance; their rates should not be compared as if they were the same generation.

Restore engine prompt time on the 23,283-token reused history fell from 93.5 to 42.3 ms
in W9; corresponding client TTFT was 165.3 to 112.9 ms.

## Changes and validation

- [V8-R1](https://github.com/0FL01/GCNStrata/commit/c623a9a5): four-GPU target head, lineage-tagged KV
  mirrors, async keep and W8 attention helpers. W4 matched eight serving phases and nine AgentOps turns.
- [V8-R2](https://github.com/0FL01/GCNStrata/commit/8baa23fb): exact fused stage controls/mix, improved
  DualTP scheduling and asynchronous draft context. Scoped FP contraction control fixes the initial mix
  divergence; the 1920-token wave-alignment experiment was rejected because the MoE path regressed.
- [V8-R3](https://github.com/0FL01/GCNStrata/commit/bd13ef75): exact filtered sampler, canonical suffix
  restore, async mirror seed and tiled router. W9 matched 48/48 checked output fingerprints
  (8 serving phases, 4 filtered cases including warmup, 9 turns in each of 4 AgentOps runs).
  Host sampler tests passed 16,022 token/RNG/probability checks, and suffix tests passed 10/10.
  Output fingerprints alone do not establish bitwise equivalence for all possible inputs.

Single-visit canonical distributed MoE was rejected at small verifier widths: the PCIe return
traffic outweighed the extra GPU compute. Noncanonical partial sums are not selected in the
qualified target configuration.

Raw profiler captures and complete execution receipts remain in the associated control-plane
research archive (`artifacts/distributed-v8-20261126/`). This summary is a measurement record,
not a self-contained benchmark runner or a qualification of other GPU counts, models, drivers,
concurrent streams or every possible context length. No new runtime changes are introduced by this document.
