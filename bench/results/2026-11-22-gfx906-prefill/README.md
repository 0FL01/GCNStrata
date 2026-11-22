# Production prefill baseline — amude, 2026-11-22

Use engine commit `dc790de9021ea26cbf49aed1dab116d5946e3bc4` for subsequent comparisons on four gfx906 cards. The deployed ELF SHA-256 is `a1328ef7946f43365ede095c6f99de02ae48c0b204a953f16cd701de254e6155`. Model: B3 Q4_0; config: `orcarouter-qwen38-b3-abliterated-q4_0-nvme-prod-pp3000-hs-20261121.json`, SHA-256 `f6d2eee14be9b33047b957641f5deb811fd1c98f9d9d752341c08e835c3e2ddb`. Keep `--prefill-pipe 3000`, ring 96, head split `0,1,2` and `--adapt-every 4` for speed measurements.

| Input | Old TTFT (`5d6ae32a`), s | Baseline TTFT, s | Baseline server prefill, tok/s |
|---|---:|---:|---:|
| 1K | 3.400 | 3.326 | 311 |
| 4K | 7.103 | 6.317 | 655 |
| 16K | 15.598 | 13.605 | 1187 |
| 60–64K | 39.733 | 36.134 | 1696 |

Medians of three unprofiled HTTP requests per length (`s0/s1/s0`, same inputs per binary, no prefix reuse, 16 generated tokens). TTFT is client time to the first nonempty text or reasoning SSE delta. No consistent append speedup was established. Correctness was checked separately: 26/26 complete static token-ID sequences matched. [Qualification, raw measurements and rollback](https://github.com/0FL01/llama-mi50-lain/blob/master/docs/goals/2026-11-22-prefill-e2e-gfx906.md).
