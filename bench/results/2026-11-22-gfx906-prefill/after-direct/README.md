# B3 prefill: qualified production release 453e60db

Four gfx906 cards, same B3 Q4_0 weights, split 10/23/36, INT8 KV, adapt4, `--prefill-pipe 3000`, ring 96, DPM high. Relative to the dc790de9 configuration, only `STRATA_MMQ_DIRECT=1` and `STRATA_MMQ_RING_DIRECT=1` were added. Cache slots remained 4172/5113/5087/4625. Qualified ELF SHA-256: `b73ce7bc37ad6571f2d0c2f1cbc1c6e4d14cb518dcf766dcbfc72ed75720de97`.

| Input | Reference TTFT, s | Release TTFT, s | Reference / release prompt rate, tok/s |
|---|---:|---:|---:|
| ~1K | 3.331 | 3.285 | 312 / 314 |
| ~4K | 6.319 | 5.679 | 654 / 729 |
| ~16K | 13.582 | 12.728 | 1189 / 1270 |
| 60–64K | 36.130 | 34.888 | 1695 / 1756 |
| Append ~4K after 60–64K | 8.016 | 7.405 | 528 / 574 |

Medians of three matching requests per row. Uninstrumented HTTP SSE, first nonempty reasoning or text delta, 16 generated tokens. `Bend.jsonl` is the reloaded dc790de9 control at the end of the window; `ShRG.jsonl` is the final release. The four cold series have zero reuse and two distinct texts (s0/s1/s0). For append, combine the `append` row in each JSONL with both `append` rows in the matching `.extra` file: 3,886–4,042 fresh and 60,574–64,419 reused tokens, identical between arms. First request after restart is **not** a warm latency guarantee. On the deployed endpoint the first 4K took 6.407 s; warmed 4K/16K/60K on s0 took 5.694/12.737/34.888 s. No loaded-concurrency or other-model result is claimed.

Stages qualified separately: routed pinned uploads (`6acdf7e3`), half HC gate read (`e6a24e52`), shadow rows for local Q4 (`72a7148a`), direct ring groups (`61d4af95`), wider GDN recurrence (`453e60db`). All five passed 26/26 complete static greedy token sequences. Rejected: padded slots (different decode placement, 19/26), bounded weight loader in the hot loop (exact but slower), reduced lookahead (less H2D but worse TTFT), ring 128 (regressed 1K), and compact MMQ grid (both 4K and 16K slower). The final 60K native trace is 96.804 aggregate GPU-stage seconds, **not** client latency; gate/up plus down 26.783, QSA attention 11.998, HC 9.680, GDN recurrence 3.944. These intervals include stream waits; do not infer a sum of kernel speedups.
