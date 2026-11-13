# gfx906 Q2_0 gate/up: stable candidate (mode 13)

Base: Strata 0.1.40, 82f46a8c8f475f001ad76d92f58f4a4f8ffb0253. Upstream main was still this revision at 2026-10-06 14:27 UTC.
Hardware: two gfx906 16 GiB GPUs, Xeon E5-2698B v3 (AVX2), 128 GiB RAM.
Model: shefowl Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0, revision 55568c1b41d2e381a59a281447fb10935c3c6d0a.

## Change

Q2_0 gate/up uses exact integer expansion of packed two-bit codes, a token-inner loop for up to four entries, global activation reads and 16-row blocks. This removes LDS copies/barriers and restores the baseline block granularity for singleton groups. The candidate is restricted to gfx906, Q2_0, H=2560 and FF=640. Down projection, weights, KV precision and sampling are unchanged.

The earlier LDS modes 9–12 and screening modes 14/15 are not in the proposed patch. Their sources and binaries remain archived. Mode 13 is opt-in with STRATA_EXP_MODE=13; default mode 7 remains unchanged.

## Performance evidence

ABBA means baseline 7, candidate 13, candidate 13, baseline 7. Each run starts a fresh engine, uses the same pinned container and profile, no prompt reuse, 1024 output tokens, 204800 context capacity, INT8 KV / 32768 resident KV, PLE locked in RAM, MTP4 plus suffix lookup3. Vision is disabled only for the text benchmark. Temperature1/top_p0.95/top_k20; code seed12345 and Russian seed54321. EOS stopping is disabled to fix the output count. Load time is excluded. Clocks were not locked.

| Workload | Baseline TG | Candidate TG | Change | Equal outputs |
|---|---:|---:|---:|---|
| code4k | 58.106049 | 58.964267 | +1.477% | True |
| code64k | 46.438751 | 47.101683 | +1.428% | True |
| ru64k | 31.469203 | 31.860610 | +1.244% | False |

Both candidate repetitions beat both baseline repetitions in each workload. The two code workloads qualify as equal-output comparisons; PP is essentially unchanged (about340 tok/s at4K and584 tok/s at64K).

The Russian row is observational only: baseline A1/A2 first diverged at output index213; A1/candidate at386; both candidate repetitions were identical. Different generated sequences and draft acceptance counters prevent treating this row as a controlled equal-work gain. The cause of baseline nondeterminism has not been isolated. No broad-quality or universal-speedup claim is made.

## Numerical and regression checks

- 280 real-weight component cases on both GPUs: layers0/23/47, G1/2/8/32/64, T1..8, plus mixed group lengths and padded launch grids. Every final output was bitwise equal. Every measured component case improved; singleton whole-expert speedup1.0310–1.0496x, median1.0381x.
- Stronger intermediate parity:282 cases, including all48 layers on both GPUs and two seeds. 70,744,320 gate/up/SwiGLU float values,26,529,120 quantized activation bytes and94,325,760 output float values compared exactly, with zero differences.
- The integer unpacking was exhaustively compared for all65,536 packed16-bit words in the earlier fast-decode slice; it maps each code to {-1,0,1,2}. No precision change.
- Cleaned candidate rebuilt successfully. A fresh 4K baseline/candidate pair reproduced all1024 IDs from the original code4K series. A further12 cleaned-source component cases passed on both GPUs.
- Minimal patch applies against the pinned upstream index (git apply --cached --check); git diff --check passes. Existing gfx906 build compatibility patches are unchanged and excluded from this optimization patch.
- Capacity204800 is configured; this does not qualify a fully occupied200K context. Other GPU architectures and model geometries were not tested.

## Artifacts and rollback

- Proposed source/benchmark patch: deploy/q2-gu-global-minimal.patch
- Full measurements, manifests and synthetic fixtures: logs/q2-gu-stable-20261006/
- Cleaned-build checks: logs/q2-gu-clean-20261006/
- Drivers: deploy/bench-q2-stable.py, deploy/bench-q2-clean.py, deploy/prepare-q2-stable.py
- Original experimental mode13 executable: logs/q2-gu-stable-20261006/strata-mode13-experimental
- Cleaned executable SHA256: 4b5601b6e5f3019be382e0d2603941ca97f4479cbd50800bdc52ccdc307e5a5e

Production returns to mode7 after tests. No upstream push or PR publication has been performed. For a PR, retain the Russian qualification caveat and avoid claiming universal parity or speedup.

## Final verification (2026-10-06 14:35 UTC)

The benchmark inspection is restricted to Q2_0 so fused/non-Q2 modes retain their previous behavior. Expert counts are read from GGUF dimensions instead of being hard-coded to256 or512. The final benchmark rebuilt and passed three additional Q2_0 cases and one256-expert Coder reference case. The production health check reports loaded=true, images=true, api_key=true and context204800. Only strata-api is running, mode7/default; production configuration is unchanged.

Patch SHA256: 1c25d5e2fc2dab929046b3070f34bcfa6d296df5344eaac7b57a204b1e40d130

User accepted the modest regression-free gain. Draft PR publication was proposed at14:35 UTC and is awaiting a reply; no external PR or push occurred.

## Current runtime: PR #1187 (2026-10-06 15:32 UTC)

Deployed commit: 2a7503d82f39ac445617d61a9fd82282ebaeef33 on deploy/pr1187-mode13.
Local production now sets STRATA_EXP_MODE=13 in compose.yaml. The upstream/default mode remains7.
Running native-process environment and executable hash were verified; API returned42 for17+25 and unauthenticated /v1/models returned401. Hybrid204800 and CPU FP16 vision configuration remain unchanged.
Binary SHA256: 4b5601b6e5f3019be382e0d2603941ca97f4479cbd50800bdc52ccdc307e5a5e.
Receipt and reversible backups: logs/deploy-pr1187-20261006/state.json and sibling files.
PR: https://github.com/Niko1221/Strata/pull/1187. Open, no merge conflicts at verification; no GitHub CI statuses or workflow runs were present. Local build/numerical/API checks are the evidence.
Older mode7 restoration reports below are historical.

