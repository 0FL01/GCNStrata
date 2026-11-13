## Current runtime: PR #1187 (2026-10-06 15:32 UTC)

Deployed commit: 2a7503d82f39ac445617d61a9fd82282ebaeef33 on deploy/pr1187-mode13.
Local production now sets STRATA_EXP_MODE=13 in compose.yaml. The upstream/default mode remains7.
Running native-process environment and executable hash were verified; API returned42 for17+25 and unauthenticated /v1/models returned401. Hybrid204800 and CPU FP16 vision configuration remain unchanged.
Binary SHA256: 4b5601b6e5f3019be382e0d2603941ca97f4479cbd50800bdc52ccdc307e5a5e.
Receipt and reversible backups: logs/deploy-pr1187-20261006/state.json and sibling files.
PR: https://github.com/Niko1221/Strata/pull/1187. Open, no merge conflicts at verification; no GitHub CI statuses or workflow runs were present. Local build/numerical/API checks are the evidence.
Older mode7 restoration reports below are historical.

Runtime updated2026-10-06 to0.1.40 at82f46a8 with minimalgfx906 build fixes; seeUPGRADE_0140.md.

# Active production: shefowl Hybrid

Updated2026-10-06. API0.0.0.0:8080, existing API key required, UI disabled. Target model Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0. Context capacity204800, INT8 KV/resident32768, PLE lockedRAM, MTP enabled. User selected this as primary model.

Config: configs/shefowl-hybrid-200k.json. compose.yaml and start.sh select Hybrid; restart unless-stopped. stop.sh removes autostart; logs.sh provides chronological logs. Coder files/config retained inactive.

Vision: CPU-only build-vision-cpu/bin/strata-vision, gpu:false,threads16,max_tokens1024; bartowski mmproj-Qwen3.8-Flash-Next-f16.gguf, revision928589fdb66c6ff07f22ac561e3fbce76553548f, SHA2569881c02403cd665a11b4d98dfc7bdd37e0c82b77c0c23e6b94797bb6036cfe85. Weights at /home/radneon/models/shefowl-qwen38-hybrid (SATA SSD).

Verification: standalone finite1024-token image embeddings, noGPUdevices; API red/blue image twice, text42, auth401/UI404, helper no GPUlibraries/fds. Evidence logs/shefowl-production-20261006/. Prior64Kthroughput logs/shefowl-hybrid-64k-20261006/REPORT.md. Full occupied200K and broad vision quality untested. FP16 performance advantage overBF16 not established.
