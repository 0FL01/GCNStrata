CURRENT2026-10-06: production is shefowl Hybrid204800 with CPU FP16 vision; see SHEFOWL_PRODUCTION.md. Prior Coder details below are historical.

# Current update: CPU vision enabled —2026-10-06

Strata is now RUNNING with CPU-only vision after user authorization. See
CPU_VISION.md and cpu-vision-verification.json. Earlier shutdown receipts below
refer to the original deployment test, not the current service state.

# MI50 Coder deployment
Source: Niko1221/Strata 6f32ec070f23ced9f50e704d854d775da52591ab (MIT)
Dependency: ggml-org/llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d.
Existing ROCm7.14 image: llama.cpp-gfx906:cmake-4.4.3.
Original IQ1_M shards are mounted read-only; derived runtime pack is on root disk.
Authentication uses the existing LLAMA_API_KEY from ~/llama/.env via Compose
interpolation into STRATA_API_KEY. Never run compose config without --quiet,
dump container environment, or copy the key into logs/source.
No separate frontend, browser launch, request monitor or MCP servers.
Local hardware sampling is disabled by STRATA_DISABLE_TELEMETRY=1.
Static UI routes return404 under STRATA_API_ONLY=1; documented deployment patch.
Upstream server may include static UI routes; API is the deployment interface.
start.sh starts service and enables Docker reboot restart. stop.sh removes the
service and its reboot policy. Docker itself is already enabled at boot.
logs.sh opens chronological logs at the bottom (less: PgUp/PgDn, g/G, q).
logs.sh --follow tails live. Build/pack/MTP logs have corresponding switches.
A server's configured128K limit does not prove an occupied128K request passed.
Status: preparation in progress; no successful inference claimed yet.

Build compatibility fix: wrap five template kernel pointers for HIP macro parsing and map the preferred shared-memory carveout enum. No kernel arithmetic change; STRATA_GR_V3 remains off. Patch copies in deploy/.

## Verified 2026-10-05
Build succeeds with the documented HIP compatibility patch. Five targeted GPU
CTests pass; 26 server/tokenizer/lifecycle tests pass (5 skipped).
The final Coder configuration loads all12,288 experts into VRAM with
STRATA_STAGE_TRIM=1 (upstream now requires opting into own-layer dense weights).
MTP4 runtime was prepared by upstream tools from31 tensors at the pinned
Qwen checkpoint and copied byte-for-byte via SCP; dense bounds and file sizes
validated. The .ready marker guards against an incomplete runtime transfer.
Model shards remained read-only.

Live API: correct arithmetic reply42; unauthorized models401; authenticated
models200; UI404; streaming Python code; primary-sampling5053-token prompt
with76 output tokens; disconnect then reuse; oversized request400.
The5053-token functional smoke reported PP372.6/TG41.8 tok/s.
This is one short smoke on Coder IQ1_M, not a paired benchmark or a comparison
against the Q4_0 engine. Context131072 is configured; occupied128K was not tested.
Stopped, restarted, tested generation/auth again, and stopped again.
Final stop confirmed2026-10-05T23:28:58Z: container absent, port8080 closed,
Docker autostart policy removed. No physical reboot was performed.
Docker itself is boot-enabled; the old llama-server.service is disabled.

Use ~/strata/start.sh to start and enable Docker boot restart.
Use ~/strata/stop.sh to stop and remove it from boot restart.
Use ~/strata/logs.sh (pager at bottom; PgUp/PgDn, g/G, q) or --follow.
Offline logs.sh reads logs/service-last.log. Full initial validation log is
logs/initial-live-server.log; structured receipts live in deploy/verification.json
and logs/*result.json. Build output is logs/build-retry2.log.
Existing authentication still comes from ~/llama/.env; keep that file.
The runtime uses python3 from the image PATH, not the image's system Python.
