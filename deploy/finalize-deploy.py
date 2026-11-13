from pathlib import Path
import json,datetime,subprocess
r=Path('/home/radneon/strata');life=json.loads((r/'logs/lifecycle-verification.json').read_text())
assert life['final_off'] and len(life['events'])==4
a=json.loads((r/'logs/deployment-api-result.json').read_text());b=json.loads((r/'logs/live-extended-result.json').read_text());c=json.loads((r/'logs/restart-api-result.json').read_text())
assert a['passed'] and b['passed'] and c['passed']
long=[x for x in b['results'] if x['kind']=='nonstream' and x['response']['usage']['prompt_tokens']>4000][0]
summary={'source_revision':'6f32ec070f23ced9f50e704d854d775da52591ab','local_patches':['gfx906-template-macro-fix.patch','api-only-no-telemetry.patch'],'passed':True,'configured_context':131072,'largest_tested_prompt_tokens':long['response']['usage']['prompt_tokens'],'long_smoke_response':long['response'],'long_smoke_elapsed_s':long['elapsed_s'],'targeted_gpu_ctest':{'passed':5,'failed':0},'python_unittest':{'run':26,'skipped':5,'failed':0},'scope':'deployment and short/5053-token functional smoke, not independent model parity or occupied128K qualification','final_off':True,'lifecycle':life}
(r/'deploy/verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
p=r/'deploy/DEPLOYMENT.md'
with p.open('a') as f:f.write("""
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
""")
print(json.dumps({'deployment_verified':True,'final_off':True,'largest_tested_prompt':summary['largest_tested_prompt_tokens']}))

