from pathlib import Path
import json
r=Path('/home/radneon/strata')
p=r/'compose.yaml';s=p.read_text().replace('image: llama.cpp-gfx906:cmake-4.4.3','image: sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646')
s=s.replace('      STRATA_ARENA_MMAP: "1"','      STRATA_ARENA_MMAP: "1"\n      STRATA_API_ONLY: "1"\n      STRATA_DISABLE_TELEMETRY: "1"')
s=s.replace('    restart: unless-stopped','    healthcheck:\n      test: ["CMD", "python3", "/src/deploy/healthcheck.py"]\n      interval: 30s\n      timeout: 5s\n      retries: 3\n      start_period: 300s\n    restart: unless-stopped')
p.write_text(s)
(r/'deploy/healthcheck.py').write_text("import json,urllib.request\nr=json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4))\nassert r['status']=='ok' and r['loaded'] and r['max_context']>0 and r['api_key']\n")
p=r/'logs.sh';s=p.read_text().replace('--build) file=logs/build.log','--build) file=logs/build-retry2.log');p.write_text(s)
p=r/'deploy/DEPLOYMENT.md';s=p.read_text().replace('Hardware metrics are local to the service; no outbound telemetry has been enabled.','Local hardware sampling is disabled by STRATA_DISABLE_TELEMETRY=1.\nStatic UI routes return404 under STRATA_API_ONLY=1; documented deployment patch.')
s += "\nBuild compatibility fix: wrap five template kernel pointers for HIP macro parsing and map the preferred shared-memory carveout enum. No kernel arithmetic change; STRATA_GR_V3 remains off. Patch copies in deploy/.\n"
p.write_text(s)
print('Pinned image, API-only/privacy switches and readiness healthcheck configured')

