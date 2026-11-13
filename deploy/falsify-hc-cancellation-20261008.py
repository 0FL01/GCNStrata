from pathlib import Path
import subprocess,json,hashlib,shutil,time,urllib.request
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261008/hc-cancellation'
O.mkdir(exist_ok=False)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-hc-cancellation-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake --build build-906 --target gfx906_hc_guard_probe -j4']
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(O/'build.json').write_text(json.dumps({'argv':cmd,'exit_code':p.returncode},indent=2));assert p.returncode==0
shutil.copy2(W/'build-906/gfx906_hc_guard_probe',O/'gfx906_hc_guard_probe')
shutil.copy2(W/'bench/gfx906_hc_guard_probe.cpp',O/'gfx906_hc_guard_probe.cpp')
rows=[];restore={}
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 for dev in [0,1]:
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-hc-cancellation','--entrypoint','/src/logs/perf-campaign-20261008/hc-cancellation/gfx906_hc_guard_probe','-e','HIP_VISIBLE_DEVICES='+str(dev),'-e','STRATA_GFX906_HC_F16=1','-e','STRATA_HC_F16_SCALE_PROBE=1','-e','STRATA_HC_F16_EXACT_INPUT=1','-e','STRATA_HC_F16_TRACE=1','-e','STRATA_HC_F16_AUDIT=0','-e','STRATA_HC_F16_SHADOW=0','-e','STRATA_HC_F16_ARENA_ONLY=0','-e','STRATA_HC_GUARD_CASE=7','-v',str(R)+':/src',image]
  log=O/('gpu%d.log'%dev)
  with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  row={'gpu':dev,'exit_code':p.returncode,'argv':cmd,'log':log.read_text()};rows.append(row);print(json.dumps(row),flush=True)
finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True,'health':h};break
  except (OSError,ValueError):pass
  time.sleep(5)
 (O/'summary.json').write_text(json.dumps({'results':rows,'production_restore':restore,'binary_sha256':hashlib.sha256((O/'gfx906_hc_guard_probe').read_bytes()).hexdigest()},indent=2))
 assert restore.get('passed')
print('HC CANCELLATION FALSIFIER COMPLETE',flush=True)
