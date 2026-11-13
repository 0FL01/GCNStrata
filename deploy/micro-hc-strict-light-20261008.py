from pathlib import Path
import subprocess,json,time,datetime,re,urllib.request
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/hc-strict-light';C=O/'micro';C.mkdir(exist_ok=False)
assert json.loads((O/'build.json').read_text())['exit_code']==0
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
rows=[];restore={}
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 for case in [0,4,5,6]:
  for audit in [0,1]:
   tag='case%d-audit%d'%(case,audit)
   cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-hc-strict-micro','--entrypoint','/src/logs/perf-campaign-20261008/hc-strict-light/gfx906_hc_guard_probe','-e','HIP_VISIBLE_DEVICES=0','-e','STRATA_GFX906_HC_F16=1','-e','STRATA_HC_F16_SCALE_PROBE=1','-e','STRATA_HC_F16_EXACT_INPUT=1','-e','STRATA_HC_F16_TRACE=1','-e','STRATA_HC_F16_AUDIT='+str(audit),'-e','STRATA_HC_GUARD_CASE='+str(case),'-v',str(R)+':/src',image]
   t=time.monotonic()
   with (C/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
   rows.append({'tag':tag,'exit_code':p.returncode,'elapsed_s':time.monotonic()-t,'argv':cmd});(C/'results.json').write_text(json.dumps(rows,indent=2))
   print(json.dumps(rows[-1]),flush=True)
   assert p.returncode==0,tag+' failed full-output native parity'
 (C/'complete.json').write_text(json.dumps({'passed':True,'cases':len(rows)},indent=2))
finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True,'health':h};break
  except (OSError,ValueError):pass
  time.sleep(5)
 (C/'restore.json').write_text(json.dumps(restore,indent=2));assert restore.get('passed')
print('STRICT MICRO COMPLETE',flush=True)
