from pathlib import Path
import subprocess,json,hashlib,re
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261007/hc-f16-route-checks'
O.mkdir(exist_ok=False)
receipt=json.loads((R/'logs/perf-campaign-20261007/hc-f16-build/complete.json').read_text())
assert hashlib.sha256((W/'build-906/strata').read_bytes()).hexdigest()==receipt['binary_sha256']
assert not subprocess.check_output(['docker','ps','-q']).strip()
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
rows=[]
for gpu in [0,1]:
 for flag in [0,1]:
  tag=f'g{gpu}-flag{flag}'
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/gfx906_hc_f16_probe','-v',str(W)+':/src','-e',f'HIP_VISIBLE_DEVICES={gpu}','-e',f'STRATA_GFX906_HC_F16={flag}','-e','STRATA_HC_F16_TRACE=1','-e','STRATA_HIP_PROMPT_F16=0',image]
  pth=O/(tag+'.log')
  with pth.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  assert p.returncode==0,tag
  text=pth.read_text()
  routes=re.findall(r'attempts=([0-9]+) taken=([0-9]+)',text)
  if flag:assert len(routes)==4 and all(int(a)==int(b)>0 for a,b in routes),routes
  else:assert not routes,routes
  rows.append({'tag':tag,'passed':True,'routes':routes});(O/'results.json').write_text(json.dumps(rows,indent=2))
(O/'complete.json').write_text(json.dumps({'passed':True,'binary_sha256':receipt['binary_sha256'],'cases':len(rows)},indent=2))
print('Isolated HC FP16 route component checks complete',flush=True)
