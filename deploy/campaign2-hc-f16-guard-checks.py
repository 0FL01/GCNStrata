from pathlib import Path
import subprocess,json,re,datetime
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261007/hc-f16-guard-checks'
assert not subprocess.check_output(['docker','ps','-q']).strip()
rows=[]
for gpu in [0,1]:
 for case in [0,1,2,3]:
  tag=f'g{gpu}-c{case}'
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/gfx906_hc_guard_probe','-v',str(W)+':/src','-e',f'HIP_VISIBLE_DEVICES={gpu}','-e','STRATA_GFX906_HC_F16=1','-e','STRATA_HC_F16_SCALE_PROBE=1','-e','STRATA_HC_F16_AUDIT=0','-e','STRATA_HC_F16_TRACE=1','-e',f'STRATA_HC_GUARD_CASE={case}','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646']
  with (O/(tag+'.log')).open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  text=(O/(tag+'.log')).read_text();parity=[x for x in text.splitlines() if x.startswith('NATIVE_PARITY')]
  routes=re.findall(r'attempts=(\d+) taken=(\d+)',text)
  assert p.returncode==0,(tag,p.returncode)
  assert len(parity)==4 and len(routes)==4,(tag,parity,routes)
  assert all(int(t)==(0 if case in [1,2] else 3) for a,t in routes),(tag,routes)
  if case in [1,2,3]:assert all('bitwise=1' in x for x in parity),tag
  rows.append({'tag':tag,'exit':p.returncode,'routes':routes,'parity':parity})
  print(json.dumps(rows[-1]),flush=True)
  (O/'results.json').write_text(json.dumps(rows,indent=2))
(O/'complete.json').write_text(json.dumps({'passed':True,'cases':len(rows),'shapes':len(rows)*4,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
