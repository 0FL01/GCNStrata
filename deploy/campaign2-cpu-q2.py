from pathlib import Path
import subprocess,json,time
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007'
for _ in range(720):
 p=O/'hc-arena-64k/summary.json'
 if p.exists():
  j=json.loads(p.read_text());assert len(j['results'])==4 and all(x['passed'] for x in j['results']) and j['production_restore']['passed'];break
 time.sleep(5)
else:raise RuntimeError('arena64K series incomplete')
subprocess.run(['docker','stop','--timeout','60','strata-api'],check=True)
assert not subprocess.check_output(['docker','ps','-q']).strip()
subprocess.run(['git','apply','--check',str(O/'cpu-q2-spread.patch')],cwd=R,check=True)
subprocess.run(['git','apply',str(O/'cpu-q2-spread.patch')],cwd=R,check=True)
P=O/'cpu-q2-spread';P.mkdir()
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--entrypoint','/bin/sh','-v',str(R)+':/src','-w','/src',image,'-c','cmake -S . -B build-906 && cmake --build build-906 --target q2_unpack_spread_check iq_avx2_parity -j4']
with (P/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
results=[]
for tag,target,flag,args in [('exact','q2_unpack_spread_check',0,[]),('existing-off','iq_avx2_parity',0,[]),('existing-on','iq_avx2_parity',1,[]),('bench','q2_unpack_spread_check',0,['--bench'])]:
 cmd=['docker','run','--rm','--network','none','--cpuset-cpus','1','--entrypoint','/src/build-906/'+target,'-v',str(R)+':/src','-e','STRATA_Q2_AVX2_SPREAD='+str(flag),image]+args
 with (P/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 results.append({'tag':tag,'exit':r.returncode});(P/'results.json').write_text(json.dumps(results,indent=2));assert r.returncode==0,tag
(P/'complete.json').write_text(json.dumps({'passed':True}))
print('CPU Q2 exact-unpack screen complete',flush=True)
