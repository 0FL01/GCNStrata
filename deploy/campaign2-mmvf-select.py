from pathlib import Path
import subprocess,json,time
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007'
assert (O/'mmvf-rpb/complete.json').exists()
subprocess.run(['docker','stop','--timeout','60','strata-api'],check=True)
assert not subprocess.check_output(['docker','ps','-q']).strip()
subprocess.run(['git','apply','--check',str(O/'mmvf-rpb-select.patch')],cwd=R,check=True)
subprocess.run(['git','apply',str(O/'mmvf-rpb-select.patch')],cwd=R,check=True)
P=O/'mmvf-rpb-select';P.mkdir()
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--entrypoint','/bin/sh','-v',str(R)+':/src','-w','/src',image,'-c','cmake --build build-906 --target mmvf_rows_parity -j4']
with (P/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
results=[]
for gpu in [0,1]:
 for ts in [0,1]:
  for rows in [4,2,8,1,1,8,2,4] if ts==0 else [1]:
   tag=f'gpu{gpu}-ts{ts}-rows{rows}-run{len(results)}'
   cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--entrypoint','/src/build-906/mmvf_rows_parity','-v',str(R)+':/src','-e','HIP_VISIBLE_DEVICES='+str(gpu),'-e','STRATA_MMVF_ROWS=1','-e','STRATA_MMVF_RPB='+str(rows),'-e','STRATA_TSUM='+str(ts),image,'--bench']
   with (P/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
   results.append({'tag':tag,'gpu':gpu,'tsum':ts,'rows':rows,'exit':r.returncode})
   (P/'results.json').write_text(json.dumps(results,indent=2));assert r.returncode==0,tag
(P/'complete.json').write_text(json.dumps({'passed':True,'runs':len(results)}))
print('MMVF row screen complete',flush=True)
