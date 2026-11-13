from pathlib import Path
import subprocess,json,time
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007'
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
for _ in range(180):
 p=O/'profile/summary.json'
 if p.exists():
  data=json.loads(p.read_text());assert len(data['results'])==2 and all(x['passed'] for x in data['results']);assert data['production_restore']['passed'];break
 time.sleep(5)
else:raise RuntimeError('profiling receipt absent; no other workload started')
subprocess.run(['docker','stop','--timeout','60','strata-api'],check=True)
assert not subprocess.check_output(['docker','ps','-q']).strip()
records=[]
for gpu in [0,1]:
 for name in ['mmvf_rows_parity','gr_multi_parity','native_grouped_parity']:
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/'+name,'-v',str(R)+':/src','-e','HIP_VISIBLE_DEVICES='+str(gpu),'-e','STRATA_MMVF_ROWS=1','-e','STRATA_TSUM=1','-e','STRATA_EXP_MODE=15','-e','STRATA_HIP_SWIGLU_FUSED=1',image]
  with (O/('tsum-'+name+'-gpu'+str(gpu)+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  records.append({'gpu':gpu,'test':name,'exit':r.returncode})
(O/'tsum-parity.json').write_text(json.dumps(records,indent=2))
subprocess.run(['git','apply','--check',str(O/'mode16.patch')],cwd=R,check=True)
subprocess.run(['git','apply',str(O/'mode16.patch')],cwd=R,check=True)
cmd=['docker','run','--rm','--network','none','--name','strata-campaign2-build','--entrypoint','/bin/sh','-v',str(R)+':/src','-w','/src',image,'-c','cmake --build build-906 --target strata native_expert_bench native_grouped_parity mmvf_rows_parity -j4']
with (O/'mode16-build.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(O/'build-state.json').write_text(json.dumps({'exit':r.returncode,'tsum_tests':records},indent=2))
assert r.returncode==0
print('mode16 build finished',flush=True)
