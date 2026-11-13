from pathlib import Path
import subprocess,json,time,hashlib,shutil
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007';W=Path('/home/radneon/strata-rnd-sgemm')
for _ in range(720):
 p=O/'help-screen/summary.json'
 if p.exists():
  j=json.loads(p.read_text());assert len(j['results'])==8 and all(x['passed'] for x in j['results']) and j['production_restore']['passed'];break
 time.sleep(5)
else:raise RuntimeError('helper completion receipt absent')
subprocess.run(['docker','stop','--timeout','60','strata-api'],check=True)
assert not subprocess.check_output(['docker','ps','-q']).strip()
subprocess.run(['git','apply','--check',str(O/'hc-selective.patch')],cwd=W,check=True)
subprocess.run(['git','apply',str(O/'hc-selective.patch')],cwd=W,check=True)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-hc-sgemm-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake --build build-906 --target gfx906_gemm_screen gfx906_sgemm_parity strata -j4']
with (O/'hc-sgemm-build.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(O/'hc-sgemm-build-state.json').write_text(json.dumps({'exit':r.returncode}));assert r.returncode==0
P=O/'hc-sgemm-checks';P.mkdir()
results=[]
for gpu in [0,1]:
 for arm in [0,1]:
  for target in ['gfx906_sgemm_parity','gfx906_gemm_screen']:
   tag='g'+str(gpu)+'-arm'+str(arm)+'-'+target
   cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/'+target,'-v',str(W)+':/src','-e','STRATA_RDNA2_SGEMM=0','-e','STRATA_GFX906_HC_SGEMM='+str(arm),'-e','HIP_VISIBLE_DEVICES='+str(gpu),image]
   with (P/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
   results.append({'tag':tag,'exit':r.returncode});(P/'results.json').write_text(json.dumps(results,indent=2));assert r.returncode==0,tag
shutil.copy2(W/'build-906/strata',O/'strata-hc-sgemm')
(P/'complete.json').write_text(json.dumps({'passed':True,'binary_sha256':hashlib.sha256((O/'strata-hc-sgemm').read_bytes()).hexdigest()}))
print('selected HC SGEMM checks completed',flush=True)
