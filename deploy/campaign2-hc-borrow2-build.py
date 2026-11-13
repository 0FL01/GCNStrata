from pathlib import Path
import subprocess,json,time,hashlib,shutil
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007';W=Path('/home/radneon/strata-rnd-sgemm')
assert json.loads((O/'hc-borrow-checks/complete.json').read_text())['passed']
subprocess.run(['docker','stop','--timeout','60','strata-api'],check=True)
assert not subprocess.check_output(['docker','ps','-q']).strip()
subprocess.run(['git','apply','-R','--check',str(O/'hc-borrow.patch')],cwd=W,check=True)
subprocess.run(['git','apply','-R',str(O/'hc-borrow.patch')],cwd=W,check=True)
subprocess.run(['git','apply','--check',str(O/'hc-borrow2.patch')],cwd=W,check=True)
subprocess.run(['git','apply',str(O/'hc-borrow2.patch')],cwd=W,check=True)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake --build build-906 --target strata gfx906_sgemm_parity gfx906_gemm_screen -j4']
with (O/'hc-borrow2-build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
P=O/'hc-borrow2-checks';P.mkdir()
results=[]
for gpu in [0,1]:
 for label,hc,borrow in [('native',0,0),('owned',1,0),('arena',1,2)]:
  for target in ['gfx906_sgemm_parity','gfx906_gemm_screen']:
   tag='g'+str(gpu)+'-'+label+'-'+target
   cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/'+target,'-v',str(W)+':/src','-e','STRATA_RDNA2_SGEMM=0','-e','STRATA_GFX906_HC_SGEMM='+str(hc),'-e','STRATA_GFX906_HC_SGEMM_BORROW='+str(borrow),'-e','STRATA_HC_SGEMM_TRACE=1','-e','HIP_VISIBLE_DEVICES='+str(gpu),image]
   with (P/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
   results.append({'tag':tag,'exit':r.returncode});(P/'results.json').write_text(json.dumps(results,indent=2));assert r.returncode==0,tag
shutil.copy2(W/'build-906/strata',O/'strata-hc-arena')
(P/'complete.json').write_text(json.dumps({'passed':True,'binary_sha256':hashlib.sha256((O/'strata-hc-arena').read_bytes()).hexdigest()}))
print('borrowed-buffer checks complete',flush=True)
