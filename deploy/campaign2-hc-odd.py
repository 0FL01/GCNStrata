from pathlib import Path
import subprocess,json,time
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007';W=Path('/home/radneon/strata-rnd-sgemm')
for _ in range(720):
 p=O/'hc-sgemm-model/summary.json'
 if p.exists():
  j=json.loads(p.read_text());assert len(j['results'])==8 and all(x['passed'] for x in j['results']) and j['production_restore']['passed'];break
 time.sleep(5)
else:raise RuntimeError('HC model series incomplete')
subprocess.run(['docker','stop','--timeout','60','strata-api'],check=True)
assert not subprocess.check_output(['docker','ps','-q']).strip()
subprocess.run(['git','apply','--check',str(O/'hc-odd-shapes.patch')],cwd=W,check=True)
subprocess.run(['git','apply',str(O/'hc-odd-shapes.patch')],cwd=W,check=True)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake --build build-906 --target gfx906_gemm_screen -j2']
with (O/'hc-odd-build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
P=O/'hc-odd-micro';P.mkdir()
results=[]
for gpu in [0,1]:
 for arm in [0,1,1,0]:
  tag=str(len(results))+'-g'+str(gpu)+'-arm'+str(arm)
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/gfx906_gemm_screen','-v',str(W)+':/src','-e','STRATA_RDNA2_SGEMM=0','-e','STRATA_GFX906_HC_SGEMM='+str(arm),'-e','HIP_VISIBLE_DEVICES='+str(gpu),image]
  with (P/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
  results.append({'tag':tag,'exit':r.returncode});(P/'results.json').write_text(json.dumps(results,indent=2));assert r.returncode==0,tag
(P/'complete.json').write_text(json.dumps({'passed':True,'invocations':len(results)}))
print('odd-shape component series complete',flush=True)
