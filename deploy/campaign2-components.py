from pathlib import Path
import subprocess,json,time,hashlib
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007';P=O/'mode16-components';P.mkdir(exist_ok=False)
for _ in range(360):
 p=O/'build-state.json'
 if p.exists():assert json.loads(p.read_text())['exit']==0;break
 time.sleep(5)
else:raise RuntimeError('no successful build receipt')
assert not subprocess.check_output(['docker','ps','-q']).strip()
results=[]
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
model='/sata/shefowl-qwen38-hybrid/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0-00001-of-00002.gguf'
for gpu in [0,1]:
 for g in [1,8,32]:
  for t in [1,2,4,8]:
   tag='g'+str(gpu)+'-G'+str(g)+'-T'+str(t)
   cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/native_expert_bench','-v',str(R)+':/src','-v','/home/radneon/models:/sata:ro','-e','HIP_VISIBLE_DEVICES='+str(gpu),image,model,'0,23,47',str(g),str(t),'15','16','100']
   with (P/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
   results.append({'tag':tag,'exit':r.returncode});(P/'results.json').write_text(json.dumps(results,indent=2))
   assert r.returncode==0,tag
(P/'complete.json').write_text(json.dumps({'cases':len(results)*3,'binary_sha256':hashlib.sha256((R/'build-906/strata').read_bytes()).hexdigest()}))
print('mode16 component screen completed',flush=True)
