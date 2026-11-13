from pathlib import Path
import subprocess,json,time,datetime,urllib.request
R=Path('/home/radneon/strata');O=R/'logs/shefowl-production-20261006'
assert json.loads((O/'probe-result.json').read_text())['passed']
compose=R/'compose.yaml';start=R/'start.sh';oldcompose=compose.read_bytes();oldstart=start.read_bytes()
assert not (O/'compose-before.yaml').exists()
(O/'compose-before.yaml').write_bytes(oldcompose);(O/'start-before.sh').write_bytes(oldstart)
name='Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0'
cfg=json.loads((R/'configs/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M.json').read_text())
changes={'--pack':'/src/runtime/shefowl-hybrid','--native':'/sata/shefowl-qwen38-hybrid/'+name+'-00001-of-00002.gguf','--ple-gguf':'/sata/shefowl-qwen38-hybrid/'+name+'-00002-of-00002.gguf','--expert-profile':'/src/data/expert-profile.bin','--max-context':'204800','--kv':'int8'}
for flag,value in changes.items():cfg['args'][cfg['args'].index(flag)+1]=value
cfg['args']+=['--ple-io','ram']
cfg['tokenizer']='/src/runtime/shefowl-hybrid/tokenizer';cfg['model_name']=name
cfg['vision']['mmproj']='/sata/shefowl-qwen38-hybrid/mmproj-Qwen3.8-Flash-Next-f16.gguf'
cfg['vision']['model']=changes['--native'];assert cfg['vision']['gpu'] is False
target=R/'configs/shefowl-hybrid-200k.json';assert not target.exists();target.write_text(json.dumps(cfg,indent=2)+chr(10))
text=oldcompose.decode();assert text.count('/src/configs/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M.json')==1
text=text.replace('/src/configs/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M.json','/src/configs/shefowl-hybrid-200k.json')
text=text.replace('      - /home/radneon/models-nvme:/models:ro','      - /home/radneon/models-nvme:/models:ro'+chr(10)+'      - /home/radneon/models:/sata:ro')
newstart=oldstart.decode().replace('runtime/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M/native_experts.txt','runtime/shefowl-hybrid/native_experts.txt')
command=['docker','compose','--env-file','/home/radneon/llama/.env','-f',str(compose)]
def health(expected,capacity):
 for _ in range(180):
  try:
   h=json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3))
   if h.get('loaded') and h.get('images') and h.get('api_key') and h.get('model')==expected and h.get('max_context')==capacity:return h
  except (OSError,ValueError):pass
  time.sleep(5)
 raise RuntimeError('Expected API health did not become ready')
check="import os,json,urllib.request;r=urllib.request.Request('http://127.0.0.1:8080/status',headers={'Authorization':'Bearer '+os.environ['STRATA_API_KEY']});print(json.load(urllib.request.urlopen(r,timeout=5)).get('busy',False))"
for _ in range(120):
 busy=subprocess.check_output(['docker','exec','strata-api','python3','-c',check],text=True).strip()
 if busy=='False':break
 assert busy=='True';time.sleep(5)
else:raise RuntimeError('User request remains active; switch deferred')
changed=False
try:
 compose.write_text(text);start.write_text(newstart);changed=True
 subprocess.run(command+['config','--quiet'],check=True)
 subprocess.run(command+['up','-d','--force-recreate'],check=True)
 h=health(name,204800)
 with (O/'api-test.log').open('w') as f:subprocess.run(['docker','exec','strata-api','python3','/src/deploy/test-shefowl-vision-api.py'],check=True,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 inspect=json.loads(subprocess.check_output(['docker','inspect','strata-api'],text=True))[0]
 policy=inspect['HostConfig']['RestartPolicy']['Name'];assert policy=='unless-stopped'
 result={'passed':True,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'health':h,'restart_policy':policy,'config':str(target),'vision_projector':'bartowski FP16 pinned928589fdb66c6ff07f22ac561e3fbce76553548f','vision_cpu_only':True,'coder_preserved_inactive':True}
 (O/'deployment-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except BaseException as e:
 print('Deployment failed: '+str(e),flush=True)
 if changed:
  compose.write_bytes(oldcompose);start.write_bytes(oldstart)
  subprocess.run(command+['up','-d','--force-recreate'],check=True)
  h=health('Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M',131072)
  (O/'rollback.json').write_text(json.dumps({'reason':str(e),'health':h},indent=2))
 raise
