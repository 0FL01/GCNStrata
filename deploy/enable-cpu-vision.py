from pathlib import Path
import json,subprocess,time,os,urllib.request,datetime
R=Path('/home/radneon/strata');O=R/'logs/cpu-vision-20261006';p=R/'configs/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M.json'
assert json.loads((O/'probe-result.json').read_text())['passed']
# Wait for a currently running user generation; only read its busy bit, never log its content.
code="import os,json,urllib.request; q=urllib.request.Request('http://127.0.0.1:8080/status',headers={'Authorization':'Bearer '+os.environ['STRATA_API_KEY']}); print(json.load(urllib.request.urlopen(q,timeout=5)).get('busy',False))"
for attempt in range(60):
 busy=subprocess.check_output(['docker','exec','strata-api','python3','-c',code],text=True).strip()
 assert busy in ['True','False'],busy
 if busy=='False':break
 print('Waiting for active generation to finish',flush=True);time.sleep(5)
else:raise RuntimeError('Service remained busy for5minutes; configuration unchanged')
old=p.read_bytes();backup=O/'config-before-vision-v2.json';assert not backup.exists();backup.write_bytes(old)
cfg=json.loads(old);assert cfg['model_name']=='Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M' and not cfg.get('vision')
native=cfg['args'][cfg['args'].index('--native')+1]
if '--vision' not in cfg['args']:cfg['args'].append('--vision')
cfg['vision']={'exe':'/src/build-vision-cpu/bin/strata-vision','mmproj':'/src/runtime/vision/mmproj-Qwen3.8-Flash-Next-BF16.gguf','model':native,'gpu':False,'threads':16,'max_tokens':1024}
def save(raw):
 tmp=p.with_suffix('.vision-tmp');assert not tmp.exists();tmp.write_bytes(raw);os.chmod(tmp,p.stat().st_mode&0o777);os.replace(tmp,p)
def restart_and_wait(images):
 subprocess.run(['docker','restart','--time','120','strata-api'],check=True,stdout=subprocess.DEVNULL)
 for attempt in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:health=json.load(f)
   if health.get('loaded') and health.get('images')==images and health.get('api_key'):return health
  except (OSError,ValueError):pass
  time.sleep(5)
 raise RuntimeError('API readiness timeout')
save((json.dumps(cfg,indent=2)+'\n').encode());print('Vision config written; restarting Strata',flush=True)
try:
 health=restart_and_wait(True);print('Strata ready with CPU vision',flush=True)
 with (O/'api-test-v2.log').open('x') as log:
  test=subprocess.run(['docker','exec','strata-api','python3','-u','/src/deploy/test-cpu-vision-api.py'],stdout=log,stderr=subprocess.STDOUT)
 assert test.returncode==0,'vision API verification failed; see api-test-v2.log'
 result={'passed':True,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'health':health,'vision_config':cfg['vision'],'old_config_backup':str(backup),'service_left_running':True}
except BaseException as e:
 print('Vision deployment failed; restoring previous text config',flush=True);save(old)
 restored=restart_and_wait(False);result={'passed':False,'error':str(e)[:500],'previous_text_service_restored':bool(restored['loaded'])}
(O/'deployment-result-v2.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] else 1)
