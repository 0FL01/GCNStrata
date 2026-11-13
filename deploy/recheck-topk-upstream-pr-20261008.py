from pathlib import Path
import subprocess,json,time,datetime,urllib.request,hashlib
R=Path('/home/radneon/strata');O=R/'logs/topk-upstream-pr-20261008'
for arm in ['baseline','candidate']:
 d=json.loads((O/(arm+'-build.json')).read_text());assert d['exit_code']==0
 assert hashlib.sha256((O/(arm+'-strata')).read_bytes()).hexdigest()==d['strata_sha256']
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def health():
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:d=json.load(f)
   if d.get('loaded') and d.get('images') and d.get('api_key') and d.get('max_context')==204800:return d
  except (OSError,ValueError):pass
  time.sleep(5)
 raise RuntimeError('production health restore timeout')
report={'started_utc':stamp(),'passed':False,'api_before':health()}
cfg=R/'configs/shefowl-hybrid-200k.json';cfgsha=hashlib.sha256(cfg.read_bytes()).hexdigest()
(O/'qualification-components-corrected.json').write_text(json.dumps(report,indent=2))
try:
 subprocess.run(['docker','stop','--time','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 for name in ['check']:
  driver=R/('deploy/'+name+'-topk-upstream-pr-20261008.py')
  with (O/(name+'-driver.log')).open('w') as f:p=subprocess.run(['python3',str(driver)],stdout=f,stderr=subprocess.STDOUT)
  report[name+'_exit']=p.returncode
  (O/'qualification-components-corrected.json').write_text(json.dumps(report,indent=2))
  assert p.returncode==0,name+' failed'
 report['passed']=True
except BaseException as e:
 report['error']=type(e).__name__+': '+str(e)
finally:
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==cfgsha,'production config changed during qualification'
 subprocess.run(['docker','start','strata-api'],check=True)
 report['api_restored']=health();report['finished_utc']=stamp()
 (O/'qualification-components-corrected.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True)
assert report['passed']
