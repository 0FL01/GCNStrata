from pathlib import Path
import subprocess,json,time,datetime,hashlib,shutil,urllib.request
R=Path('/home/radneon/strata');O=R/'logs/integrated-live-20261008';A=O/'api'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(a,**kw):return subprocess.run(a,cwd=R,check=True,**kw)
def out(a):return subprocess.check_output(a,cwd=R,text=True).strip()
def save(): (O/'deployment.json').write_text(json.dumps(report,indent=2))
expected='ce794788d64313236bbc24b47a3f8e06c841034afaf4e19a2fc5ffd84e337f14'
complete=json.loads((O/'complete.json').read_text());assert complete['passed'] and complete['runs']==12
results=json.loads((O/'summary.json').read_text())['results']
assert len(results)==12 and all(x['passed'] for x in results)
for kind,n in [('code',4096),('code',65536),('ru',65536)]:
 g=[x for x in results if x['tag'].startswith(kind+str(n))]
 assert len(g)==4 and all(x['output_ids']==g[0]['output_ids'] for x in g)
assert not out(['docker','ps','-q']),'another active container'
src=R/'logs/perf-campaign-20261007/strata-clean-exact-stack';assert sha(src)==expected
release=R/'releases/gfx906-exact-20261008';release.mkdir(parents=True,exist_ok=False)
shutil.copy2(src,release/'strata');assert sha(release/'strata')==expected
shutil.copy2(R/'logs/perf-campaign-20261007/exact-stack-build/complete.json',release/'build-receipt.json')
for p in ['compose.yaml','configs/shefowl-hybrid-200k.json']:
 shutil.copy2(R/p,O/(Path(p).name+'.before-live'))
oldcfg=json.loads((R/'configs/shefowl-hybrid-200k.json').read_text())
cfg=json.loads((R/'logs/perf-campaign-20261007/exact-stack-api/candidate.json').read_text())
cfg['exe']='/src/releases/gfx906-exact-20261008/strata';cfg['log']='/dev/stderr'
assert cfg['vision']==oldcfg['vision'] and cfg['sampling']==oldcfg['sampling']
env={'STRATA_EXP_MODE':'16','STRATA_Q2_AVX2_SPREAD':'1','STRATA_GFX906_TOPK_REG':'2','STRATA_MMVF_RPB':'4','STRATA_TSUM':'0','STRATA_GR_UP_EXACT':'0'}
compose=(R/'compose.yaml').read_text()
assert compose.count('      STRATA_EXP_MODE: "15"')==1
compose=compose.replace('      STRATA_EXP_MODE: "15"','\n'.join('      '+k+': "'+v+'"' for k,v in env.items()))
devices=[p for p in Path('/sys/bus/pci/devices').glob('*') if (p/'mem_info_vram_total').exists()]
assert len(devices)==2
def perf(mode):
 for p in devices:
  run(['sudo','-n','tee',str(p/'power_dpm_force_performance_level')],input=mode+'\n',text=True,stdout=subprocess.DEVNULL)
  assert (p/'power_dpm_force_performance_level').read_text().strip()==mode
def health():
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key') and h.get('max_context')==204800:return h
  except (OSError,ValueError):pass
  time.sleep(5)
 raise RuntimeError('health timeout')
def up():
 run(['docker','compose','--env-file','/home/radneon/llama/.env','-f','compose.yaml','up','-d','--no-build','--force-recreate','strata'],stdout=subprocess.DEVNULL)
report={'started_utc':stamp(),'passed':False,'release_binary_sha256':expected,'release_path':str(release),'source_commit':'b6731bf','env':env,'R_and_D_paused':True}
save()
try:
 (R/'configs/shefowl-hybrid-200k.json').write_text(json.dumps(cfg,indent=2)+'\n')
 (R/'compose.yaml').write_text(compose)
 perf('high');up();report['health']=health();save()
 for kind in ['tools','vision','parking']:
  with (A/(kind+'-driver.log')).open('w') as f:
   run(['docker','exec','strata-api','python3','/src/logs/integrated-live-20261008/api/'+kind+'.py'],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
  report[kind]='passed';save()
 inspect="""from pathlib import Path
import json,hashlib
wanted=EXPECTED
keys=KEYS
found=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  args=(p/'cmdline').read_bytes().split(bytes([0]))
  if args and args[0]==wanted.encode() and b'--serve' in args:
   e=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(bytes([0])) if b'=' in x)
   found.append({'pid':int(p.name),'argv':[x.decode() for x in args if x],'env':{k:e.get(k.encode(),b'').decode() for k in keys},'binary_sha256':hashlib.sha256((p/'exe').read_bytes()).hexdigest()})
 except OSError:pass
print(json.dumps(found))
""".replace('EXPECTED',repr(cfg['exe'])).replace('KEYS',repr(list(env)))
 procs=json.loads(out(['docker','exec','strata-api','python3','-c',inspect]))
 assert len(procs)==1 and procs[0]['env']==env and procs[0]['binary_sha256']==expected,procs
 report['processes']=procs
 report['health']=health()
 report['restart_policy']=out(['docker','inspect','--format','{{.HostConfig.RestartPolicy.Name}}','strata-api'])
 assert report['restart_policy']=='unless-stopped'
 report['passed']=True;report['finished_utc']=stamp();report['left_running_for_user']=True
 save();print(json.dumps(report),flush=True)
except BaseException as e:
 report['error']=type(e).__name__+': '+str(e);report['failed_utc']=stamp()
 try:
  with (A/'failed-container.log').open('w') as f:subprocess.run(['docker','logs','strata-api'],stdout=f,stderr=subprocess.STDOUT)
  subprocess.run(['docker','stop','--time','60','strata-api'],stdout=subprocess.DEVNULL)
  shutil.copy2(O/'compose.yaml.before-live',R/'compose.yaml')
  shutil.copy2(O/'shefowl-hybrid-200k.json.before-live',R/'configs/shefowl-hybrid-200k.json')
  perf('auto');up();report['rollback_health']=health()
 except BaseException as x:report['rollback_error']=str(x)
 save();print(json.dumps(report),flush=True);raise
