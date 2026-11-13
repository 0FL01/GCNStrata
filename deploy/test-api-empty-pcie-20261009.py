from pathlib import Path
import subprocess,json,time,datetime,hashlib,shutil,urllib.request
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261009/empty-pcie/api'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(a,**kw):return subprocess.run(a,cwd=R,check=True,**kw)
def health():
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key') and h.get('max_context')==204800:return h
  except (OSError,ValueError):pass
  time.sleep(5)
 raise RuntimeError('API health timeout')
def up():
 run(['docker','compose','--env-file','/home/radneon/llama/.env','-f','compose.yaml','up','-d','--no-build','--force-recreate','strata'],stdout=subprocess.DEVNULL)
cfgpath=R/'configs/shefowl-hybrid-200k.json';composepath=R/'compose.yaml'
origcfg=cfgpath.read_bytes();origcompose=composepath.read_bytes()
assert json.loads((O.parent/'qualification/summary.json').read_text())['decision']['200k']['capacity_passed']
assert subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).split()==['strata-api']
assert not (O/'config-before.json').exists()
(O/'config-before.json').write_bytes(origcfg);(O/'compose-before.yaml').write_bytes(origcompose)
cfg=json.loads(origcfg);assert cfg.get('vision',{}).get('gpu') is False
cfg['exe']='/src/logs/perf-campaign-20261009/empty-pcie/strata'
expected='050dd8dcbe7d3e9cfaeae870d45df872a1aecded6fff1e1d4611d52f1194c4f4'
assert hashlib.sha256((R/'logs/perf-campaign-20261009/empty-pcie/strata').read_bytes()).hexdigest()==expected
env={'STRATA_VERIFY_SKIP_EMPTY_PCIE':'1','STRATA_GFX906_HC_F16':'1','STRATA_HC_F16_SCALE_PROBE':'1','STRATA_HC_F16_EXACT_INPUT':'1','STRATA_HC_F16_AUDIT':'0','STRATA_HC_F16_TRACE':'1','STRATA_GFX906_MMQ_OPT_CAP':'32','STRATA_GFX906_ATTN_QUERY_SWIZZLE':'1','STRATA_ATTN_LANECELL':'0','STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE':'1'}
compose=origcompose.decode();anchor='      STRATA_EXP_MODE: "16"';assert compose.count(anchor)==1
compose=compose.replace(anchor,anchor+'\n'+'\n'.join('      '+k+': "'+v+'"' for k,v in env.items()))
shutil.copy2(R/'logs/integrated-live-20261008/api/red-blue.png',O/'red-blue.png')
report={'passed':False,'started_utc':stamp(),'candidate_binary_sha256':expected,'candidate_flags':env,'temporary_only':True}
try:
 cfgpath.write_text(json.dumps(cfg,indent=2)+'\n');composepath.write_text(compose)
 up();report['candidate_health']=health()
 inspect = "from pathlib import Path;import hashlib,json;wanted=" + repr(cfg['exe']) + ";keys=" + repr(list(env)) + "\nfound=[]\nfor p in Path('/proc').glob('[0-9]*'):\n try:\n  a=(p/'cmdline').read_bytes().split(bytes([0]))\n  if a and a[0]==wanted.encode() and b'--serve' in a:\n   e=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(bytes([0])) if b'=' in x)\n   found.append({'binary_sha256':hashlib.sha256((p/'exe').read_bytes()).hexdigest(),'env':{k:e.get(k.encode(),b'').decode() for k in keys}})\n except OSError:pass\nprint(json.dumps(found))"
 actual=json.loads(subprocess.check_output(['docker','exec','strata-api','python3','-c',inspect],text=True))
 assert len(actual)==1 and actual[0]['binary_sha256']==expected and actual[0]['env']==env,actual
 report['candidate_process']=actual[0]
 for name in ['tools','vision','parking']:
  with (O/(name+'-driver.log')).open('w') as f:
   run(['docker','exec','strata-api','python3','/src/logs/perf-campaign-20261009/empty-pcie/api/'+name+'.py'],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
  report[name]='passed';(O/'summary.json').write_text(json.dumps(report,indent=2))
 report['passed']=True
except BaseException as e:report['error']=type(e).__name__+': '+str(e)
finally:
 cfgpath.write_bytes(origcfg);composepath.write_bytes(origcompose)
 up();report['restored_health']=health()
 assert cfgpath.read_bytes()==origcfg and composepath.read_bytes()==origcompose
 report['original_configuration_restored']=True;report['finished_utc']=stamp()
 (O/'summary.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True)
assert report['passed'],report.get('error')
