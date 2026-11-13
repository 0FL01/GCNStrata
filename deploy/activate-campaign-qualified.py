from pathlib import Path
import subprocess,json,hashlib,shutil,time,datetime,urllib.request,sys
R=Path('/home/radneon/strata');C=R/'logs/perf-campaign-20261006';O=C/'deployment';O.mkdir(exist_ok=False)
target=sys.argv[1]
env=json.loads((C/'final-ab/selected.json').read_text())['candidate_env']
expected_binary='3fdb12152c5e148c7af241cc4cde0e4f71494f657bba48f6ca9bb363b90757f6'
def run(a,**kw):return subprocess.run(a,cwd=R,check=True,**kw)
def out(a):return subprocess.check_output(a,cwd=R,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def health():
 for _ in range(180):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key') and h.get('max_context')==204800:return h
  except (OSError,ValueError):pass
  time.sleep(5)
 raise RuntimeError('health timeout')
def compose_up():run(['docker','compose','--env-file','/home/radneon/llama/.env','-f','compose.yaml','up','-d','--force-recreate','strata'],stdout=subprocess.DEVNULL)
for name,count in [('final-ab',16),('capacity-200k',1)]:
 ss=json.loads((C/name/'summary.json').read_text());assert len(ss['results'])==count and all(r['passed'] for r in ss['results']),name
assert sha(C/'strata-before')==expected_binary
assert (R/'configs/shefowl-hybrid-200k.json').read_bytes()==(C/'config-before.json').read_bytes()
assert (R/'compose.yaml').read_bytes()==(C/'compose-before.yaml').read_bytes()
shutil.copy2(R/'compose.yaml',O/'compose-before.yaml')
(O/'tracked-before.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=R))
report={'started_utc':stamp(),'target_commit':target,'selected_env':env}
(O/'state.json').write_text(json.dumps(report,indent=2))
try:
 run(['docker','stop','--timeout','60','strata-api'],stdout=subprocess.DEVNULL)
 assert not out(['docker','ps','-q']),'another active container'
 for name in ['pr1181.patch','pr1120.patch']:run(['git','apply','-R','--check',str(C/name)])
 for name in ['pr1181.patch','pr1120.patch']:run(['git','apply','-R',str(C/name)])
 for name in ['CMakeLists.txt','src/kernels/native_grouped_parity.cpp']:
  shutil.copy2(R/name,O/(Path(name).name+'.before'))
 run(['git','restore','--source=HEAD','--','CMakeLists.txt','src/kernels/native_grouped_parity.cpp'])
 p=R/'src/kernels/mmvf_rows_parity.cpp'
 if p.exists():shutil.move(str(p),str(O/'mmvf_rows_parity-before.cpp'))
 run(['git','fetch','--no-tags','https://github.com/0FL01/Strata.git','refs/heads/perf/gfx906-hybrid-qualified'])
 assert out(['git','rev-parse','FETCH_HEAD'])==target
 run(['git','switch','-c','deploy/hybrid-qualified'])
 run(['git','merge','--ff-only',target])
 run(['docker','run','--rm','--network','none','--entrypoint','/bin/sh','-v',str(R)+':/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c','cp /src/logs/perf-campaign-20261006/strata-before /src/build-906/strata'])
 assert sha(R/'build-906/strata')==expected_binary
 compose=(R/'compose.yaml').read_text();needle='      STRATA_EXP_MODE: "13"'
 assert compose.count(needle)==1
 lines=['      '+k+': "'+v+'"' for k,v in env.items()]
 for k in env:
  if k!='STRATA_EXP_MODE':assert k+':' not in compose,k
 (R/'compose.yaml').write_text(compose.replace(needle,chr(10).join(lines)))
 compose_up();h=health()
 for script in ['campaign-final-tools.py','campaign-final-vision.py']:
  with (O/(script+'.log')).open('w') as log:
   run(['docker','exec','strata-api','python3','/src/deploy/'+script],stdout=log,stderr=subprocess.STDOUT)
 inspect="""from pathlib import Path
import json,hashlib
keys=KEYS
found=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  args=(p/'cmdline').read_bytes().split(bytes([0]))
  if args and args[0]==b'/src/build-906/strata' and b'--serve' in args:
   e=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(bytes([0])) if b'=' in x)
   found.append({'pid':int(p.name),'env':{k:e.get(k.encode(),b'').decode() for k in keys},'binary_sha256':hashlib.sha256((p/'exe').read_bytes()).hexdigest()})
 except OSError:pass
print(json.dumps(found))
""".replace('KEYS',repr(list(env)))
 procs=json.loads(out(['docker','exec','strata-api','python3','-c',inspect]))
 assert len(procs)==1 and procs[0]['env']==env and procs[0]['binary_sha256']==expected_binary,procs
 assert (R/'configs/shefowl-hybrid-200k.json').read_bytes()==(C/'config-before.json').read_bytes()
 report.update(passed=True,finished_utc=stamp(),health=h,processes=procs,commit=out(['git','rev-parse','HEAD']),restart_policy=out(['docker','inspect','--format','{{.HostConfig.RestartPolicy.Name}}','strata-api']),config_unchanged=True,api_tests=['tool roundtrip','CPU vision','text after vision','models401','UI404'])
 (O/'state.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
except BaseException as e:
 report.update(passed=False,error=type(e).__name__+': '+str(e),failed_utc=stamp())
 try:
  subprocess.run(['docker','stop','--timeout','60','strata-api'],cwd=R,stdout=subprocess.DEVNULL)
  shutil.copy2(O/'compose-before.yaml',R/'compose.yaml')
  run(['docker','run','--rm','--network','none','--entrypoint','/bin/sh','-v',str(R)+':/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c','cp /src/logs/perf-campaign-20261006/strata-before /src/build-906/strata'])
  compose_up();report['rollback_health']=health()
 except BaseException as x:report['rollback_error']=str(x)
 (O/'state.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);raise
