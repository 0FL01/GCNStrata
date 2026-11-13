from pathlib import Path
import subprocess,json,hashlib,shutil,time,datetime,urllib.request
R=Path('/home/radneon/strata');O=R/'logs/deploy-pr1187-20261006';O.mkdir(exist_ok=False)
TARGET='2a7503d82f39ac445617d61a9fd82282ebaeef33'
IMAGE='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
def run(args,**kw):return subprocess.run(args,cwd=R,check=True,**kw)
def output(args):return subprocess.check_output(args,cwd=R,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
assert output(['git','rev-parse','HEAD'])=='82f46a8c8f475f001ad76d92f58f4a4f8ffb0253'
expected={'src/kernels/cuda/iq_kernels.cu':'685394dab062b304e810d0f94bbc5fcb666d9495884ea0cd91972f0fe0513f7d','src/kernels/native_expert_bench.cpp':'f9b32acd5f03d12f13b98ffdf21c36e103f62ac62f14e4e80bb14d4cdbe4e6d1'}
for f,h in expected.items():assert sha(R/f)==h,f
for f,name in [('compose.yaml','compose-before.yaml'),('configs/shefowl-hybrid-200k.json','config-before.json'),('build-906/strata','strata-before'),('src/kernels/cuda/iq_kernels.cu','iq-kernels-before.cu'),('src/kernels/native_expert_bench.cpp','bench-before.cpp')]:shutil.copy2(R/f,O/name)
(O/'tracked-before.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=R))
report={'started_utc':stamp(),'before_commit':output(['git','rev-parse','HEAD']),'target_commit':TARGET,'before_binary_sha256':sha(O/'strata-before')}
(O/'state.json').write_text(json.dumps(report,indent=2))
stopped=False
try:
 run(['git','fetch','--no-tags','https://github.com/0FL01/Strata.git','refs/heads/perf/gfx906-q2-gu-global'])
 assert output(['git','rev-parse','FETCH_HEAD'])==TARGET
 path=R/'bench/results/2026-10-06-gfx906-q2-gu-global'
 if path.exists():shutil.move(str(path),str(O/'bench-report-before'))
 run(['git','restore','--source=HEAD','--',*expected.keys()])
 run(['git','switch','-c','deploy/pr1187-mode13'])
 run(['git','merge','--ff-only',TARGET])
 assert output(['git','rev-parse','HEAD'])==TARGET
 run(['docker','stop','--time','120','strata-api'],stdout=subprocess.DEVNULL);stopped=True
 print('API stopped; building exact PR source',flush=True)
 with (O/'build.log').open('w') as log:
  run(['docker','run','--rm','--network','none','--name','strata-pr1187-build','--entrypoint','/bin/sh','-v',str(R)+':/src','-w','/src',IMAGE,'-c','cmake --build build-906 --target strata native_expert_bench -j4'],stdout=log,stderr=subprocess.STDOUT)
 compose=(R/'compose.yaml').read_text();needle='      STRATA_STAGE_TRIM: "1"'
 assert compose.count(needle)==1 and 'STRATA_EXP_MODE:' not in compose
 (R/'compose.yaml').write_text(compose.replace(needle,needle+chr(10)+'      STRATA_EXP_MODE: "13"'))
 run(['docker','compose','--env-file','/home/radneon/llama/.env','-f','compose.yaml','up','-d','--force-recreate','strata'])
 for attempt in range(180):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:health=json.load(f)
   if health.get('loaded') and health.get('images') and health.get('api_key'):break
  except (OSError,ValueError):pass
  time.sleep(5)
 else:raise RuntimeError('health did not become ready')
 exe=json.loads((R/'configs/shefowl-hybrid-200k.json').read_text())['exe']
 inspect_code="""from pathlib import Path
import os,json,hashlib
matches=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  args=(p/'cmdline').read_bytes().split(bytes([0]))
  if args and args[0].decode()==EXE and b'--serve' in args:
   env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(bytes([0])) if b'=' in x)
   matches.append({'pid':int(p.name),'mode':env.get(b'STRATA_EXP_MODE',b'').decode(),'binary_sha256':hashlib.sha256((p/'exe').read_bytes()).hexdigest()})
 except (OSError,UnicodeError):pass
print(json.dumps(matches))
""".replace('EXE',repr(exe))
 processes=json.loads(output(['docker','exec','strata-api','python3','-c',inspect_code]))
 assert len(processes)==1 and processes[0]['mode']=='13',processes
 assert processes[0]['binary_sha256']==sha(R/'build-906/strata')
 assert (R/'configs/shefowl-hybrid-200k.json').read_bytes()==(O/'config-before.json').read_bytes()
 api_code="""import os,json,urllib.request,urllib.error
body={'model':'current','messages':[{'role':'user','content':'What is 17 + 25? Put only the integer in your final answer.'}],'temperature':0,'max_tokens':256}
req=urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['STRATA_API_KEY']})
with urllib.request.urlopen(req,timeout=180) as f:d=json.load(f)
choice=d['choices'][0];content=choice['message'].get('content','')
assert '42' in content,content
try:urllib.request.urlopen('http://127.0.0.1:8080/v1/models',timeout=5)
except urllib.error.HTTPError as e:assert e.code==401
else:raise AssertionError('unauthenticated models endpoint admitted')
print(json.dumps({'content':content,'finish_reason':choice.get('finish_reason'),'usage':d.get('usage'),'unauthenticated_status':401}))
"""
 smoke=json.loads(output(['docker','exec','strata-api','python3','-c',api_code]))
 report.update(finished_utc=stamp(),passed=True,health=health,processes=processes,api_smoke=smoke,commit=output(['git','rev-parse','HEAD']),binary_sha256=sha(R/'build-906/strata'),mode=13,config_unchanged=True)
 (O/'state.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
except BaseException as e:
 report.update(failed_utc=stamp(),passed=False,error=str(e))
 if stopped:
  try:
   run(['docker','stop','--time','120','strata-api'],stdout=subprocess.DEVNULL)
   run(['docker','run','--rm','--network','none','--entrypoint','/bin/sh','-v',str(R)+':/src',IMAGE,'-c','cp /src/logs/deploy-pr1187-20261006/strata-before /src/build-906/strata'])
   shutil.copy2(O/'compose-before.yaml',R/'compose.yaml')
   shutil.copy2(O/'config-before.json',R/'configs/shefowl-hybrid-200k.json')
   run(['docker','compose','--env-file','/home/radneon/llama/.env','-f','compose.yaml','up','-d','--force-recreate','strata'])
   report['rollback_started']=True
  except BaseException as rollback:report['rollback_error']=str(rollback)
 (O/'state.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);raise
