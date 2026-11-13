from pathlib import Path
import subprocess,json,time,urllib.request
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/swiglu-d4';D=O/'micro';D.mkdir(exist_ok=False)
assert json.loads((O/'build.json').read_text())['exit_code']==0
summary={'runs':[]};image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
def run(tag,gpu,args,on):
 cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-swiglu-d4-micro','--entrypoint','/src/logs/perf-campaign-20261008/swiglu-d4/gfx906_swiglu_quant_probe','-e','HIP_VISIBLE_DEVICES='+str(gpu),'-e','STRATA_GFX906_MMQ_SWIGLU_QUANT='+str(on),'-v',str(R)+':/src',image]+args
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
 (D/(tag+'.log')).write_text(p.stdout)
 row={'tag':tag,'gpu':gpu,'exit_code':p.returncode,'last_lines':p.stdout.splitlines()[-3:]};summary['runs'].append(row);print(json.dumps(row),flush=True)
 assert p.returncode==0,p.stdout[-4000:]
 return p.stdout
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 run('gate-off',0,['--gate-off'],0)
 for gpu in [0,1]:
  t=run('strict-gpu%d'%gpu,gpu,['--device','0'],1)
  assert 'STRICT_GATE_PASS executed_cases=54 mandatory_type42_cases=18 mismatched_bytes=0 guards=exact timing_cases=0 model_callsite=absent' in t
 for gpu in [0,1]:
  t=run('bench-gpu%d'%gpu,gpu,['--device','0','--bench'],1)
  assert 'STRICT_GATE_PASS executed_cases=64 mandatory_type42_cases=28 mismatched_bytes=0 guards=exact timing_cases=10 model_callsite=absent' in t
 summary['passed']=True
except Exception as e:
 summary['error']=repr(e);raise
finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 summary['api_restored']=False
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):summary['api_restored']=True;break
  except (OSError,ValueError):pass
  time.sleep(5)
 (D/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);assert summary['api_restored']
