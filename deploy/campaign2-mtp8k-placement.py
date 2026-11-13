from pathlib import Path
import json,subprocess,datetime,hashlib
R=Path('/home/radneon/strata');B=R/'logs/perf-campaign-20261007';O=B/'mtp8k-placement'
assert not subprocess.check_output(['docker','ps','-q']).strip()
receipt=json.loads((B/'exact-stack-build/complete.json').read_text())
assert hashlib.sha256((B/'strata-clean-exact-stack').read_bytes()).hexdigest()==receipt['binary_sha256']
base=json.loads((B/'exact-stack-model/runs/code65536-B1.manifest.json').read_text())['argv']
rows=[]
for reserve in [634,635]:
 cmd=list(base);cmd[cmd.index('--name')+1]='strata-mtp8k-placement'
 cmd[cmd.index('--mtp-window')+1]='8192'
 cmd[cmd.index('--vram-reserve-later-mib')+1]=str(reserve)
 with (O/(str(reserve)+'.stderr')).open('x') as err:
  p=subprocess.run(cmd,input='QUIT'+chr(10),text=True,stdout=subprocess.PIPE,stderr=err,timeout=240)
 (O/(str(reserve)+'.stdout')).write_text(p.stdout)
 assert p.returncode==0,p.returncode
 line=next(x for x in p.stdout.splitlines() if x.startswith('INFO '))
 info=dict(x.split('=',1) for x in line.split()[1:] if '=' in x)
 rows.append({'reserve':reserve,'info':info,'argv':cmd})
 print(json.dumps({'reserve':reserve,'info':info}),flush=True)
 (O/'results.json').write_text(json.dumps(rows,indent=2))
 if int(info['expert_slots'])==19078 and int(info['expert_slots_primary'])==9900:
  (O/'complete.json').write_text(json.dumps({'matched':True,'reserve':reserve,'info':info,'binary_sha256':receipt['binary_sha256'],'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));break
else:raise RuntimeError('neither nearest reserve preserves expert placement')
