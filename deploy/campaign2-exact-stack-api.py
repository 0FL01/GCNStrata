from pathlib import Path
import subprocess,json,time,datetime,hashlib
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007/exact-stack-api'
d=json.loads((R/'logs/perf-campaign-20261007/exact-stack-model/summary.json').read_text())
assert len(d['results'])==7 and all(x['passed'] for x in d['results'])
for n,kind in [(4096,'code'),(65536,'code'),(65536,'ru')]:
 rows=[x for x in d['results'] if x['tag'].startswith(kind+str(n))]
 assert len(rows)==2 and rows[0]['output_ids']==rows[1]['output_ids']
assert not subprocess.check_output(['docker','ps','-q']).strip()
devices=[p for p in Path('/sys/bus/pci/devices').glob('*') if (p/'mem_info_vram_total').exists()]
assert len(devices)==2
def perf(mode):
 for p in devices:
  subprocess.run(['sudo','-n','tee',str(p/'power_dpm_force_performance_level')],input=mode+chr(10),text=True,stdout=subprocess.DEVNULL,check=True)
  assert (p/'power_dpm_force_performance_level').read_text().strip()==mode
assert all((p/'power_dpm_force_performance_level').read_text().strip()=='auto' for p in devices)
compose=['docker','compose','-p','strata-rnd-check','--env-file','/home/radneon/llama/.env','-f',str(R/'compose.yaml'),'-f',str(O/'override.yaml')]
result={'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'binary_sha256':hashlib.sha256((R/'logs/perf-campaign-20261007/strata-clean-exact-stack').read_bytes()).hexdigest()}
try:
 perf('high')
 subprocess.run(compose+['up','-d','--no-build'],check=True)
 for _ in range(90):
  health=subprocess.check_output(['docker','inspect','--format','{{.State.Health.Status}}','strata-candidate-api'],text=True).strip()
  if health=='healthy':break
  time.sleep(5)
 else:raise RuntimeError('candidate API did not become healthy')
 result['health']=health
 for kind in ['tools','vision','parking']:
  with (O/(kind+'-driver.log')).open('w') as f:p=subprocess.run(['docker','exec','strata-candidate-api','python3','/src/deploy/campaign2-clean-api-'+kind+'.py'],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
  result[kind+'_exit']=p.returncode
  assert p.returncode==0,kind
 result['passed']=True
finally:
 with (O/'container.log').open('w') as f:subprocess.run(['docker','logs','strata-candidate-api'],stdout=f,stderr=subprocess.STDOUT)
 p=subprocess.run(compose+['down'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (O/'cleanup.log').write_text(p.stdout);result['cleanup_exit']=p.returncode
 perf('auto')
 result['restored_auto']=all((p/'power_dpm_force_performance_level').read_text().strip()=='auto' for p in devices)
 result['production_left_stopped_for_RnD']=True
 result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (O/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
