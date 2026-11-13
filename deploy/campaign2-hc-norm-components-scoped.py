from pathlib import Path
import subprocess,json,datetime,signal,hashlib
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007/hc-norm-reuse'
assert not subprocess.check_output(['docker','ps','-q']).strip()
for variant in ['baseline','candidate']:
 assert json.loads((O/(variant+'-build.json')).read_text())['build_exit']==0
devices=[d.resolve() for d in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')) if (d/'mem_info_vram_total').exists()]
assert len(devices)==2
saved={str(d):(d/'power_dpm_force_performance_level').read_text().strip() for d in devices};assert set(saved.values())=={'auto'}
def perf(mode):
 for d in devices:
  subprocess.run(['sudo','-n','tee',str(d/'power_dpm_force_performance_level')],input=mode+'\n',text=True,stdout=subprocess.DEVNULL,check=True)
  assert (d/'power_dpm_force_performance_level').read_text().strip()==mode
def sensors():
 out=[]
 for d in devices:
  r={'pci':d.name,'mode':(d/'power_dpm_force_performance_level').read_text().strip()}
  for hw in (d/'hwmon').glob('hwmon*'):
   for key in ['temp1_input','power1_input','power1_cap','freq1_input','freq2_input']:
    p=hw/key
    if p.exists():r[key]=int(p.read_text())
   for i in [1,2,3]:
    p=hw/f'temp{i}_input';q=hw/f'temp{i}_crit'
    if p.exists() and q.exists():assert int(p.read_text())<int(q.read_text())-10000
  assert r.get('power1_cap')==190000000
  out.append(r)
 return out
def interrupt(sig,frame):raise KeyboardInterrupt(str(sig))
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGHUP,interrupt)
rows=[]
try:
 perf('high')
 for gpu in [0,1]:
  for variant in ['baseline','candidate']:
   tag=f'g{gpu}-{variant}-parity-scope'
   cmd=['docker','run','--rm','--name','strata-hc-norm-component','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/logs/perf-campaign-20261007/hc-norm-reuse/'+variant+'-gr_parity','-v',str(R)+':/src','-e',f'HIP_VISIBLE_DEVICES={gpu}','-e','STRATA_HC_SPLIT=2','-e','STRATA_GR_V3=0','-e','STRATA_GR_SPLIT=0','-e','STRATA_QFUSE=0','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','--selftest']
   with (O/(tag+'.log')).open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
   rows.append({'tag':tag,'exit':p.returncode,'sensors':sensors()});print(json.dumps(rows[-1]),flush=True)
   text=(O/(tag+'.log')).read_text()
   assert p.returncode==1 and 'QFUSE: T=1 rep 0: q8_1 bytes differ' in text and 'gr_read/gr_write: 1 failures' in text,tag
   assert 'fused GR multi max-T differs' not in text and 'captured graph differs' not in text,tag
   rows[-1]['known_baseline_optional_qfuse_failure']=True
  for arm,variant in [('A1','baseline'),('B1','candidate'),('B2','candidate'),('A2','baseline')]:
   tag=f'g{gpu}-{arm}'
   cmd=['docker','run','--rm','--name','strata-hc-norm-component','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/logs/perf-campaign-20261007/hc-norm-reuse/'+variant+'-fused_gr_bench','-v',str(R)+':/src','-e',f'HIP_VISIBLE_DEVICES={gpu}','-e','STRATA_HC_SPLIT=2','-e','STRATA_GR_V3=0','-e','STRATA_GR_SPLIT=0','-e','STRATA_QFUSE=0','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','300','1','8']
   before=sensors()
   with (O/(tag+'.log')).open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
   text=(O/(tag+'.log')).read_text()
   rows.append({'tag':tag,'variant':variant,'exit':p.returncode,'sensors_before':before,'sensors_after':sensors()})
   print(json.dumps(rows[-1]),flush=True)
   assert p.returncode==0 and 'variant=3 scope=whole-read' in text and 'fused_gr_bench: 0 differing' in text and 'FAIL' not in text,tag
finally:
 subprocess.run(['docker','rm','-f','strata-hc-norm-component'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 perf('auto')
 (O/'summary.json').write_text(json.dumps({'rows':rows,'restored_power_modes':{str(d):(d/'power_dpm_force_performance_level').read_text().strip() for d in devices},'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print('HC norm component ABBA complete',flush=True)
