from pathlib import Path
import hashlib,json,subprocess,time,fcntl,importlib.util,re,datetime
R=Path('/home/radneon/strata');D=R/'logs/perf-campaign-20261009/qsa-topk-middle-probe';O=D/'transition-runs-v2'
lock=open('/tmp/strata-direct-ff34-ce.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['docker','ps','-q']).strip()
exe=D/'qsa_reg33_transition'
assert hashlib.sha256(exe.read_bytes()).hexdigest()=='6648c48e4bbe49d31426c14e7cda70495e1c2787cdba95aefc43d7a66255f357'
helper=R/'deploy/run-qsa-reduce12-public-probe-20261009.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45'
sp=importlib.util.spec_from_file_location('safety',helper);s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
ds=s.devices();s.gpu(ds);O.mkdir(exist_ok=False);results=[];summary={'results':results,'api_left_stopped':True,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
try:
 for device in [0,1]:
  for flag in [0,1]:
   for nq in [1,4,6,8,9]:
    tag=f'gpu{device}-mid{flag}-t{nq}';name='strata-middle-transition-'+tag
    assert not subprocess.check_output(['docker','ps','-q']).strip()
    fixture=D/'reg33-transition-fixtures'/f'reg33-transition-t{nq}.bin'
    cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','-v',str(R)+':/src:ro','-e','HIP_VISIBLE_DEVICES=0,1','-e','STRATA_GFX906_TOPK_REG=2','-e','STRATA_TOPK_TRACE=1','-e','STRATA_GFX906_TOPK_MID='+str(flag),'--entrypoint','/src/'+str(exe.relative_to(R)),s.IMAGE,'/src/'+str(fixture.relative_to(R)),str(device)]
    p=None;samples=[];start=time.monotonic()
    try:
     with (O/(tag+'.stdout')).open('x') as out,(O/(tag+'.stderr')).open('x') as err:
      p=subprocess.Popen(cmd,stdout=out,stderr=err)
      while p.poll() is None:
       assert time.monotonic()-start<180,'parity timeout'
       assert not (O/'cancel').exists(),'cancellation marker'
       samples.append(s.gpu(ds));time.sleep(1)
      assert p.returncode==0,'transition process failed '+tag
     text=(O/(tag+'.stdout')).read_text();err=(O/(tag+'.stderr')).read_text()
     m=re.search(r'TRANSITION_PASS nq=([0-9]+) cases=([0-9]+) graph_instantiations=1 constant_addresses=1 constant_capacities=1 mid=([01]) timing_claim=0',text)
     assert m and int(m[1])==nq and int(m[3])==flag,tag
     assert text.count('CASE_PASS ')==int(m[2]),'missing case'
     if nq<=8:assert f'strata topk mid probe: candidate={flag} middle_blocks=33792 device_max_guard=1' in err
     row={'tag':tag,'passed':True,'cases':int(m[2]),'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start,'gpu_samples':samples}
     results.append(row);(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='gpu_samples'}),flush=True)
    finally:
     if p is not None and p.poll() is None:
      subprocess.run(['docker','stop','--time','15',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
      try:p.wait(timeout=30)
      except subprocess.TimeoutExpired:subprocess.run(['docker','kill',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);p.wait(timeout=30)
 summary['passed']=len(results)==20
finally:
 summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();summary['containers_after']=subprocess.check_output(['docker','ps','--format','{{.Names}}']).decode().split()
 (O/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'passed':summary.get('passed',False),'processes':len(results),'cases':sum(x['cases'] for x in results),'api_left_stopped':True}),flush=True)
