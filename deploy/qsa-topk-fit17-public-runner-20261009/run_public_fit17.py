#!/usr/bin/env python3
"""Root-only offline public FIT17 qualification; no GPU action without --execute."""
import argparse,fcntl,hashlib,json,re,signal,math
from pathlib import Path
import private_parser_support as P
S=P.S
PREFIX='strata-public-fit17-'
BINARIES=[f'{test}.{variant}' for test in ['public-parity','capture','transition'] for variant in ['base','fit17']]
SOURCES={'qsa_select.base.cu':'6432aa2d336496a0ebb3d2d3d14ce177caf67327eecb33bd61801497528d6aad','qsa_select.fit17.cu':'5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a'}

def gate(a):
 S.need(S.sha(a.gate)==a.gate_sha,'root expected resource receipt hash mismatch')
 d=json.loads(a.gate.read_text());S.need(d.get('passed') is True and d.get('resource_isa_passed') is True,'actual ISA/resource PASS required')
 S.need(d.get('allowed_modes')==[0,1] and d.get('public_commit')=='ca3533157d066a13b128f3daa61f56411dcf3155','wrong public scope/modes')
 S.need(d.get('support_archive_used') is False,'public build must not use support archive')
 S.need(S.sha(a.build/'build.json')==d.get('build_json_sha256'),'build receipt binding')
 build=json.loads((a.build/'build.json').read_text());S.need(build.get('passed') is True and build.get('support_archive_used') is False,'public build not passed')
 S.need(build.get('input_manifest_sha256')=='00a102313d8823d3415e102eb713172704d3a571bad143cda45dca45f405a0a5' and S.sha(a.build/'INPUTS.sha256')=='00a102313d8823d3415e102eb713172704d3a571bad143cda45dca45f405a0a5','pinned public input manifest mismatch')
 S.need((a.build/'build.exit_code').read_text().strip()=='0','build exited nonzero')
 for name,sha in SOURCES.items():S.need(S.sha(a.build/name)==sha,'public source mismatch')
 S.need(set(d.get('binaries',{}))==set(BINARIES),'six actual binary hashes required')
 for name,sha in d['binaries'].items():
  S.need(re.fullmatch('[0-9a-f]{64}',sha) and S.sha(a.build/name)==sha and build['files'].get(name)==sha,'binary binding '+name)
 resources=d.get('resource_files',{})
 S.need(all(n in resources for n in ['base.resources.log','fit17.resources.log','preprocessor-proof.json']),'resource receipts incomplete')
 S.need(any(n.startswith('qsa_select.base') and n.endswith('.s') for n in resources) and any(n.startswith('qsa_select.fit17') and n.endswith('.s') for n in resources),'actual baseline and candidate ISA bindings required')
 for name,sha in resources.items():
  file=a.build/name;S.need(file.resolve().is_relative_to(a.build.resolve()),'resource path escape')
  S.need(S.sha(file)==sha and build['files'].get(name)==sha,'resource file binding '+name)
 proof=json.loads((a.build/'preprocessor-proof.json').read_text());S.need(proof.get('platform_shim') is True and proof.get('half_wave_ballot') is True,'public preprocessor proof')
 for device,f in enumerate([a.real0,a.real1]):S.need(f.stat().st_size==295268 and S.sha(f)==P.REAL[device],'real fixture mismatch')
 mf=a.fixtures/'reg17-transition-manifest.json';S.need(S.sha(mf)==P.TRANSITION_MANIFEST,'transition manifest mismatch')
 manifest=json.loads(mf.read_text());S.need([f['nq'] for f in manifest['fixtures']]==[1,4,6,8,9],'T coverage')
 for f in manifest['fixtures']:
  file=a.fixtures/f['file'];S.need(file.resolve().is_relative_to(a.fixtures.resolve()) and S.sha(file)==f['sha256'],'transition fixture binding')
 return d,manifest

def trace(stderr,enabled,nq,capacity,cap):
 lines=[s for s in stderr.splitlines() if s.startswith('strata gfx906 topk fit17:')]
 if enabled and nq<=8:
  expected=f'strata gfx906 topk fit17: enabled=1 topology=guarded-ref256/reg17/reg66 max_blocks=17408 nq={nq} capacity_blocks={capacity} cap={cap} (device steps select the per-window arm)'
  S.need(lines==[expected],'public FIT17 topology trace mismatch')
 else:S.need(not lines,'unexpected enabled topology trace')

def parse_capture(text,stderr,device,enabled,batch,samples,trace_on=True):
 S.need('PARITY_PASS production=reference=captured; padding/end_canaries=PASS' in text,'captured/reference parity missing')
 S.need('input_bytes_unchanged=PASS graph_ids=PASS' in text,'graph/input parity missing')
 metas=re.findall(r'^META (.*)$',text,re.M);S.need(len(metas)==1,'metadata count')
 tokens=[f'replay_device={device}',f'captured_device={device}',f'layer={23 if device==0 else 47}','pos0=65536','nq=4','capacity_blocks=51202','cap=2051',f'batch={batch}',f'samples={samples}']
 for token in tokens:S.need(re.search(r'(^| )'+re.escape(token)+r'( |$)',metas[0]),'capture metadata '+token)
 lines=re.findall(r'^SAMPLES_MS_PER_CALL (.*)$',text,re.M);S.need(len(lines)==1,'sample line count')
 vals=list(map(float,lines[0].split()));S.need(len(vals)==samples and all(math.isfinite(v) and v>0 for v in vals),'invalid sample values')
 med=sorted(vals)[len(vals)//2];r=re.search(r'^RESULT median_ms_per_call=([0-9.eE+-]+) min=([0-9.eE+-]+) max=([0-9.eE+-]+)',text,re.M)
 S.need(r and abs(float(r[1])-med)<1e-7 and abs(float(r[2])-min(vals))<1e-7 and abs(float(r[3])-max(vals))<1e-7,'inconsistent timing summary')
 trace(stderr,enabled and trace_on,4,51202,2051)
 return {'median_ms_per_call':med,'samples_ms':vals,'all_parity_passed':True,'trace_enabled':trace_on,'timing_scope':'public real-input isolated component'}

def parse_transition(text,stderr,device,enabled,mode,f):
 # Parse unchanged driver stdout directly; public source has no private route traces.
 rows=re.findall(r'^CASE_PASS nq=(\d+) case=(\S+) last_nkv=(\d+) max_active_blocks=(\d+) candidate_intended_route=(\S+) IDs=graph=reference=CPU padding=end_canaries=inputs_unchanged=PASS$',text,re.M)
 S.need(len(rows)==len(f['cases']),'transition case count')
 for row,c in zip(rows,f['cases']):S.need((int(row[0]),row[1],int(row[2]),int(row[3]),row[4])==(f['nq'],c['name'],c['nkv'][-1],c['max_active_blocks'],c['expected_route']),'transition case mismatch')
 end=f"TRANSITION_PASS nq={f['nq']} cases={len(f['cases'])} graph_instantiations=1 constant_addresses=1 constant_capacities=1 fit17={mode} timing_claim=0"
 S.need(text.splitlines().count(end)==1,'transition completion')
 trace(stderr,enabled,f['nq'],65537,2080)
 return {'cases':len(rows),'parity_passed':True,'timing_claim':False}

def parse_public(text):
 S.need(text.splitlines().count('qsa_topk_parity: PASS')==1 and 'FAIL' not in text,'public selftest summary')
 rows=re.findall(r'^PASS context .*$',text,re.M);S.need(len(rows)==18,'public selftest 18 cases required')
 adv=re.findall(r'^adversarial=(\d+)/(\d+) changed-step-graph=(PASS|SKIP)$',text,re.M)
 S.need(len(adv)==18 and all(x==y for x,y,_ in adv),'public adversarial equality')
 return {'selftest_cases':18,'parity_passed':True,'timing_claim':False}

def execute(a):
 S.need(a.leave_api_stopped and a.build.resolve().is_relative_to(S.R),'offline build scope')
 S.need(not a.output.exists(),'output exists');g,mf=gate(a)
 S.need(not S.call(['docker','ps','-q']).strip(),'all containers/API must already be stopped')
 ds=S.devices();power=S.gpu(ds);a.output.mkdir(mode=0o700);out=a.output
 summary={'scope':'public isolated selector qualification; no model gain claim','gate_sha256':a.gate_sha,'started_utc':S.stamp(),'correctness':[],'timing':[],'component':[],'promoted':False,'api_start_stop_actions':False,'cleanup':{'passed':False}}
 names=[];failure=None
 def stop(signum,frame):raise InterruptedError('requested stop')
 old={s:signal.signal(s,stop) for s in [signal.SIGINT,signal.SIGTERM]}
 def run(binary,device,mode,tag,args,mount=None,public=False,trace_on=True):
  S.need(not S.call(['docker','ps','-q']).strip(),'another container became active')
  S.need(S.sha(a.build/binary)==g['binaries'][binary],'binary changed before run')
  before=S.cooldown(ds,out);name=PREFIX+tag.lower();names.append(name)
  cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','-e',f'HIP_VISIBLE_DEVICES={device}' if public else 'HIP_VISIBLE_DEVICES=0,1','-e','STRATA_GFX906_TOPK_REG=2','-e','STRATA_TOPK_TRACE=1','-e',f'STRATA_GFX906_TOPK_FIT17_TRACE={int(trace_on)}','-v',str(a.build)+':/work:ro']
  if mount:cmd+=['-v',str(mount)+':/fixture:ro']
  cmd+=['--entrypoint','/usr/bin/env',S.IMAGE,'-u','STRATA_TOPK_OLD','-u','STRATA_VERIFY_PROFILE','-u','STRATA_GFX906_TOPK_FIT17']
  if mode is not None:cmd += [f'STRATA_GFX906_TOPK_FIT17={mode}']
  cmd+=['/work/'+binary]+list(map(str,args));driver=P.launch(cmd,name,ds,out,tag,600 if public else 180)
  return (out/(tag+'.stdout')).read_text(),(out/(tag+'.stderr')).read_text(),{'tag':tag,'binary':binary,'binary_sha256':g['binaries'][binary],'device':device,'mode':mode,'gpu_before':before,'driver':driver}
 try:
  # Public harness has no device CLI. Each selftest sees one physical GPU at local ordinal0.
  configs=[('base',None),('fit17',None),('fit17',0),('fit17',1)]
  for device in [0,1]:
   for variant,mode in configs:
    tag=f'public-g{device}-{variant}-m{mode}';text,err,row=run('public-parity.'+variant,device,mode,tag,['--selftest'],public=True)
    row.update(parse_public(text));enabled=variant=='fit17' and mode==1
    trace(err,enabled,8,33796,2051) # first eligible selftest is ctx135176/cap135176
    summary['correctness'].append(row);S.write(out/'summary.json',summary)
  for device in [0,1]:
   for variant,mode in [('base',0),('fit17',0),('fit17',1)]:
    for f in mf['fixtures']:
     file=a.fixtures/f['file'];S.need(S.sha(file)==f['sha256'],'transition changed')
     tag=f'trans-g{device}-{variant}-m{mode}-t{f["nq"]}'
     text,err,row=run('transition.'+variant,device,mode,tag,['/fixture/'+file.name,device],a.fixtures)
     row.update(parse_transition(text,err,device,variant=='fit17' and mode==1,mode,f));summary['correctness'].append(row);S.write(out/'summary.json',summary)
  for device,fixture in enumerate([a.real0,a.real1]):
   for variant,mode in configs:
    S.need(S.sha(fixture)==P.REAL[device],'real fixture changed');tag=f'capture-check-g{device}-{variant}-m{mode}'
    text,err,row=run('capture.'+variant,device,mode,tag,['/fixture/'+fixture.name,device,1,1],fixture.parent)
    result=parse_capture(text,err,device,variant=='fit17' and mode==1,1,1);result.pop('median_ms_per_call');result.pop('samples_ms');result['timing_claim']=False
    row.update(result);summary['correctness'].append(row);S.write(out/'summary.json',summary)
  summary['all_correctness_passed']=True;S.write(out/'summary.json',summary)
  for device,fixture in enumerate([a.real0,a.real1]):
   rows=[]
   for label,mode in [('A1',0),('B1',1),('B2',1),('A2',0)]:
    S.need(S.sha(fixture)==P.REAL[device],'real fixture changed');tag=f'abba-g{device}-{label}'
    text,err,row=run('capture.fit17',device,mode,tag,['/fixture/'+fixture.name,device,64,7],fixture.parent,trace_on=False)
    row.update(parse_capture(text,err,device,mode==1,64,7,trace_on=False));rows.append(row);summary['timing'].append(row);S.write(out/'summary.json',summary)
   score=P.score_abba(rows,device,1);summary['component'].append(score);S.write(out/'summary.json',summary)
   score['original_30pct_discovery_gate_passed']=score.pop('first_gpu_gate_passed') if device==0 else None
   score['status']='public_reproduction_measurement'
   summary['original_discovery_gate_redefined']=False
  summary['public_both_pairs_positive']=all(r['all_B_faster_than_all_A'] for r in summary['component'])
  summary['qualification_completed']=True
 except BaseException as e:failure=e;summary.update(error_type=type(e).__name__,error=str(e),qualification_completed=False)
 finally:
  for sig in old:signal.signal(sig,signal.SIG_IGN)
  try:
   for name in names:S.cleanup(name)
   S.need(not S.call(['docker','ps','-q']).strip(),'unexpected container after cleanup');after=S.gpu(ds)
   S.need([(r['pci'],r['mode'],r['power1_cap']) for r in after]==[(r['pci'],r['mode'],r['power1_cap']) for r in power],'settings changed')
   summary['cleanup']={'passed':True,'api_left_stopped':True,'power_after':after}
  except BaseException as e:summary['cleanup']={'passed':False,'error':str(e)}
  summary['finished_utc']=S.stamp();S.write(out/'summary.json',summary)
  for sig,handler in old.items():signal.signal(sig,handler)
 S.need(summary['cleanup']['passed'],'cleanup failed')
 if failure:raise failure
 print(json.dumps({'qualification_completed':True,'component':summary['component'],'model_gain_claim':False}))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for n in ['build','gate','fixtures','real0','real1','output']:ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--gate-sha',required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--leave-api-stopped',action='store_true');a=ap.parse_args()
 S.need(all(getattr(a,n).is_absolute() for n in ['build','gate','fixtures','real0','real1','output']),'absolute paths required');S.need(re.fullmatch('[0-9a-f]{64}',a.gate_sha),'gate SHA required');S.need(a.leave_api_stopped,'offline only')
 if not a.execute:print('arguments valid; not executed');return
 S.need(__debug__,'python -O forbidden')
 with open('/tmp/strata-direct-ff34-ce.lock','w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()
