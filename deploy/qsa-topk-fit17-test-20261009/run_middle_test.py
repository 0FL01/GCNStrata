#!/usr/bin/env python3
"""Root-run, one-variant middle parity/component screen. No model-gain claim; --execute required."""
import argparse,fcntl,hashlib,importlib.util,json,math,re,signal,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SUPPORT_SHA='749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45'
p=HERE/'qsa_probe_safety.py'
if hashlib.sha256(p.read_bytes()).hexdigest()!=SUPPORT_SHA:raise RuntimeError('safety helper hash mismatch')
spec=importlib.util.spec_from_file_location('safe',p);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
SOURCE='6bc449d9a436330e56569e91044bb9c87edf2cd4cb7cc1f9a94eef514f0681ce'
PATCH='ed4e7bb2024f5c340e24ffb8517381706f9a6d071f5cb2d0e8f55f6cf5ac76b2'
HARNESS='0eac4ceb11f783e14977f35afdbe0bf6c39fbddaa0fe8c39675e4a0f346a667e'
REAL=['06e572c257f2187e67787f6c813b3d43c0a68584455adf1cff6bed5757b56a1b','a2ccc7241c927f4d7995705dc5a0ad95e6d2f03a8638f093b833156ffe8e2fac']
TRANSITION_SOURCE='d3d1f39b5b0ed6039cb492b6166c4fe17ce23c29da0e837515bdb4341eb34bab'
TRANSITION_MANIFEST='ccb83c03bd8c488efc2b70aeb009bac0469e54b5e610da6b5a99d0aef86c4043'
PREFIX='strata-middle-'

def gate(a):
    d=json.loads(a.gate_receipt.read_text())
    S.need(d.get('passed') is True,'explicit root resource gate PASS required')
    S.need(d.get('source_sha256')==SOURCE and d.get('patch_sha256')==PATCH,'resource gate source binding mismatch')
    S.need(d.get('qsa_topk_middle_probe_sha256')==a.expected_binary_sha and d.get('qsa_reg33_transition_sha256')==a.expected_transition_sha,'resource gate binary binding mismatch')
    for name,expected in [('qsa_select.middle.cu',SOURCE),('qsa-topk-middle-probe.patch',PATCH),('qsa_topk_fixture_replay.cpp',HARNESS),('qsa_reg33_transition.cpp',TRANSITION_SOURCE)]:
        S.need(S.sha(a.build/name)==expected,'source/harness hash changed: '+name)
    S.need(S.sha(a.build/'qsa_topk_middle_probe')==a.expected_binary_sha,'real probe binary changed')
    S.need(S.sha(a.build/'qsa_reg33_transition')==a.expected_transition_sha,'transition binary changed')
    proof=json.loads((a.build/'preprocessor-proof.json').read_text())
    S.need(all(proof.get(k) is True for k in ['platform_shim','wrong_intrinsics_header_absent','half_wave_shift_verified']),'preprocessor gate failed')
    for i,f in enumerate([a.real0,a.real1]):S.need(S.sha(f)==REAL[i] and f.stat().st_size==295268,'real fixture hash/size mismatch')
    mf=a.transition_fixtures/'reg33-transition-manifest.json'
    S.need(S.sha(mf)==TRANSITION_MANIFEST,'transition manifest changed')
    manifest=json.loads(mf.read_text())
    S.need([f['nq'] for f in manifest['fixtures']]==[1,4,6,8,9],'transition T coverage missing')
    for fixture in manifest['fixtures']:
        f=a.transition_fixtures/fixture['file']
        S.need(f.resolve().is_relative_to(a.transition_fixtures.resolve()),'fixture path escape')
        S.need(S.sha(f)==fixture['sha256'],'transition fixture changed')
    return {'receipt_sha256':S.sha(a.gate_receipt),'manual_gate':d,'preprocessor_proof_sha256':S.sha(a.build/'preprocessor-proof.json')}

def launch(cmd,name,ds,out,tag,timeout=180):
    row={'command':cmd,'container':name,'started_utc':S.stamp(),'samples':[]};p=None
    try:
        with (out/(tag+'.stdout')).open('w') as so,(out/(tag+'.stderr')).open('w') as se:
            p=subprocess.Popen(cmd,stdout=so,stderr=se);started=time.monotonic()
            while p.poll() is None:
                S.need(time.monotonic()-started<timeout,'probe timeout')
                S.need(not (out/'cancel').exists(),'cancellation marker')
                row['samples'].append({'elapsed_s':time.monotonic()-started,'gpus':S.gpu(ds)})
                S.write(out/(tag+'.driver.json'),row);time.sleep(0.5)
            row['exit_code']=p.returncode;S.need(p.returncode==0,'probe exit nonzero')
        row['gpu_after']=S.gpu(ds)
        return row
    except BaseException as e:
        row.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        try:S.cleanup(name)
        finally:
            if p is not None:
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                row['exit_code']=p.returncode
            row['finished_utc']=S.stamp();S.write(out/(tag+'.driver.json'),row)

def mapping(ds,out):
    name=PREFIX+'mapping'
    # Read-only HIP ordinal -> BDF query, after stable API is stopped. No settings changed.
    code="import ctypes,json;h=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');r=h.hipInit(0);assert r==0,r;n=ctypes.c_int();r=h.hipGetDeviceCount(ctypes.byref(n));assert r==0 and n.value==2,(r,n.value);o=[]\nfor i in range(2):\n b=ctypes.create_string_buffer(64);r=h.hipDeviceGetPCIBusId(b,64,i);assert r==0,r;o.append(b.value.decode().lower())\nprint(json.dumps(o))"
    cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','-e','HIP_VISIBLE_DEVICES=0,1','--entrypoint','python3',S.IMAGE,'-c',code]
    S.need(not S.call(['docker','ps','-q']).strip(),'another container active before mapping')
    row=launch(cmd,name,ds,out,'mapping',60)
    got=json.loads((out/'mapping.stdout').read_text())
    S.need(got==['0000:05:00.0','0000:08:00.0'],'unexpected HIP visible-ordinal BDF mapping')
    return {'ordinals':got,'driver':row}

def parse_result(text,stderr,device,mode,batch,samples):
    S.need('PARITY_PASS production=reference=captured; padding/end_canaries=PASS' in text,'fixture/reference parity missing')
    S.need('input_bytes_unchanged=PASS graph_ids=PASS' in text,'graph/input parity missing')
    route='kernel=guarded256/reg66 nq=4 capacity_blocks=51202 active_blocks=-1 cap=2051'
    S.need(f'device={device} '+route in stderr,'actual production dispatch/device missing')
    S.need(f'candidate={mode} middle_blocks=33792 device_max_guard=1' in stderr,'middle mode trace mismatch')
    metas=re.findall(r'^META (.*)$',text,re.M);S.need(len(metas)==1,'missing/duplicate metadata')
    m=metas[0]
    for token in [f'replay_device={device}','pos0=65536','nq=4','capacity_blocks=51202','cap=2051',f'batch={batch}',f'samples={samples}']:
        S.need(re.search(r'(^| )'+re.escape(token)+r'( |$)',m) is not None,'metadata mismatch: '+token)
    captured=device;layer=23 if device==0 else 47
    S.need(f'captured_device={captured} ' in m and f'layer={layer} ' in m,'wrong fixture/device metadata')
    lines=re.findall(r'^SAMPLES_MS_PER_CALL (.*)$',text,re.M);S.need(len(lines)==1,'missing/duplicate sample line')
    values=[float(x) for x in lines[0].split()]
    S.need(len(values)==samples and all(math.isfinite(x) and x>0 for x in values),'invalid sample values')
    med=sorted(values)[len(values)//2]
    r=re.search(r'^RESULT median_ms_per_call=([0-9.eE+-]+) min=([0-9.eE+-]+) max=([0-9.eE+-]+)',text,re.M)
    S.need(r is not None and abs(float(r[1])-med)<1e-7,'inconsistent reported median')
    return {'median_ms_per_call':med,'samples_ms':values,'all_parity_passed':True,'timing_scope':'real-input isolated component'}

def run_one(a,ds,out,device,fixture,tag,mode):
    name=PREFIX+tag.lower();S.need(not S.call(['docker','ps','-q']).strip(),'another container became active')
    S.need(S.sha(a.build/'qsa_topk_middle_probe')==a.expected_binary_sha,'binary changed before run')
    expected=REAL[device]
    S.need(S.sha(fixture)==expected,'fixture changed before run')
    before=S.cooldown(ds,out);batch,samples=64,7
    cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host',
         '-e','HIP_VISIBLE_DEVICES=0,1','-e','STRATA_GFX906_TOPK_REG=2','-e','STRATA_TOPK_TRACE=1','-e',f'STRATA_GFX906_TOPK_MID={mode}',
         '-v',str(a.build)+':/work:ro','-v',str(fixture.parent)+':/fixture:ro','--entrypoint','/usr/bin/env',S.IMAGE,
         '-u','STRATA_TOPK_OLD','-u','STRATA_VERIFY_PROFILE','/work/qsa_topk_middle_probe','/fixture/'+fixture.name,str(device),str(batch),str(samples)]
    driver=launch(cmd,name,ds,out,tag)
    result=parse_result((out/(tag+'.stdout')).read_text(),(out/(tag+'.stderr')).read_text(),device,mode,batch,samples)
    result.update(tag=tag,gpu_ordinal=device,mode=mode,fixture_sha256=expected,gpu_before=before,driver=driver)
    return result

def run_transition(a,ds,out,device,mode,fixture):
    t=fixture['nq'];tag=f'trans-g{device}-m{mode}-t{t}';name=PREFIX+tag
    S.need(not S.call(['docker','ps','-q']).strip(),'another container became active')
    S.need(S.sha(a.build/'qsa_reg33_transition')==a.expected_transition_sha,'transition binary changed')
    file=a.transition_fixtures/fixture['file']
    S.need(S.sha(file)==fixture['sha256'],'transition fixture changed before run')
    before=S.cooldown(ds,out)
    cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host',
         '-e','HIP_VISIBLE_DEVICES=0,1','-e','STRATA_GFX906_TOPK_REG=2','-e','STRATA_TOPK_TRACE=1','-e',f'STRATA_GFX906_TOPK_MID={mode}',
         '-v',str(a.build)+':/work:ro','-v',str(a.transition_fixtures)+':/fixture:ro','--entrypoint','/usr/bin/env',S.IMAGE,
         '-u','STRATA_TOPK_OLD','-u','STRATA_VERIFY_PROFILE','/work/qsa_reg33_transition','/fixture/'+file.name,str(device)]
    driver=launch(cmd,name,ds,out,tag)
    text=(out/(tag+'.stdout')).read_text();stderr=(out/(tag+'.stderr')).read_text()
    parse_transition(text,stderr,device,mode,fixture)
    return {'tag':tag,'device':device,'mode':mode,'nq':t,'cases':len(fixture['cases']),'parity_passed':True,
            'fixture_sha256':fixture['sha256'],'timing_claim':False,'gpu_before':before,'driver':driver}

def parse_transition(text,stderr,device,mode,fixture):
    t=fixture['nq'];cases=fixture['cases']
    rows=re.findall(r'^CASE_PASS nq=(\d+) case=(\S+) last_nkv=(\d+) max_active_blocks=(\d+) candidate_intended_route=(\S+) IDs=graph=reference=CPU padding=end_canaries=inputs_unchanged=PASS$',text,re.M)
    S.need(len(rows)==len(cases),'missing/duplicate transition case records')
    for row,case in zip(rows,cases):
        expected_route=case['expected_route']
        S.need((int(row[0]),row[1],int(row[2]),int(row[3]),row[4])==(t,case['name'],case['nkv'][-1],case['max_active_blocks'],expected_route),'transition case mismatch')
    expected=f'TRANSITION_PASS nq={t} cases={len(cases)} graph_instantiations=1 constant_addresses=1 constant_capacities=1 mid={mode} timing_claim=0'
    S.need(text.splitlines().count(expected)==1,'missing/duplicate transition completion')
    if t<=8:
        S.need(f'candidate={mode} middle_blocks=33792 device_max_guard=1' in stderr,'transition candidate mode trace missing')
        S.need(f'device={device} kernel=guarded256/reg66 nq={t} capacity_blocks=65537 active_blocks=-1' in stderr,'transition guarded route/device trace missing')
    else:
        S.need(f'device={device} kernel=reference256 nq=9 capacity_blocks=65537 active_blocks=-1' in stderr,'unchanged nq9 fallback trace missing')
    return True

def completed_parity(a):
    if a.completed_parity_summary is None:return None
    path=a.completed_parity_summary
    S.need(path.is_absolute(),'absolute completed parity receipt required')
    d=json.loads(path.read_text())
    S.need(d.get('passed') is True and d.get('binary_sha256')==a.expected_transition_sha,'completed transition receipt not passed/matching binary')
    S.need(d.get('api_left_stopped') is True and d.get('containers_after')==[],'transition cleanup/stopped state incomplete')
    rows=d.get('results',[])
    tags={f'gpu{g}-mid{m}-t{t}' for g in [0,1] for m in [0,1] for t in [1,4,6,8,9]}
    S.need(len(rows)==20 and {r.get('tag') for r in rows}==tags,'need exactly20 distinct transition tags')
    manifest=json.loads((a.transition_fixtures/'reg33-transition-manifest.json').read_text())
    by_t={f['nq']:f for f in manifest['fixtures']}
    for r in rows:
        t=int(r['tag'].rsplit('-t',1)[1]);f=by_t[t]
        S.need(r.get('passed') is True and r.get('cases')==len(f['cases']) and r.get('fixture_sha256')==f['sha256'],'incomplete or different transition case receipt')
    return {'path':str(path),'sha256':S.sha(path),'processes':20,'case_windows':sum(r['cases'] for r in rows),'binary_sha256':a.expected_transition_sha,'passed':True}

def execute(a):
    S.need(a.leave_api_stopped,'this offline wrapper requires --leave-api-stopped')
    S.need(a.build.resolve().is_relative_to(S.R),'build must lie beneath Strata root')
    S.need(not a.output.exists(),'output already exists')
    g=gate(a);prior_parity=completed_parity(a)
    S.need(not S.call(['docker','ps','-q']).strip(),'API and all other containers must already be stopped')
    ds=S.devices();power=S.gpu(ds)
    a.output.mkdir(mode=0o700);out=a.output
    summary={'scope':'offline right-sized selector component/parity test; no model-gain claim','started_utc':S.stamp(),
             'binary_sha256':a.expected_binary_sha,'transition_binary_sha256':a.expected_transition_sha,'gate':g,
             'api_state_before':'stopped','api_start_stop_actions':False,'power_before':power,'completed_transition_receipt':prior_parity,
             'transition_results':[],'real_results':[],'component_by_device':[],'cleanup':{'passed':False},'promoted':False}
    S.write(out/'summary.json',summary);failure=None
    def stop(signum,frame):raise InterruptedError('requested stop')
    old={sig:signal.signal(sig,stop) for sig in [signal.SIGINT,signal.SIGTERM]}
    try:
        # Explicit ordinals and HIP_VISIBLE_DEVICES=0,1 remain fixed as in completed parity.
        # No extra GPU mapping process is inserted into this bounded20+8 process task.
        if prior_parity is None:
            manifest=json.loads((a.transition_fixtures/'reg33-transition-manifest.json').read_text())
            for device in [0,1]:
                for mode in [0,1]:
                    for fixture in manifest['fixtures']:
                        summary['transition_results'].append(run_transition(a,ds,out,device,mode,fixture))
                        S.write(out/'summary.json',summary)
        for device,fixture in enumerate([a.real0,a.real1]):
            rows=[]
            for tag,mode in [('A1',0),('B1',1),('B2',1),('A2',0)]:
                result=run_one(a,ds,out,device,fixture,f'real-g{device}-{tag}',mode)
                rows.append(result);summary['real_results'].append(result);S.write(out/'summary.json',summary)
            aa=[rows[i]['median_ms_per_call'] for i in [0,3]];bb=[rows[i]['median_ms_per_call'] for i in [1,2]]
            control=sum(aa)/2;candidate=sum(bb)/2;reduction=100*(1-candidate/control)
            record={'device':device,'A_ms':control,'B_ms':candidate,'latency_reduction_pct':reduction,'all_B_faster_than_all_A':max(bb)<min(aa),'engineering_gate_pct':30 if device==0 else None}
            summary['component_by_device'].append(record);S.write(out/'summary.json',summary)
            if device==0:
                S.need(max(bb)<min(aa) and reduction>=30,'first-GPU engineering gate below30% or ABBA separation failed; raw timings preserved, stop without tuning')
        summary['component_screen_completed']=True
        summary['first_gpu_gate_passed']=True
        summary['both_devices_clearly_improve']=all(r['all_B_faster_than_all_A'] for r in summary['component_by_device'])
    except BaseException as e:
        failure=e;summary.update(component_screen_completed=False,error_type=type(e).__name__,error=str(e))
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        try:
            names=[PREFIX+'mapping']+[PREFIX+f'trans-g{d}-m{m}-t{t}' for d in [0,1] for m in [0,1] for t in [1,4,6,8,9]]+[PREFIX+f'real-g{d}-{t}'.lower() for d in [0,1] for t in ['A1','B1','B2','A2']]
            for name in names:S.cleanup(name)
            S.need(not S.call(['docker','ps','-q']).strip(),'unexpected active container after probe cleanup')
            after=S.gpu(ds)
            S.need([(r['pci'],r['mode'],r['power1_cap']) for r in after]==[(r['pci'],r['mode'],r['power1_cap']) for r in power],'power settings changed')
            summary['cleanup']={'passed':True,'api_state_after':'intentionally_stopped','api_start_stop_actions':False,'config_writes':False,'power_after':after}
        except BaseException as e:summary['cleanup']={'passed':False,'error_type':type(e).__name__,'error':str(e)}
        summary['finished_utc']=S.stamp();S.write(out/'summary.json',summary)
        for sig,handler in old.items():signal.signal(sig,handler)
    S.need(summary['cleanup']['passed'],'probe cleanup/final stopped state verification failed')
    if failure:raise failure
    print(json.dumps({'component_screen_completed':True,'component_by_device':summary['component_by_device'],'cleanup':summary['cleanup'],'model_gain_claim':False}),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['build','real0','real1','gate-receipt','transition-fixtures','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--expected-binary-sha',required=True);ap.add_argument('--expected-transition-sha',required=True);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--leave-api-stopped',action='store_true');ap.add_argument('--completed-parity-summary',type=Path);a=ap.parse_args()
    S.need(all(re.fullmatch('[0-9a-f]{64}',v) is not None for v in [a.expected_binary_sha,a.expected_transition_sha]),'explicit expected binary SHAs required')
    S.need(all(getattr(a,k).is_absolute() for k in ['build','real0','real1','gate_receipt','transition_fixtures','output']),'absolute paths required')
    S.need(a.leave_api_stopped,'offline wrapper requires --leave-api-stopped')
    if not a.execute:print(json.dumps({'arguments_valid':True,'executed':False}));return
    S.need(__debug__,'python -O forbidden')
    with open('/tmp/strata-direct-ff34-ce.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()
