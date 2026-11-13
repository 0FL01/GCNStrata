#!/usr/bin/env python3
"""Root-run, resource-gated FIT17 parity/component screen. No model-gain claim; --execute required."""
import argparse,fcntl,hashlib,importlib.util,json,math,re,signal,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SUPPORT_SHA='749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45'
p=HERE/'qsa_probe_safety.py'
if hashlib.sha256(p.read_bytes()).hexdigest()!=SUPPORT_SHA:raise RuntimeError('safety helper hash mismatch')
spec=importlib.util.spec_from_file_location('safe',p);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
HARNESS='0eac4ceb11f783e14977f35afdbe0bf6c39fbddaa0fe8c39675e4a0f346a667e'
REAL=['06e572c257f2187e67787f6c813b3d43c0a68584455adf1cff6bed5757b56a1b','a2ccc7241c927f4d7995705dc5a0ad95e6d2f03a8638f093b833156ffe8e2fac']
TRANSITION_SOURCE='65738c8661973c6c14c9a7910e29b28c3a75c4e49ac3fd4208be9e7903c17f35'
TRANSITION_MANIFEST='5dee4fb41ba853d3bc94e6d4401c4d8c980b87d1df5efedc8059360ab258a231'
PREFIX='strata-fit17-'
SOURCE='21efb8501aa41ce938daa497c5b3196e275e39a4fadd9138da2cb4e8cf2aded1'
PATCH='a119128d75a0bbadc3af8b40ee1f60a9bdfe190a0e14903bfbcac3877212a81e'

def gate(a):
    S.need(a.expected_source_sha==SOURCE and a.expected_patch_sha==PATCH,'expected source/patch must match reviewed FIT17 revision')
    d=json.loads(a.gate_receipt.read_text())
    S.need(d.get('passed') is True,'explicit root resource gate PASS required')
    S.need(d.get('source_sha256')==a.expected_source_sha and d.get('patch_sha256')==a.expected_patch_sha,'resource gate source binding mismatch')
    S.need(d.get('qsa_topk_fit17_probe_sha256')==a.expected_binary_sha and d.get('qsa_reg17_transition_sha256')==a.expected_transition_sha,'resource gate binary binding mismatch')
    for name,expected in [('qsa_select.fit17.cu',a.expected_source_sha),('qsa-topk-fit17-probe.patch',a.expected_patch_sha),('qsa_topk_fixture_replay.cpp',HARNESS),('qsa_reg17_transition.cpp',TRANSITION_SOURCE)]:
        S.need(S.sha(a.build/name)==expected,'source/harness hash changed: '+name)
    S.need(S.sha(a.build/'qsa_topk_fit17_probe')==a.expected_binary_sha,'real probe binary changed')
    S.need(S.sha(a.build/'qsa_reg17_transition')==a.expected_transition_sha,'transition binary changed')
    proof=json.loads((a.build/'preprocessor-proof.json').read_text())
    S.need(all(proof.get(k) is True for k in ['platform_shim','wrong_intrinsics_header_absent','half_wave_shift_verified']),'preprocessor gate failed')
    for i,f in enumerate([a.real0,a.real1]):S.need(S.sha(f)==REAL[i] and f.stat().st_size==295268,'real fixture hash/size mismatch')
    mf=a.transition_fixtures/'reg17-transition-manifest.json'
    S.need(S.sha(mf)==TRANSITION_MANIFEST,'transition manifest changed')
    manifest=json.loads(mf.read_text())
    S.need([f['nq'] for f in manifest['fixtures']]==[1,4,6,8,9],'transition T coverage missing')
    for fixture in manifest['fixtures']:
        f=a.transition_fixtures/fixture['file']
        S.need(f.resolve().is_relative_to(a.transition_fixtures.resolve()),'fixture path escape')
        S.need(S.sha(f)==fixture['sha256'],'transition fixture changed')
    allowed=d.get('allowed_modes')
    S.need(isinstance(allowed,list) and len(allowed)>0 and all(type(m) is int and m in [1,2] for m in allowed) and len(allowed)==len(set(allowed)),'allowed_modes must explicitly approve a nonempty unique subset of [1,2]')
    return {'allowed_modes':sorted(allowed),'receipt_sha256':S.sha(a.gate_receipt),'manual_gate':d,'preprocessor_proof_sha256':S.sha(a.build/'preprocessor-proof.json')}

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

def parse_mode_trace(stderr,mode):
    expected_hist=1 if mode==2 else 0
    expected_arms='ref256/reg66' if mode==0 else 'ref256/reg17/reg66'
    rows=re.findall(r'^strata topk fit17 probe: variant=(\d+) max_blocks=(\d+) device_max_guard=(\d+) hist2=(\d+) arms=(\S+)$',stderr,re.M)
    S.need(rows==[(str(mode),'17408','1',str(expected_hist),expected_arms)],'FIT17 exact mode trace missing/duplicate/mismatched')

def score_abba(rows,device,mode):
    S.need([r['mode'] for r in rows]==[0,mode,mode,0],'not a fresh ABBA sequence')
    aa=[rows[i]['median_ms_per_call'] for i in [0,3]];bb=[rows[i]['median_ms_per_call'] for i in [1,2]]
    S.need(all(math.isfinite(x) and x>0 for x in aa+bb),'invalid ABBA medians')
    control=sum(aa)/2;candidate=sum(bb)/2;reduction=100*(1-candidate/control)
    separate=max(bb)<min(aa);passed=separate and reduction>=30
    return {'device':device,'mode':mode,'A_ms':control,'B_ms':candidate,
            'A_run_medians':aa,'B_run_medians':bb,'latency_reduction_pct':reduction,
            'all_B_faster_than_all_A':separate,'engineering_gate_pct':30 if device==0 else None,
            'first_gpu_gate_passed':passed if device==0 else None,
            'status':('gpu1_eligible' if passed else 'negative_candidate') if device==0 else ('cross_gpu_improvement' if separate else 'cross_gpu_not_confirmed')}

def parse_result(text,stderr,device,mode,batch,samples):
    S.need('PARITY_PASS production=reference=captured; padding/end_canaries=PASS' in text,'fixture/reference parity missing')
    S.need('input_bytes_unchanged=PASS graph_ids=PASS' in text,'graph/input parity missing')
    route='kernel=guarded256/reg66 nq=4 capacity_blocks=51202 active_blocks=-1 cap=2051'
    S.need(f'device={device} '+route in stderr,'actual production dispatch/device missing')
    parse_mode_trace(stderr,mode)
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
    S.need(S.sha(a.build/'qsa_topk_fit17_probe')==a.expected_binary_sha,'binary changed before run')
    expected=REAL[device]
    S.need(S.sha(fixture)==expected,'fixture changed before run')
    before=S.cooldown(ds,out);batch,samples=64,7
    cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host',
         '-e','HIP_VISIBLE_DEVICES=0,1','-e','STRATA_GFX906_TOPK_REG=2','-e','STRATA_TOPK_TRACE=1','-e',f'STRATA_GFX906_TOPK_FIT17={mode}',
         '-v',str(a.build)+':/work:ro','-v',str(fixture.parent)+':/fixture:ro','--entrypoint','/usr/bin/env',S.IMAGE,
         '-u','STRATA_TOPK_OLD','-u','STRATA_VERIFY_PROFILE','/work/qsa_topk_fit17_probe','/fixture/'+fixture.name,str(device),str(batch),str(samples)]
    driver=launch(cmd,name,ds,out,tag)
    result=parse_result((out/(tag+'.stdout')).read_text(),(out/(tag+'.stderr')).read_text(),device,mode,batch,samples)
    result.update(tag=tag,gpu_ordinal=device,mode=mode,fixture_sha256=expected,gpu_before=before,driver=driver)
    return result

def run_transition(a,ds,out,device,mode,fixture):
    t=fixture['nq'];tag=f'trans-g{device}-m{mode}-t{t}';name=PREFIX+tag
    S.need(not S.call(['docker','ps','-q']).strip(),'another container became active')
    S.need(S.sha(a.build/'qsa_reg17_transition')==a.expected_transition_sha,'transition binary changed')
    file=a.transition_fixtures/fixture['file']
    S.need(S.sha(file)==fixture['sha256'],'transition fixture changed before run')
    before=S.cooldown(ds,out)
    cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host',
         '-e','HIP_VISIBLE_DEVICES=0,1','-e','STRATA_GFX906_TOPK_REG=2','-e','STRATA_TOPK_TRACE=1','-e',f'STRATA_GFX906_TOPK_FIT17={mode}',
         '-v',str(a.build)+':/work:ro','-v',str(a.transition_fixtures)+':/fixture:ro','--entrypoint','/usr/bin/env',S.IMAGE,
         '-u','STRATA_TOPK_OLD','-u','STRATA_VERIFY_PROFILE','/work/qsa_reg17_transition','/fixture/'+file.name,str(device)]
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
    expected=f'TRANSITION_PASS nq={t} cases={len(cases)} graph_instantiations=1 constant_addresses=1 constant_capacities=1 fit17={mode} timing_claim=0'
    S.need(text.splitlines().count(expected)==1,'missing/duplicate transition completion')
    if t<=8:
        parse_mode_trace(stderr,mode)
        S.need(f'device={device} kernel=guarded256/reg66 nq={t} capacity_blocks=65537 active_blocks=-1' in stderr,'transition guarded route/device trace missing')
    else:
        S.need('strata topk fit17 probe:' not in stderr,'unexpected FIT17 probe trace in nq9 fallback')
        S.need(f'device={device} kernel=reference256 nq=9 capacity_blocks=65537 active_blocks=-1' in stderr,'unchanged nq9 fallback trace missing')
    return True

def execute(a):
    S.need(a.leave_api_stopped,'this offline wrapper requires --leave-api-stopped')
    S.need(a.build.resolve().is_relative_to(S.R),'build must lie beneath Strata root')
    S.need(not a.output.exists(),'output already exists')
    g=gate(a);allowed=g['allowed_modes'];modes=[0]+allowed
    S.need(not S.call(['docker','ps','-q']).strip(),'API and all other containers must already be stopped')
    ds=S.devices();power=S.gpu(ds)
    a.output.mkdir(mode=0o700);out=a.output
    summary={'scope':'offline right-sized selector component/parity test; no model-gain claim','started_utc':S.stamp(),
             'binary_sha256':a.expected_binary_sha,'transition_binary_sha256':a.expected_transition_sha,'gate':g,
             'api_state_before':'stopped','api_start_stop_actions':False,'power_before':power,'allowed_modes':allowed,
             'transition_results':[],'real_results':[],'component_by_device':[],'cleanup':{'passed':False},'promoted':False}
    S.write(out/'summary.json',summary);failure=None
    def stop(signum,frame):raise InterruptedError('requested stop')
    old={sig:signal.signal(sig,stop) for sig in [signal.SIGINT,signal.SIGTERM]}
    try:
        manifest=json.loads((a.transition_fixtures/'reg17-transition-manifest.json').read_text())
        for device in [0,1]:
            for mode in modes:
                for fixture in manifest['fixtures']:
                    summary['transition_results'].append(run_transition(a,ds,out,device,mode,fixture))
                    S.write(out/'summary.json',summary)
        summary['transition_parity_passed']=True
        def abba(device,fixture,mode):
            rows=[]
            for tag,arm in [('A1',0),('B1',mode),('B2',mode),('A2',0)]:
                result=run_one(a,ds,out,device,fixture,f'real-g{device}-m{mode}-{tag}',arm)
                rows.append(result);summary['real_results'].append(result);S.write(out/'summary.json',summary)
            record=score_abba(rows,device,mode)
            summary['component_by_device'].append(record);S.write(out/'summary.json',summary)
            return record
        # Finish ALL approved GPU0 candidates even when a preceding one is negative.
        passed=[]
        for mode in allowed:
            record=abba(0,a.real0,mode)
            if record['first_gpu_gate_passed']:passed.append(mode)
        summary['gpu1_eligible_modes']=passed;S.write(out/'summary.json',summary)
        for mode in passed:abba(1,a.real1,mode)
        summary['component_screen_completed']=True
        summary['first_gpu_gate_passed_modes']=passed
        summary['negative_candidate_modes']=[m for m in allowed if m not in passed]
        summary['both_devices_clearly_improve_modes']=[m for m in passed if all(r['all_B_faster_than_all_A'] for r in summary['component_by_device'] if r['mode']==m)]
    except BaseException as e:
        failure=e;summary.update(component_screen_completed=False,error_type=type(e).__name__,error=str(e))
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        try:
            names=[PREFIX+f'trans-g{d}-m{m}-t{t}' for d in [0,1] for m in modes for t in [1,4,6,8,9]]+[PREFIX+f'real-g{d}-m{m}-{t}'.lower() for d in [0,1] for m in allowed for t in ['A1','B1','B2','A2']]
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
    ap.add_argument('--expected-source-sha',required=True);ap.add_argument('--expected-patch-sha',required=True)
    ap.add_argument('--leave-api-stopped',action='store_true');a=ap.parse_args()
    S.need(all(re.fullmatch('[0-9a-f]{64}',v) is not None for v in [a.expected_binary_sha,a.expected_transition_sha,a.expected_source_sha,a.expected_patch_sha]),'explicit expected binary SHAs required')
    S.need(all(getattr(a,k).is_absolute() for k in ['build','real0','real1','gate_receipt','transition_fixtures','output']),'absolute paths required')
    S.need(a.leave_api_stopped,'offline wrapper requires --leave-api-stopped')
    if not a.execute:print(json.dumps({'arguments_valid':True,'executed':False}));return
    S.need(__debug__,'python -O forbidden')
    with open('/tmp/strata-direct-ff34-ce.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()
