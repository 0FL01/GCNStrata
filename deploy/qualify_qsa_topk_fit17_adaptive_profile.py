#!/usr/bin/env python3
"""Conditional default adaptive-profile end-to-end FIT17 ABBA; includes policy response. No API start, health check or promotion.
Default is offline spec validation. Root supplies new binary hash and production gate.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
R=Path('/home/radneon/strata')
O=R/'logs/perf-campaign-20261009/qsa-topk-fit17-model'
BASE_SHA='ff348b38ad9b531e8df701311b1ae736f1378a6aaef322eeaa86e78f685917d9'
IMAGE='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
FLAG='STRATA_GFX906_TOPK_FIT17';TRACE=FLAG+'_TRACE'
FF34_FLAGS_SHA='efb2e4020356ba79f1f2624197b05cf0a3b45ba42dd692ca952b82ba71639474'
NATIVE_ARGS_SHA='d94b0bb9ffb0463dcce52afe87bb57d8e8ea3c2d037ca2022b254b77b099d953'
def need(ok,label):
    if not ok:raise RuntimeError(label)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def call(args,timeout=120):
    return subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,timeout=timeout).stdout


def devices():
    d=[x.resolve() for x in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')) if (x/'mem_info_vram_total').exists()]
    need(len(d)==2 and {x.name for x in d}=={'0000:05:00.0','0000:08:00.0'},'expected two GPUs')
    return d

def gpu(ds):
    rows=[]
    for d in ds:
        row={'pci':d.name,'mode':(d/'power_dpm_force_performance_level').read_text().strip()}
        need(row['mode']=='high','power mode must remain high')
        hw=list((d/'hwmon').glob('hwmon*'));need(len(hw)==1,'unique hwmon required');h=hw[0]
        for name in ['temp1_input','temp2_input','temp3_input','power1_input','power1_cap','freq1_input','freq2_input']:
            f=h/name
            if f.exists():row[name]=int(f.read_text())
        need(row.get('power1_cap')==190000000 and 'temp1_input' in row,'power cap/temperature telemetry')
        for i,ceiling in [(1,80000),(2,90000),(3,90000)]:
            key=f'temp{i}_input'
            if key in row:
                crit=h/f'temp{i}_crit';limit=min(ceiling,int(crit.read_text())-10000) if crit.exists() else ceiling
                need(row[key]<limit,'thermal safety margin reached')
        rows.append(row)
    return rows

def cooldown(ds,out):
    started=time.monotonic()
    while True:
        need(not (out/'cancel').exists(),'cancellation marker')
        rows=gpu(ds)
        if all(max(v for k,v in r.items() if k.startswith('temp'))<=55000 for r in rows):return rows
        need(time.monotonic()-started<900,'cooldown timeout')
        time.sleep(5)


def run_arm(tag,r,s,ids,ds,out):
    need(not call(['docker','ps','-q']).strip(),'unexpected active container')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    need(available>=64*1024**3,'insufficient RAM')
    need(sha(r['binary'])==r['binary_sha256'],'binary changed')
    initial=cooldown(ds,out)
    name='strata-fit17-adaptive-'+tag.lower();binary='/src/'+str(Path(r['binary']).relative_to(R))
    cmd=['docker','run','--rm','-i','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--ulimit','memlock=-1:-1','--entrypoint',binary]
    for k,v in sorted(r['flags'].items()):cmd+=['-e',k+'='+v]
    cmd+=['-v',str(R)+':/src:ro','-v','/home/radneon/models-nvme:/models:ro','-v','/home/radneon/models:/sata:ro',IMAGE,'--serve']+r['native_args']
    m={'tag':tag,'release_binary_sha256':r['binary_sha256'],'release_artifacts':r['release_artifacts'],'flags':r['flags'],'native_args':r['native_args'],'fixture_sha256':s['fixture_sha256'],'sampling':{'temperature':0,'top_p':1,'top_k':1,'seed':12345},'started_utc':stamp(),'gpu_before':initial,'fresh_process':True}
    write(out/(tag+'.manifest.json'),m);print(json.dumps({'event':'start','tag':tag}),flush=True)
    private=out/'private';private.mkdir(mode=0o700,exist_ok=True);need(private.stat().st_mode & 0o777==0o700,'private log directory mode')
    stderr_path=private/(tag+'.stderr');m['private_stderr_file']='private/'+tag+'.stderr'
    p=None;tokens=[];info={};samples=[];started=time.monotonic();ready=None;request_start=None;done=None;q=queue.Queue()
    try:
        # Full stderr is private diagnostic evidence, never a public report/log upload.
        fd=os.open(stderr_path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        with os.fdopen(fd,mode='w+t') as err:
            p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
            def reader():
                try:
                    for line in p.stdout:q.put(line)
                finally:q.put(None)
            t=threading.Thread(target=reader,daemon=True);t.start();last=0
            while done is None:
                now=time.monotonic();need(now-started<1800,'native arm timeout');need(not (out/'cancel').exists(),'cancellation marker')
                if now-last>=5:samples.append({'elapsed_s':now-started,'gpus':gpu(ds)});last=now
                try:line=q.get(timeout=5)
                except queue.Empty:continue
                need(line is not None,'engine ended before DONE');need(not line.startswith('ERR '),'native error')
                if line.startswith('INFO '):info=dict(x.split('=',1) for x in line.split()[1:] if '=' in x)
                elif line.startswith('READY '):
                    need(ready is None and line.split()[1]=='204800','READY protocol')
                    for k,v in {'expert_slots':'19078','expert_slots_primary':'9900','kv':'int8','context':'204800','lookup':'3','mtp_max':'4','spec':'6','kv_resident':'32768','pcie_frac':'0.00'}.items():need(info.get(k)==v,'engine configuration '+k)
                    err.flush();need('locked in RAM' in os.pread(err.fileno(),os.fstat(err.fileno()).st_size,0).decode(errors='replace'),'PLE RAM lock missing')
                    ready=time.monotonic();m['load_wall_s']=ready-started;m['engine_info']=info;request_start=time.monotonic()
                    p.stdin.write('GEN 1024 temperature=0 top_p=1 top_k=1 seed=12345 '+','.join(map(str,ids))+'\n');p.stdin.flush()
                elif line.startswith('T '):tokens.append(int(line.split()[1]));need(len(tokens)<=1024,'too many outputs')
                elif line.startswith('DONE '):need(request_start is not None,'DONE before READY');done=line.split();m['request_wall_s']=time.monotonic()-request_start
            p.stdin.write('QUIT\n');p.stdin.flush();p.stdin.close();code=p.wait(timeout=120);t.join(timeout=5);need(code==0,'native exit')
            err.flush();stderr=os.pread(err.fileno(),os.fstat(err.fileno()).st_size,0).decode(errors='replace');need('strata gfx906 topk fit17:' not in stderr,'unexpected FIT17 trace');need('strata attn-reduce12: selected=1 ' not in stderr,'unexpected performance trace')
            suffix=suffix_totals(stderr);m['suffix_drafts']=suffix
            if tag.startswith('B') and r['flags'].get('STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE')=='1':need('attn-query-swizzle: active=1 ' in stderr,'candidate QSA route absent')
        need(len(done)>=15 and int(done[1])==len(tokens)==1024 and int(done[2])==65536 and done[5]=='length','DONE output coverage')
        need(int(done[8])==0 and int(done[14])==65536,'prompt reuse/read')
        need(all(0<=x<248320 for x in tokens),'output ID domain')
        pm,dm=map(float,done[3:5]);need(math.isfinite(pm) and math.isfinite(dm) and pm>0 and dm>0,'durations')
        extended=native_counter_fields(done)
        result={'tag':tag,'passed':True,'prompt_tokens':65536,'actual_outputs':1024,'prompt_ms':pm,'decode_ms':dm,'PP':65536000/pm,'TG':1024000/dm,'accepted':int(done[6]),'offered':int(done[7]),'cache_hits':int(done[9]),'lookups':int(done[10]),'prompt_reused':int(done[8]),'prompt_read':int(done[14]),'output_ids':tokens,**extended,'engine_info':info,'suffix_drafts':suffix,'request_wall_s':m['request_wall_s'],'load_wall_s':m['load_wall_s']}
        write(out/(tag+'.result.json'),result);return result
    finally:
        if p is not None and p.poll() is None:
            try:call(['docker','stop','--time','30',name],60)
            except (subprocess.SubprocessError,OSError):
                try:call(['docker','kill',name],30)
                except (subprocess.SubprocessError,OSError):pass
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        if stderr_path.exists():m['private_stderr_sha256']=sha(stderr_path)
        m.update(finished_utc=stamp(),elapsed_s=time.monotonic()-started,native_exit=p.returncode if p else None)
        write(out/(tag+'.manifest.json'),m);write(out/(tag+'.gpu-samples.json'),samples)

TOPOLOGY_KEYS=['expert_slots','expert_slots_primary','expert_cache_mib','expert_cache_primary_mib','context','kv_resident','kv','spec','mtp_max','lookup','pcie_frac','spec_min_p','arena_mib','pool_workers','conversation_cache_mib','conversation_cache_slots','conversation_cache_min_free_mib','tail_role_token','vram_elastic','cvec','engine']
COUNTER_KEYS=['accepted','offered','cache_hits','lookups','prompt_reused','prompt_read','ram_blobs','file_blobs','file_read_mb','offloaded_experts','lookup_chain_accepted','lookup_chain_offered','native_suffix_accepted','native_suffix_offered','decode_windows']

def native_counter_fields(done):
    # Verified qualified generate.cpp e7a5a754, DONE emission lines10012–10030.
    need(len(done)==21,'complete default lookup-chain DONE counters required')
    integer_fields={11:'ram_blobs',12:'file_blobs',15:'offloaded_experts',16:'lookup_chain_accepted',17:'lookup_chain_offered',18:'native_suffix_accepted',19:'native_suffix_offered',20:'decode_windows'}
    values={name:int(done[i]) for i,name in integer_fields.items()};values['file_read_mb']=float(done[13])
    need(all(math.isfinite(v) and v>=0 for v in values.values()),'native counter domain')
    need(values['lookup_chain_accepted']<=values['lookup_chain_offered'] and values['native_suffix_accepted']<=values['native_suffix_offered'],'native draft counter domain')
    return values

def matches_historical_profile(result,history):
    need(result['passed'] and result['actual_outputs']==1024 and result['prompt_tokens']==65536,'complete default-profile arm')
    need(result['output_ids']==history['output_ids'],'historical default full ID mismatch')
    for key in TOPOLOGY_KEYS:need(result['engine_info'][key]==history['engine_info'][key],'historical topology mismatch '+key)

def counter_values(result):
    return {**{k:result[k] for k in COUNTER_KEYS},**{'suffix_'+k:result['suffix_drafts'][k] for k in ['windows','accepted','offered']}}

def compare(results):
    need([r['tag'] for r in results]==['A1','B1','B2','A2'],'ABBA coverage')
    ref=results[0]
    for result in results:matches_historical_profile(result,ref)
    counters=[counter_values(r) for r in results]
    decision={'all1024_ids_equal':True,'all_topology_equal':True,'all_counters_equal':all(x==counters[0] for x in counters),'constant_workload_claim':False,'method':'predeclared default adaptive-profile practical end-to-end test including timing-trained DraftPolicy response; no constant-workload attribution','paired_counter_deltas':[]}
    for ai,bi in [(0,1),(3,2)]:
        decision['paired_counter_deltas'].append({'pair':results[bi]['tag']+'/'+results[ai]['tag'],'candidate_minus_control':{k:counters[bi][k]-counters[ai][k] for k in counters[ai]}})
    for key in ['PP','TG']:
        a=(results[0][key]+results[3][key])/2;b=(results[1][key]+results[2][key])/2
        decision[key]={'control_mean':a,'candidate_mean':b,'gain_pct':100*(b/a-1),'paired_gain_pct':[100*(results[1][key]/results[0][key]-1),100*(results[2][key]/results[3][key]-1)]}
    return decision


def suffix_totals(stderr):
    pattern=r'strata serve: suffix drafts: ([0-9]+) windows, ([0-9]+) of ([0-9]+) drafts accepted'
    totals={'present':False,'windows':0,'accepted':0,'offered':0}
    for line in stderr.splitlines():
        if line.startswith('strata serve: suffix drafts:'):
            match=re.fullmatch(pattern,line);need(match is not None,'malformed suffix totals')
            windows,accepted,offered=map(int,match.groups());need(accepted<=offered,'invalid suffix totals')
            totals['present']=True;totals['windows']+=windows;totals['accepted']+=accepted;totals['offered']+=offered
    return totals

def decode_cost_gate(decision):
    pp=decision['PP'];tg=decision['TG']
    return (decision.get('all1024_ids_equal') is True and decision.get('all_topology_equal') is True
            and tg['gain_pct']>=1 and len(tg['paired_gain_pct'])==2
            and min(tg['paired_gain_pct'])>0 and pp['gain_pct']>=-1)

def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def validate_spec(s):
    from build_qsa_topk_fit17 import SIX
    need(s['base_binary_sha256']==BASE_SHA and s['base_source_sha256']==SIX,'qualified ff34 identity')
    need(s['image']==IMAGE and s['binary_path']==str(O/'strata') and s['build_receipt']==str(O/'build.json'),'model build paths')
    need(s['study_binary_sha256']=='c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049','fixed study binary')
    need('STRATA_VERIFY_PROFILE' not in s['ff34_flags'],'verify profiling disabled')
    need(s['max_abba_sequences']==1 and s['counter_policy']=='report timing-trained DraftPolicy counters; do not require equality','predeclared counter policy')
    need(s['flag']==FLAG and s['trace_flag']==TRACE and s['arm_values']==['0','1','1','0'],'scalar ABBA scope')
    need(s['fixture_sha256']=='63ce3d4042027a00b2193feab143f1d9443b5c31a863a962ec9543ccb54d9695','fixture identity')
    need(s['reference_sha256']=='9a0542235ff1f7d195260b672ff12683775fc0132dbcf25c9c9836a2c30bb6a3','qualified ff34 reference')
    need(canonical_sha(s['ff34_flags'])==FF34_FLAGS_SHA,'unchanged ff34 flags')
    need(canonical_sha(s['native_args'])==NATIVE_ARGS_SHA,'unchanged normalized native args')
    need(FLAG not in s['ff34_flags'] and TRACE not in s['ff34_flags'],'new feature/trace must not be hidden in base flags')
    need(s['ff34_flags'].get('STRATA_GFX906_ATTN_REDUCE12')=='1','reduce12 retained')
    need(s['ff34_flags'].get('STRATA_PRIMARY_COMMIT_OVERLAP')=='1','primary commit retained')
    for k in ['patch_sha256','select_candidate_sha256']:need(digest(s[k]),'source digests')

def validate_gate(g,build,binary_hash,s):
    need(digest(binary_hash) and binary_hash!=BASE_SHA,'explicit newly built binary hash')
    need(build['exit_code']==0 and build['sources_restored'] is True,'full build and restoration')
    need(build['binary_sha256']==g['binary_sha256']==binary_hash,'build/gate/CLI binary identity')
    need(build['base_binary_sha256']==BASE_SHA and build['patch_sha256']==s['patch_sha256'],'base/patch provenance')
    expected=dict(s['base_source_sha256']);expected['src/kernels/cuda/qsa_select.cu']=s['select_candidate_sha256']
    need(build['compiled_sources']==expected,'all seven compiled source digests')
    need(g['select_object_sha256']==build['select_object_sha256'],'resource object identity')
    for k in ['production_scalar_isa_matches_diagnostic','production_control_and_wide_isa_unchanged','resource_gate_passed','diagnostic_parity_passed']:need(g[k] is True,'root production precheck '+k)

def validate_replication_gate(gate,fixed,s,binary_hash):
    need(gate['binary_sha256']==fixed['binary_sha256']==binary_hash==s['study_binary_sha256'],'conditional study binary identity')
    need(gate['methodological_review_approved'] is True,'methodological review required')
    need(gate['counter_policy']==s['counter_policy'] and gate['max_abba_sequences']==1,'predeclared practical test scope')
    need(fixed['benchmark_passed'] is True and fixed['owned_containers_cleaned'] is True,'fixed study completion/cleanup')
    need([r['tag'] for r in fixed['results']]==['A1','B1','B2','A2'],'full fixed ABBA required')
    decision=fixed['decision']
    need(decision['all1024_ids_equal'] is True and decision['all_counters_equal'] is True and decision['passes_cost_gate'] is True,'fixed study exact parity and cost gate')
    need(decision['TG']['gain_pct']>=1 and len(decision['TG']['paired_gain_pct'])==2 and min(decision['TG']['paired_gain_pct'])>0 and decision['PP']['gain_pct']>=-1,'fixed study numeric cost gate')

def execute(s,binary_hash,gate_path,replication_gate_path):
    validate_spec(s);out=O/'adaptive-profile-model';need(not out.exists(),'fresh model output required')
    conditional=json.loads(replication_gate_path.read_text());fixed=json.loads(Path(s['fixed_study_summary_path']).read_text())
    validate_replication_gate(conditional,fixed,s,binary_hash)
    need(conditional['fixed_study_summary_sha256']==sha(s['fixed_study_summary_path']),'exact fixed-study receipt')
    need(not call(['docker','ps','-q']).strip(),'another container active; API must remain off')
    build=json.loads(Path(s['build_receipt']).read_text());g=json.loads(gate_path.read_text());validate_gate(g,build,binary_hash,s);need(binary_hash==s['study_binary_sha256'],'same c183 study binary required')
    need(g['build_receipt_sha256']==sha(s['build_receipt']),'precheck bound to exact build receipt')
    need(sha(s['binary_path'])==binary_hash,'new binary digest')
    need(sha(R/'releases/gfx906-qualified-20261009-reduce12/strata')==BASE_SHA,'base release digest')
    expected=dict(s['base_source_sha256']);expected['src/kernels/cuda/qsa_select.cu']=s['select_candidate_sha256']
    for f,h in expected.items():need(sha(O/'compiled-source'/f)==h,'compiled source snapshot changed')
    need(sha(O/'qsa_select.cu.o')==build['select_object_sha256'],'production object changed')
    need(sha(s['fixture_path'])==s['fixture_sha256'],'fixture bytes');ids=json.loads(Path(s['fixture_path']).read_text())
    need(len(ids)==65536 and all(type(i)is int and 0<=i<248320 for i in ids),'fixture coverage')
    need(sha(s['reference_path'])==s['reference_sha256'],'reference bytes');historical=json.loads(Path(s['reference_path']).read_text())
    need(historical['passed'] and historical['prompt_tokens']==65536 and len(historical['output_ids'])==1024,'historical reference coverage')
    ds=devices();gpu(ds);out.mkdir(mode=0o700);write(out/'spec.sanitized.json',s)
    # Store only explicitly allowed precheck fields, never arbitrary gate contents.
    gatekeys=['binary_sha256','build_receipt_sha256','select_object_sha256','production_scalar_isa_matches_diagnostic','production_control_and_wide_isa_unchanged','resource_gate_passed','diagnostic_parity_passed']
    summary={'results':[],'decision':{},'started_utc':stamp(),'binary_sha256':binary_hash,'base_binary_sha256':BASE_SHA,'build_receipt_sha256':sha(s['build_receipt']),'compiled_source_sha256':expected,'fixture_sha256':s['fixture_sha256'],'reference_sha256':s['reference_sha256'],'historical_default_output_ids':historical['output_ids'],'historical_default_IDs_required':True,'constant_workload_claim':False,'counter_policy':s['counter_policy'],'fixed_study_summary_sha256':sha(s['fixed_study_summary_path']),'methodological_review_approved':True,'workload_profile':s['workload_profile'],'production_prechecks':{k:g[k] for k in gatekeys},'api_touched':False,'promoted':False,'benchmark_passed':False}
    def interrupted(signum,frame):raise InterruptedError('model stop requested')
    old={sig:signal.signal(sig,interrupted) for sig in [signal.SIGINT,signal.SIGTERM]}
    try:
        for tag,value in zip(['A1','B1','B2','A2'],s['arm_values']):
            arm={'binary':s['binary_path'],'binary_sha256':binary_hash,'release_artifacts':{},'flags':{**s['ff34_flags'],FLAG:value},'native_args':s['native_args']}
            result=run_arm(tag,arm,s,ids,ds,out);summary['results'].append(result)
            # Quality/topology stay strict; timing-trained policy counters are measurements.
            matches_historical_profile(result,historical)
            summary['historical_ID_topology_passed_through']=tag;write(out/'summary.json',summary)
        summary['decision']=compare(summary['results']);summary['benchmark_passed']=True
        pp=summary['decision']['PP'];tg=summary['decision']['TG']
        summary['decision']['screen_gate']='direct mean TG >=1%, both TG pairs positive; PP >=-1%; exact IDs/topology; counters reported, not equality-gated'
        summary['decision']['passes_cost_gate']=decode_cost_gate(summary['decision'])
    except BaseException as e:
        summary['error_type']=type(e).__name__;raise
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        failures=[]
        # Do not touch any API container or any name outside this runner's namespace.
        for tag in ['A1','B1','B2','A2']:
            name='strata-fit17-adaptive-'+tag.lower()
            try:
                if call(['docker','ps','-q','--filter','name=^/'+name+'$']).strip():
                    try:call(['docker','stop','--time','30',name],60)
                    except (OSError,subprocess.SubprocessError):call(['docker','kill',name],30)
                    need(not call(['docker','ps','-q','--filter','name=^/'+name+'$']).strip(),'owned container remains')
            except BaseException:failures.append(tag)
        summary['owned_containers_cleaned']=not failures;summary['cleanup_failed_arms']=failures;summary['finished_utc']=stamp();write(out/'summary.json',summary)
        for sig,handler in old.items():signal.signal(sig,handler)
    need(summary['owned_containers_cleaned'],'owned container cleanup failed')
    print(json.dumps({'decision':summary['decision'],'owned_containers_cleaned':True,'api_touched':False}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--binary-sha256');p.add_argument('--production-gate',type=Path);p.add_argument('--replication-gate',type=Path);a=p.parse_args();s=json.loads(a.spec.read_text());validate_spec(s)
    if not a.execute:print(json.dumps({'spec_valid':True,'executed':False}));return
    need(a.binary_sha256 is not None and a.production_gate is not None and a.replication_gate is not None,'explicit binary hash, production gate and conditional replication gate required')
    with open('/tmp/strata-qsa-fit17-qualification.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(s,a.binary_sha256,a.production_gate,a.replication_gate)
if __name__=='__main__':main()
