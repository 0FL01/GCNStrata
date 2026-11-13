#!/usr/bin/env python3
"""Correctness-only 4K/200K checks against ff34; no benchmark gain or API operation. No API start, health check or promotion.
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
    n=len(ids);new=s['requested_outputs']
    need(not call(['docker','ps','-q']).strip(),'unexpected active container')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    need(available>=64*1024**3,'insufficient RAM')
    need(sha(r['binary'])==r['binary_sha256'],'binary changed')
    initial=cooldown(ds,out)
    name='strata-fit17-correctness-'+tag.lower();binary='/src/'+str(Path(r['binary']).relative_to(R))
    cmd=['docker','run','--rm','-i','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--ulimit','memlock=-1:-1','--entrypoint',binary]
    for k,v in sorted(r['flags'].items()):cmd+=['-e',k+'='+v]
    cmd+=['-v',str(R)+':/src:ro','-v','/home/radneon/models-nvme:/models:ro','-v','/home/radneon/models:/sata:ro',IMAGE,'--serve']+r['native_args']
    m={'tag':tag,'release_binary_sha256':r['binary_sha256'],'release_artifacts':r['release_artifacts'],'flags':r['flags'],'native_args':r['native_args'],'fixture_sha256':s['fixture_sha256'],'prompt_tokens':n,'requested_outputs':new,'sampling':{'temperature':0,'top_p':1,'top_k':1,'seed':12345},'started_utc':stamp(),'gpu_before':initial,'fresh_process':True}
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
                    p.stdin.write('GEN '+str(new)+' temperature=0 top_p=1 top_k=1 seed=12345 '+','.join(map(str,ids))+'\n');p.stdin.flush()
                elif line.startswith('T '):tokens.append(int(line.split()[1]));need(len(tokens)<=new,'too many outputs')
                elif line.startswith('DONE '):
                    # Persist raw protocol and complete received IDs BEFORE parsing or validation.
                    m['private_native_forensics']=capture_native_done(private,tag,line,tokens)
                    write(out/(tag+'.manifest.json'),m)
                    need(request_start is not None,'DONE before READY');done=line.split();m['request_wall_s']=time.monotonic()-request_start
            p.stdin.write('QUIT\n');p.stdin.flush();p.stdin.close();code=p.wait(timeout=120);t.join(timeout=5);need(code==0,'native exit')
            err.flush();stderr=os.pread(err.fileno(),os.fstat(err.fileno()).st_size,0).decode(errors='replace');topology=fit17_topology(stderr);need(bool(topology),'FIT17 host topology trace required');m['fit17_host_topology']=topology;need('strata attn-reduce12: selected=1 ' not in stderr,'unexpected performance trace')
            suffix=suffix_totals(stderr);m['suffix_drafts']=suffix
            if r['flags'].get('STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE')=='1':need('attn-query-swizzle: active=1 ' in stderr,'candidate QSA route absent')
        need(len(done)>=15 and int(done[1])==len(tokens)==new and int(done[2])==n and done[5]=='length','DONE output coverage')
        need(int(done[8])==0 and int(done[14])==n,'prompt reuse/read')
        need(all(0<=x<248320 for x in tokens),'output ID domain')
        pm,dm=map(float,done[3:5]);need(math.isfinite(pm) and math.isfinite(dm) and pm>0 and dm>0,'durations')
        extended=native_counter_fields(done)
        result={'tag':tag,'passed':True,'prompt_tokens':n,'actual_outputs':new,'prompt_ms':pm,'decode_ms':dm,'PP':n*1000/pm,'TG':new*1000/dm,'accepted':int(done[6]),'offered':int(done[7]),'cache_hits':int(done[9]),'lookups':int(done[10]),'prompt_reused':int(done[8]),'prompt_read':int(done[14]),'output_ids':tokens,**extended,'engine_info':info,'suffix_drafts':suffix,'fit17_host_topology':topology,'timing_scope':'unpaired diagnostic timing only; no performance comparison','request_wall_s':m['request_wall_s'],'load_wall_s':m['load_wall_s']}
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

def capture_native_done(private,tag,line,tokens):
    files={'raw_done':(private/(tag+'.done.txt'),line),'output_ids':(private/(tag+'.output-ids.json'),json.dumps(tokens)+'\n')}
    result={}
    for label,(path,text) in files.items():
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        result[label]={'relative_path':'private/'+path.name,'sha256':sha(path)}
    return result

def native_counter_fields(done):
    # Qualified e7a5: base16. Five tail counters ONLY with explicit lookup_chain>0.
    # INFO lookup reports suffix_draft and must not determine this protocol length.
    need(len(done) in (16,21),'native DONE must have16 base or21 chain fields')
    values={name:int(done[i]) for i,name in {11:'ram_blobs',12:'file_blobs',15:'offloaded_experts'}.items()}
    values['file_read_mb']=float(done[13])
    need(all(math.isfinite(v) and v>=0 for v in values.values()),'native counter domain')
    optional={16:'lookup_chain_accepted',17:'lookup_chain_offered',18:'native_suffix_accepted',19:'native_suffix_offered',20:'decode_windows'}
    values.update({name:int(done[i]) if len(done)==21 else None for i,name in optional.items()})
    values['optional_chain_counters_available']=len(done)==21
    values['native_done_field_count']=len(done)
    if len(done)==21:
        need(all(values[k]>=0 for k in optional.values()),'optional native counter domain')
        need(values['lookup_chain_accepted']<=values['lookup_chain_offered'] and values['native_suffix_accepted']<=values['native_suffix_offered'],'native draft counter domain')
    return values

def suffix_totals(stderr):
    pattern=r'strata serve: suffix drafts: ([0-9]+) windows, ([0-9]+) of ([0-9]+) drafts accepted'
    totals={'present':False,'windows':0,'accepted':0,'offered':0}
    for line in stderr.splitlines():
        if line.startswith('strata serve: suffix drafts:'):
            match=re.fullmatch(pattern,line);need(match is not None,'malformed suffix totals')
            windows,accepted,offered=map(int,match.groups());need(accepted<=offered,'invalid suffix totals')
            totals['present']=True;totals['windows']+=windows;totals['accepted']+=accepted;totals['offered']+=offered
    return totals

def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def validate_gate(g,build,binary_hash,s):
    need(digest(binary_hash) and binary_hash!=BASE_SHA,'explicit newly built binary hash')
    need(build['exit_code']==0 and build['sources_restored'] is True,'full build and restoration')
    need(build['binary_sha256']==g['binary_sha256']==binary_hash,'build/gate/CLI binary identity')
    need(build['base_binary_sha256']==BASE_SHA and build['patch_sha256']==s['patch_sha256'],'base/patch provenance')
    expected=dict(s['base_source_sha256']);expected['src/kernels/cuda/qsa_select.cu']=s['select_candidate_sha256']
    need(build['compiled_sources']==expected,'all seven compiled source digests')
    need(g['select_object_sha256']==build['select_object_sha256'],'resource object identity')
    for k in ['production_scalar_isa_matches_diagnostic','production_control_and_wide_isa_unchanged','resource_gate_passed','diagnostic_parity_passed']:need(g[k] is True,'root production precheck '+k)


def fit17_topology(stderr):
    pattern=r'strata gfx906 topk fit17: enabled=1 topology=guarded-ref256/reg17/reg66 max_blocks=17408 nq=([0-9]+) capacity_blocks=([0-9]+) cap=([0-9]+) \(device steps select the per-window arm\)'
    rows=[]
    for line in stderr.splitlines():
        if line.startswith('strata gfx906 topk fit17:'):
            match=re.fullmatch(pattern,line);need(match is not None,'FIT17 trace format')
            nq,capacity,cap=map(int,match.groups());need(1<=nq<=8 and capacity>=17408 and cap>0,'guarded topology geometry')
            rows.append({'nq':nq,'capacity_blocks':capacity,'cap':cap})
    return rows

def inferred_decode_fallback(prompt_tokens,generate):
    # Conservative logical-token range includes the last prompt token and a full8-row window.
    low=prompt_tokens-1;high=prompt_tokens+generate+8
    block_low=low//4+1;block_high=high//4+1
    if high<=24576:owner='guarded-reference256'
    elif low>24576 and block_low>17408:owner='guarded-register66-wide'
    else:raise RuntimeError('case does not establish one fallback owner by source bounds')
    return {'expected_decode_owner':owner,'inferred_not_instrumented':True,'runtime_branch_counter_available':False,'conservative_n_kv_bounds':[low,high],'max_bid_plus_one_bounds':[block_low,block_high],'n_bid_formula':'floor(n_kv/4)','short_cutoff_cells':24576,'fit17_capacity_blocks':17408,'source_sha256':'8d836acd211b7e493cd50bd680d128489f3f25f94ea26725afbab2402ef899bc','evidence':'known full-prefix case plus reviewed step construction/device guards and production wide/control ISA gate; host trace shows only launch topology'}

def validate_spec(s):
    from build_qsa_topk_fit17 import SIX
    need(s['base_binary_sha256']==BASE_SHA and s['base_source_sha256']==SIX,'ff34 source provenance')
    need(s['study_binary_sha256']=='c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049','qualified candidate identity')
    need(s['image']==IMAGE and s['binary_path']==str(O/'strata') and s['build_receipt']==str(O/'build.json'),'build identity')
    need(canonical_sha(s['ff34_flags'])==FF34_FLAGS_SHA and canonical_sha(s['native_args'])==NATIVE_ARGS_SHA,'original default profile')
    need(FLAG not in s['ff34_flags'] and TRACE not in s['ff34_flags'] and 'STRATA_VERIFY_PROFILE' not in s['ff34_flags'],'no hidden feature/profile flags')
    need(s['reference_sha256']=='13b8fd8b67d24cc65a027c990281f8d19696d0b77a72a75316638a438737fb9d','reference extract digest')
    need([(c['tag'],c['prompt_tokens'],c['generate'],c['fixture_sha256']) for c in s['cases']]==[('code4096',4096,1024,'6709011b1777b4b847dff88d3eab088d166f763c478f35c2eb42edb6bca4b058'),('code200000',200000,256,'736636e97ede94cc37e32447eefde4217a38bd17c81a7349345be98060e766c4')],'exact existing fixtures/cases')

def quality_check(result,reference):
    need(result['passed'] and result['prompt_tokens']==reference['prompt_tokens'] and result['actual_outputs']==reference['actual_outputs'],'completed case')
    need(result['output_ids']==reference['output_ids'],'full ff34 output ID mismatch')
    for k in TOPOLOGY_KEYS:need(result['engine_info'][k]==reference['engine_info'][k],'qualified topology mismatch '+k)
    need(bool(result['fit17_host_topology']),'guarded FIT17 topology was not enabled')
    result['all_reference_ids_equal']=True
    result['fallback_source_inference']=inferred_decode_fallback(result['prompt_tokens'],result['actual_outputs'])
    # Adaptive draft/cache counters are retained, not equality-gated.
    result['counter_policy']='report numeric native/suffix counters; default policy is timing-trained'

def execute(s,binary_hash,gate_path):
    validate_spec(s);out=O/'correctness-4k-200k';need(not out.exists(),'fresh correctness output required')
    need(not call(['docker','ps','-q']).strip(),'another container active; API must remain off')
    build=json.loads(Path(s['build_receipt']).read_text());gate=json.loads(gate_path.read_text());validate_gate(gate,build,binary_hash,s)
    need(binary_hash==s['study_binary_sha256'] and sha(s['binary_path'])==binary_hash,'same c183 candidate bytes')
    need(gate['build_receipt_sha256']==sha(s['build_receipt']) and sha(O/'qsa_select.cu.o')==build['select_object_sha256'],'production receipt/object provenance')
    expected={**s['base_source_sha256'],'src/kernels/cuda/qsa_select.cu':s['select_candidate_sha256']}
    for f,h in expected.items():need(sha(O/'compiled-source'/f)==h,'compiled source snapshot changed')
    need(sha(s['reference_path'])==s['reference_sha256'],'reference extract changed');refs=json.loads(Path(s['reference_path']).read_text())
    need(refs['binary_sha256']==BASE_SHA,'reference binary identity')
    fixtures={}
    for case in s['cases']:
        need(sha(case['fixture_path'])==case['fixture_sha256'],'fixture bytes')
        ids=json.loads(Path(case['fixture_path']).read_text());need(len(ids)==case['prompt_tokens'] and all(type(i)is int and 0<=i<248320 for i in ids),'fixture coverage');fixtures[case['tag']]=ids
        ref=refs['results'][str(case['prompt_tokens'])];need(ref['passed'] and len(ref['output_ids'])==case['generate'],'reference coverage')
    ds=devices();gpu(ds);out.mkdir(mode=0o700);write(out/'spec.sanitized.json',s)
    summary={'results':[],'qualification_passed':False,'started_utc':stamp(),'binary_sha256':binary_hash,'base_binary_sha256':BASE_SHA,'compiled_source_sha256':expected,'reference_sha256':s['reference_sha256'],'reference_source_sha256':refs['source_sha256'],'build_receipt_sha256':sha(s['build_receipt']),'performance_comparison':False,'api_touched':False,'promoted':False,'trace_scope':'correctness-only host topology; not a device branch counter'}
    def interrupted(signum,frame):raise InterruptedError('correctness stop requested')
    old={sig:signal.signal(sig,interrupted) for sig in [signal.SIGINT,signal.SIGTERM]}
    try:
        for case in s['cases']:
            arm={'binary':s['binary_path'],'binary_sha256':binary_hash,'release_artifacts':{},'flags':{**s['ff34_flags'],FLAG:'1',TRACE:'1'},'native_args':s['native_args']}
            case_spec={**s,'fixture_sha256':case['fixture_sha256'],'requested_outputs':case['generate']}
            result=run_arm(case['tag'],arm,case_spec,fixtures[case['tag']],ds,out);summary['results'].append(result)
            quality_check(result,refs['results'][str(case['prompt_tokens'])]);write(out/(case['tag']+'.result.json'),result);write(out/'summary.json',summary)
        summary['qualification_passed']=True
    except BaseException as e:summary['error_type']=type(e).__name__;raise
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        failures=[]
        for case in s['cases']:
            name='strata-fit17-correctness-'+case['tag'].lower()
            try:
                if call(['docker','ps','-q','--filter','name=^/'+name+'$']).strip():
                    try:call(['docker','stop','--time','30',name],60)
                    except (OSError,subprocess.SubprocessError):call(['docker','kill',name],30)
                    need(not call(['docker','ps','-q','--filter','name=^/'+name+'$']).strip(),'owned container remains')
            except BaseException:failures.append(case['tag'])
        summary['owned_containers_cleaned']=not failures;summary['cleanup_failed_cases']=failures;summary['finished_utc']=stamp();write(out/'summary.json',summary)
        for sig,handler in old.items():signal.signal(sig,handler)
    need(summary['owned_containers_cleaned'],'owned container cleanup failed')
    print(json.dumps({'qualification_passed':True,'cases':[{'tag':r['tag'],'ids_checked':r['actual_outputs'],'all_reference_ids_equal':True,'expected_fallback':r['fallback_source_inference']['expected_decode_owner']} for r in summary['results']],'performance_comparison':False,'api_touched':False}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--binary-sha256');p.add_argument('--production-gate',type=Path);a=p.parse_args();s=json.loads(a.spec.read_text());validate_spec(s)
    if not a.execute:print(json.dumps({'spec_valid':True,'executed':False}));return
    need(a.binary_sha256 is not None and a.production_gate is not None,'explicit binary identity and production gate required')
    with open('/tmp/strata-qsa-fit17-qualification.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(s,a.binary_sha256,a.production_gate)
if __name__=='__main__':main()
