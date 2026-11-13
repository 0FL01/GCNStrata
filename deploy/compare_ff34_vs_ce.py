#!/usr/bin/env python3
"""Prepare/check, or explicitly execute, direct frozen-release ABBA. No promotion.
Default --check-only is offline: validates the supplied sanitized spec, not MI50 state.
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

R=Path('/home/radneon/strata')
EXPECTED={'A':'ce794788d64313236bbc24b47a3f8e06c841034afaf4e19a2fc5ffd84e337f14','B':'ff348b38ad9b531e8df701311b1ae736f1378a6aaef322eeaa86e78f685917d9'}
IMAGE='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
FLAGS=set('STRATA_PRIMARY_COMMIT_OVERLAP STRATA_GFX906_ATTN_REDUCE12 STRATA_GFX906_ATTN_REDUCE12_TRACE STRATA_GFX906_ATTN_QUERY_SWIZZLE STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE STRATA_VERIFY_SKIP_EMPTY_PCIE STRATA_ATTN_LANECELL STRATA_EXP_MODE STRATA_MMVF_ROWS STRATA_HIP_SWIGLU_FUSED STRATA_GDN_HEAD STRATA_GDN_PP STRATA_GDN_CONVL2 STRATA_GDN_NOY STRATA_TSUM STRATA_Q2_AVX2_SPREAD STRATA_GFX906_TOPK_REG STRATA_GR_UP_EXACT STRATA_MMVF_RPB STRATA_GFX906_HC_F16 STRATA_HC_F16_AUDIT STRATA_HC_F16_SCALE_PROBE STRATA_HC_F16_EXACT_INPUT STRATA_HC_F16_TRACE STRATA_PREFILL_MMQ STRATA_GFX906_MMQ_OPT_CAP STRATA_MMQ_OPT_TRACE HIP_VISIBLE_DEVICES STRATA_STAGE_TRIM STRATA_ARENA_MMAP DO_NOT_TRACK HF_HUB_DISABLE_TELEMETRY'.split())
NATIVE=set('--pack --native --ple-gguf --expert-profile --expert-cache --prefill --spec --spec-min-p --mtp --max-context --kv --kv-resident --vram-reserve-mib --pcie-frac --ple-io --conversation-cache-mib --conversation-cache-slots --conversation-cache-min-free-mib --mtp-window --vram-reserve-later-mib --adapt-every --layer-split --prompt-cache --prompt-cache-every --eos-ids'.split())
FIXED={'--expert-cache':'9900','--max-context':'204800','--kv':'int8','--kv-resident':'32768','--pcie-frac':'0','--ple-io':'ram','--adapt-every':'0','--layer-split':'27','--prompt-cache':'0','--prompt-cache-every':'0','--eos-ids':'2147483647'}

def need(ok,label):
    if not ok:raise RuntimeError(label)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def call(args,timeout=120):
    return subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,timeout=timeout).stdout

def validate_spec(s):
    need(s['image']==IMAGE,'image identity')
    need(re.fullmatch('[0-9a-f]{64}',s['fixture_sha256']) is not None,'fixture SHA256 required')
    need(set(s['releases'])=={'A','B'},'two releases required')
    need('STRATA_PRIMARY_COMMIT_OVERLAP' not in s['releases']['A']['flags'] and 'STRATA_GFX906_ATTN_REDUCE12' not in s['releases']['A']['flags'],'new opt-ins must be absent from stable')
    for arm,r in s['releases'].items():
        need(r['binary_sha256']==EXPECTED[arm],'release binary identity')
        need(Path(r['binary']).parent.name==('gfx906-exact-20261008' if arm=='A' else 'gfx906-qualified-20261009-reduce12'),'immutable release path')
        need(set(r['flags'])<=FLAGS and all(isinstance(v,str) and re.fullmatch(r'[0-9,.-]+',v) for v in r['flags'].values()),'unapproved env data')
        need(r['flags'].get('STRATA_GFX906_ATTN_REDUCE12_TRACE','0')!='1','reduce12 trace must be off in performance arms')
        need(r['flags'].get('HIP_VISIBLE_DEVICES')=='0,1','GPU placement')
        args=r['native_args'];need(len(args)%2==0,'paired native args')
        pairs=dict(zip(args[::2],args[1::2]));need(len(pairs)*2==len(args) and set(pairs)<=NATIVE,'duplicate/unsupported native option')
        need(all(pairs.get(k)==v for k,v in FIXED.items()),'fixed workload configuration')
        for k in ['--pack','--native','--ple-gguf','--expert-profile','--mtp']:need(pairs.get(k,'').startswith(('/src/','/models/','/sata/')),'model input path')
        need(isinstance(r['release_artifacts'],dict),'release artifact provenance type')
        for path,digest in r['release_artifacts'].items():
            need(Path(path).is_relative_to(Path(r['binary']).parent) and re.fullmatch('[0-9a-f]{64}',digest),'release artifact location/hash')
    a=dict(zip(s['releases']['A']['native_args'][::2],s['releases']['A']['native_args'][1::2]));b=dict(zip(s['releases']['B']['native_args'][::2],s['releases']['B']['native_args'][1::2]))
    need(a==b,'native workload configuration differs between releases')
    need(s['releases']['B']['flags'].get('STRATA_GFX906_ATTN_REDUCE12')=='1','candidate reduce12 enabled')
    need(s['releases']['B']['flags'].get('STRATA_PRIMARY_COMMIT_OVERLAP')=='1','candidate primary commit retained')

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

def api_identity():
    # No environ, config, .env, command line, or credentials are exported.
    code="from pathlib import Path;import hashlib,json,os\nwanted='"+EXPECTED['A']+"';found=[]\nfor p in Path('/proc').glob('[0-9]*'):\n try:\n  exe=os.readlink(p/'exe')\n  if Path(exe).name.startswith('strata'):\n   digest=hashlib.sha256((p/'exe').read_bytes()).hexdigest()\n   if digest==wanted:found.append(digest)\n except OSError:pass\nprint(json.dumps(found))"
    found=json.loads(call(['docker','exec','strata-api','python3','-c',code],120));need(found==[EXPECTED['A']],'live API stable binary identity')
    return {'binary_sha256':EXPECTED['A'],'matching_processes':1}

def health():
    with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
    need(h.get('loaded') and h.get('images') and h.get('api_key') and h.get('max_context')==204800,'API health incomplete')
    return {'loaded':True,'images':True,'authentication_enabled':True,'max_context':204800}

def run_arm(tag,r,s,ids,ds,out):
    need(not call(['docker','ps','-q']).strip(),'unexpected active container')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    need(available>=64*1024**3,'insufficient RAM')
    need(sha(r['binary'])==r['binary_sha256'],'binary changed')
    initial=cooldown(ds,out)
    name='strata-direct-ff34-ce-'+tag.lower();binary='/src/'+str(Path(r['binary']).relative_to(R))
    cmd=['docker','run','--rm','-i','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--ulimit','memlock=-1:-1','--entrypoint',binary]
    for k,v in sorted(r['flags'].items()):cmd+=['-e',k+'='+v]
    cmd+=['-v',str(R)+':/src:ro','-v','/home/radneon/models-nvme:/models:ro','-v','/home/radneon/models:/sata:ro',IMAGE,'--serve']+r['native_args']
    m={'tag':tag,'release_binary_sha256':r['binary_sha256'],'release_artifacts':r['release_artifacts'],'flags':r['flags'],'native_args':r['native_args'],'fixture_sha256':s['fixture_sha256'],'sampling':{'temperature':0,'top_p':1,'top_k':1,'seed':12345},'started_utc':stamp(),'gpu_before':initial,'fresh_process':True}
    write(out/(tag+'.manifest.json'),m);print(json.dumps({'event':'start','tag':tag}),flush=True)
    p=None;tokens=[];info={};samples=[];started=time.monotonic();ready=None;request_start=None;done=None;q=queue.Queue()
    try:
        # Temporary native stderr is used only for PLE/route checks, then deleted.
        with tempfile.TemporaryFile(mode='w+t') as err:
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
                    for k,v in {'expert_slots':'19078','expert_slots_primary':'9900','kv':'int8','context':'204800','lookup':'3','mtp_max':'4','kv_resident':'32768','pcie_frac':'0.00'}.items():need(info.get(k)==v,'engine configuration '+k)
                    err.flush();need('locked in RAM' in os.pread(err.fileno(),os.fstat(err.fileno()).st_size,0).decode(errors='replace'),'PLE RAM lock missing')
                    ready=time.monotonic();m['load_wall_s']=ready-started;m['engine_info']=info;request_start=time.monotonic()
                    p.stdin.write('GEN 1024 temperature=0 top_p=1 top_k=1 seed=12345 '+','.join(map(str,ids))+'\n');p.stdin.flush()
                elif line.startswith('T '):tokens.append(int(line.split()[1]));need(len(tokens)<=1024,'too many outputs')
                elif line.startswith('DONE '):need(request_start is not None,'DONE before READY');done=line.split();m['request_wall_s']=time.monotonic()-request_start
            p.stdin.write('QUIT\n');p.stdin.flush();p.stdin.close();code=p.wait(timeout=120);t.join(timeout=5);need(code==0,'native exit')
            err.flush();stderr=os.pread(err.fileno(),os.fstat(err.fileno()).st_size,0).decode(errors='replace');need('strata attn-reduce12: selected=1 ' not in stderr,'unexpected performance trace')
            if tag.startswith('B') and r['flags'].get('STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE')=='1':need('attn-query-swizzle: active=1 ' in stderr,'candidate QSA route absent')
        need(len(done)>=15 and int(done[1])==len(tokens)==1024 and int(done[2])==65536 and done[5]=='length','DONE output coverage')
        need(int(done[8])==0 and int(done[14])==65536,'prompt reuse/read')
        need(all(0<=x<248320 for x in tokens),'output ID domain')
        pm,dm=map(float,done[3:5]);need(math.isfinite(pm) and math.isfinite(dm) and pm>0 and dm>0,'durations')
        result={'tag':tag,'passed':True,'prompt_tokens':65536,'actual_outputs':1024,'prompt_ms':pm,'decode_ms':dm,'PP':65536000/pm,'TG':1024000/dm,'accepted':int(done[6]),'offered':int(done[7]),'cache_hits':int(done[9]),'lookups':int(done[10]),'prompt_reused':int(done[8]),'prompt_read':int(done[14]),'output_ids':tokens,'engine_info':info,'request_wall_s':m['request_wall_s'],'load_wall_s':m['load_wall_s']}
        write(out/(tag+'.result.json'),result);return result
    finally:
        if p is not None and p.poll() is None:
            try:call(['docker','stop','--time','30',name],60)
            except (subprocess.SubprocessError,OSError):
                try:call(['docker','kill',name],30)
                except (subprocess.SubprocessError,OSError):pass
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        m.update(finished_utc=stamp(),elapsed_s=time.monotonic()-started,native_exit=p.returncode if p else None)
        write(out/(tag+'.manifest.json'),m);write(out/(tag+'.gpu-samples.json'),samples)

def compare(results):
    need([r['tag'] for r in results]==['A1','B1','B2','A2'],'ABBA coverage')
    ref=results[0]
    for r in results:
        need(r['output_ids']==ref['output_ids'],'full output ID mismatch')
        for k in ['accepted','offered','cache_hits','lookups','prompt_reused','prompt_read']:need(r[k]==ref[k],'counter mismatch '+k)
        for k in ['expert_slots','expert_slots_primary','context','kv_resident','kv','spec','mtp_max','lookup','pcie_frac']:need(r['engine_info'][k]==ref['engine_info'][k],'engine mismatch '+k)
    decision={'all1024_ids_equal':True,'all_counters_equal':True,'method':'direct fresh-process ABBA; no sum of isolated gains'}
    for k in ['PP','TG']:
        a=(results[0][k]+results[3][k])/2;b=(results[1][k]+results[2][k])/2
        decision[k]={'stable_mean':a,'candidate_mean':b,'gain_pct':100*(b/a-1),'paired_gain_pct':[100*(results[1][k]/results[0][k]-1),100*(results[2][k]/results[3][k]-1)]}
    return decision

def execute(s,out):
    need(not out.exists(),'output directory already exists');need(call(['docker','ps','--format','{{.Names}}']).split()==['strata-api'],'exclusive API state required')
    need(call(['docker','inspect','--format','{{.Image}}','strata-api']).strip()==IMAGE,'API container image changed')
    before=api_identity();health();ds=devices();gpu(ds)
    protected={'config':R/'configs/shefowl-hybrid-200k.json','compose':R/'compose.yaml'}
    protected_before={k:sha(path) for k,path in protected.items()}
    for r in s['releases'].values():
        need(sha(r['binary'])==r['binary_sha256'],'preflight release binary hash')
        for path,digest in r['release_artifacts'].items():need(sha(path)==digest,'preflight release artifact hash')
    need(sha(s['fixture_path'])==s['fixture_sha256'],'fixture hash');ids=json.loads(Path(s['fixture_path']).read_text());need(len(ids)==65536 and all(type(i)is int and 0<=i<248320 for i in ids),'fixture IDs')
    out.mkdir(parents=True,mode=0o700);write(out/'spec.sanitized.json',s)
    results=[];summary={'results':results,'decision':{},'production_restore':{'passed':False},'promoted':False,'started_utc':stamp(),'configuration_sha256_before':protected_before};armed=False
    def stop_signal(signum,frame):raise InterruptedError('requested signal stop')
    old={sig:signal.signal(sig,stop_signal) for sig in [signal.SIGINT,signal.SIGTERM]}
    try:
        armed=True;call(['docker','stop','--timeout','120','strata-api'],150)
        for tag in ['A1','B1','B2','A2']:
            r=run_arm(tag,s['releases'][tag[0]],s,ids,ds,out);results.append(r)
            if len(results)>1:
                need(r['output_ids']==results[0]['output_ids'],'full output mismatch')
                for k in ['accepted','offered','cache_hits','lookups','prompt_reused','prompt_read']:need(r[k]==results[0][k],'counter mismatch')
            write(out/'summary.json',summary)
        summary['decision']=compare(results);summary['benchmark_passed']=True
    except BaseException as e:
        summary['benchmark_passed']=False;summary['error_type']=type(e).__name__;raise
    finally:
        # A second termination signal must not interrupt restoration.
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        if armed:
            try:
                call(['docker','start','strata-api'],120)
                deadline=time.monotonic()+600
                while True:
                    try:
                        h=health();identity=api_identity();break
                    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):
                        need(time.monotonic()<deadline,'stable API restoration timeout');time.sleep(5)
                protected_after={k:sha(path) for k,path in protected.items()}
                unchanged=protected_after==protected_before
                summary['configuration_sha256_after']=protected_after
                summary['production_restore']={'passed':unchanged,'health':h,'identity':identity,'same_stable_binary':identity==before,'config_and_compose_changed':not unchanged,'configuration_hashes_equal':unchanged}
            except BaseException as e:summary['production_restore']={'passed':False,'error_type':type(e).__name__}
        summary['finished_utc']=stamp();write(out/'summary.json',summary)
        for sig,handler in old.items():signal.signal(sig,handler)
    need(summary['production_restore']['passed'],'stable API restoration failed')
    print(json.dumps({'event':'complete','decision':summary['decision'],'production_restore':summary['production_restore']}),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('spec',type=Path);ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path,default=R/'logs/perf-campaign-20261009/direct-ff34-vs-ce');a=ap.parse_args();s=json.loads(a.spec.read_text());validate_spec(s)
    if not a.execute:print(json.dumps({'spec_valid':True,'executed':False}));return
    with open('/tmp/strata-direct-ff34-ce.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(s,a.output)
if __name__=='__main__':main()
