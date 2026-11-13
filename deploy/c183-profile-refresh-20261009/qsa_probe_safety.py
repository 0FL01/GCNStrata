#!/usr/bin/env python3
"""Root-run isolated public-QSA component test; no full-tree build claim.
Default checks syntax/arguments only. --execute is required for MI50 actions.
"""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
R=Path('/home/radneon/strata')
IMAGE='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
EXPECTED={'A':'ce794788d64313236bbc24b47a3f8e06c841034afaf4e19a2fc5ffd84e337f14'}
SOURCE_SHA='ec0a11258848d6e2bf6ca0bbe463f06eada460a72be10ff5bb4dce065abe3ab0'
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

def api_identity():
    # No environ, config, .env, command line, or credentials are exported.
    code="from pathlib import Path;import hashlib,json,os\nwanted='"+EXPECTED['A']+"';found=[]\nfor p in Path('/proc').glob('[0-9]*'):\n try:\n  exe=os.readlink(p/'exe')\n  if Path(exe).name.startswith('strata'):\n   digest=hashlib.sha256((p/'exe').read_bytes()).hexdigest()\n   if digest==wanted:found.append(digest)\n except OSError:pass\nprint(json.dumps(found))"
    found=json.loads(call(['docker','exec','strata-api','python3','-c',code],120));need(found==[EXPECTED['A']],'live API stable binary identity')
    return {'binary_sha256':EXPECTED['A'],'matching_processes':1}

def health():
    with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
    need(h.get('loaded') and h.get('images') and h.get('api_key') and h.get('max_context')==204800,'API health incomplete')
    return {'loaded':True,'images':True,'authentication_enabled':True,'max_context':204800}

def check_manifest(input_dir):
    for line in (input_dir/'public-build-inputs.sha256').read_text().splitlines():
        digest,rel=line.split('  ',1)
        path=(input_dir/rel).resolve()
        need(path.is_relative_to(input_dir.resolve()),'manifest path outside input')
        need(sha(path)==digest,'public input hash mismatch: '+rel)

def cleanup(name):
    active=call(['docker','ps','-q','--filter','name=^/'+name+'$']).strip()
    if active:
        try:call(['docker','stop','--time','15',name],30)
        except (subprocess.SubprocessError,OSError):call(['docker','kill',name],30)
    need(not call(['docker','ps','-q','--filter','name=^/'+name+'$']).strip(),'probe container still active')

def run_gpu(index,a,ds,out):
    name='strata-public-reduce12-gpu'+str(index)
    need(not call(['docker','ps','-q']).strip(),'another container became active')
    need(sha(a.build/'gfx906_qsa_query_swizzle_probe')==a.expected_binary_sha,'binary changed')
    pre=cooldown(ds,out)
    binary='/src/'+str((a.build/'gfx906_qsa_query_swizzle_probe').relative_to(R))
    cmd=['docker','run','--rm','--name',name,'--network','none',
         '--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host',
         '--entrypoint','/bin/bash','-v',str(R)+':/src:ro',
         '-v',str(a.input)+':/input:ro','-v',str(out)+':/out',IMAGE,
         '/input/run-existing-probe.sh',binary,'/out/gpu'+str(index),str(index)]
    started=time.monotonic();p=None;samples=[]
    row={'gpu_ordinal':index,'started_utc':stamp(),'binary_sha256':a.expected_binary_sha,
         'gpu_before':pre,'time_limit_s':600,'container':name,'command':cmd}
    try:
        with (out/f'gpu{index}.driver.stdout').open('w') as stdout, (out/f'gpu{index}.driver.stderr').open('w') as stderr:
            p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr)
            while p.poll() is None:
                need(time.monotonic()-started<600,'GPU probe timeout')
                need(not (out/'cancel').exists(),'cancellation marker')
                samples.append({'elapsed_s':time.monotonic()-started,'gpus':gpu(ds)})
                write(out/f'gpu{index}.thermal.json',samples)
                time.sleep(2)
            need(p.returncode==0,'probe wrapper exit nonzero')
        d=out/f'gpu{index}'
        need((d/'binary.sha256').read_text().split()[0]==a.expected_binary_sha,'runtime probe hash')
        need('selected=1 query_swizzle=1 kv_mode=1' in (d/'route-smoke.stderr').read_text(),'actual reduce12 route absent')
        arm_names=['A1','B1','B2','A2']
        logs=[(d/(n+'.stdout')).read_text() for n in arm_names]
        need(len({re.search(r'INPUT_FNV1A64=([0-9a-f]+)',t).group(1) for t in logs})==1,'different ABBA input hashes')
        need((d/'A1.bin').stat().st_size==786432,'incomplete full output')
        for n in ['B1','B2','A2','route-smoke']:need((d/(n+'.bin')).read_bytes()==(d/'A1.bin').read_bytes(),'full output mismatch')
        for prefix,size in [('edge1',24576),('edge33',811008)]:
            need((d/(prefix+'-A.bin')).stat().st_size==size,'edge dump size')
            need((d/(prefix+'-A.bin')).read_bytes()==(d/(prefix+'-B.bin')).read_bytes(),'edge full-output mismatch')
        result=json.loads((d/'abba-summary.json').read_text())
        need(result['component_latency_reduction_pct']>=12,'component latency gate below 12%')
        row.update(passed=True,measurement=result)
        return row
    finally:
        try:cleanup(name)
        finally:
            if p is not None:
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            row.update(finished_utc=stamp(),elapsed_s=time.monotonic()-started,exit_code=p.returncode if p else None)
            write(out/f'gpu{index}.driver.json',row);write(out/f'gpu{index}.thermal.json',samples)

def execute(a):
    out=a.output
    need(not out.exists(),'output directory already exists')
    need(a.build.resolve().is_relative_to(R),'build directory must lie beneath R')
    need(call(['docker','ps','--format','{{.Names}}']).split()==['strata-api'],'exclusive stable API container state required')
    need(call(['docker','inspect','--format','{{.Image}}','strata-api']).strip()==IMAGE,'API image mismatch')
    check_manifest(a.input)
    need(sha(a.input/'src/kernels/cuda/qsa_decode_attn.cu')==SOURCE_SHA,'public source identity')
    need((a.build/'build.exit_code').read_text().strip()=='0' and (a.build/'BUILD-PASSED').read_text().strip()=='compile_link_symbol_checks_passed','compile gates missing')
    need(sha(a.build/'gfx906_qsa_query_swizzle_probe')==a.expected_binary_sha,'public binary identity')
    before=api_identity();h=health();ds=devices();gpu(ds)
    out.mkdir(mode=0o700)
    summary={'scope':'isolated exact-public QSA TU plus unchanged public probe and frozen support archive; not a full public-tree build',
             'started_utc':stamp(),'source_sha256':SOURCE_SHA,'binary_sha256':a.expected_binary_sha,
             'compile_receipts':{f:sha(a.build/f) for f in ['build.exit_code','outputs.sha256','public-probe.map','archive-before.sha256','archive-after.sha256']},
             'api_before':{'identity':before,'health':h},'results':[],'production_restore':{'passed':False},'promoted':False}
    write(out/'summary.json',summary)
    armed=False;failure=None
    def stop_signal(signum,frame):raise InterruptedError('requested signal stop')
    old={s:signal.signal(s,stop_signal) for s in [signal.SIGINT,signal.SIGTERM]}
    try:
        armed=True;call(['docker','stop','--timeout','120','strata-api'],150)
        for index in [0,1]:
            summary['results'].append(run_gpu(index,a,ds,out));write(out/'summary.json',summary)
        for tag in ['A1','B1','B2','A2','edge1-A','edge1-B','edge33-A','edge33-B']:
            need((out/'gpu0'/(tag+'.bin')).read_bytes()==(out/'gpu1'/(tag+'.bin')).read_bytes(),'cross-GPU full output mismatch')
        summary.update(benchmark_passed=True,cross_gpu_full_output_equal=True)
    except BaseException as e:
        failure=e;summary.update(benchmark_passed=False,error_type=type(e).__name__,error=str(e))
    finally:
        for s in old:signal.signal(s,signal.SIG_IGN)
        if armed:
            try:
                for index in [0,1]:cleanup('strata-public-reduce12-gpu'+str(index))
                call(['docker','start','strata-api'],120)
                deadline=time.monotonic()+600
                while True:
                    try:
                        restored=health();identity=api_identity();break
                    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):
                        need(time.monotonic()<deadline,'stable API restoration timeout');time.sleep(5)
                need(identity==before,'restored stable binary identity changed')
                summary['production_restore']={'passed':True,'health':restored,'identity':identity,'config_and_compose_changed':False}
            except BaseException as e:summary['production_restore']={'passed':False,'error_type':type(e).__name__,'error':str(e)}
        summary['finished_utc']=stamp();write(out/'summary.json',summary)
        for s,handler in old.items():signal.signal(s,handler)
    need(summary['production_restore']['passed'],'stable API restoration failed; inspect summary and restore manually')
    if failure:raise failure
    print(json.dumps({'benchmark_passed':True,'production_restore':summary['production_restore'],'results':summary['results']}),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--build',type=Path,default=R/'logs/perf-campaign-20261009/qsa-reduce12-public-build')
    ap.add_argument('--expected-binary-sha',required=True)
    ap.add_argument('--output',type=Path,default=R/'logs/perf-campaign-20261009/qsa-reduce12-public-probe')
    ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    need(re.fullmatch('[0-9a-f]{64}',a.expected_binary_sha) is not None,'expected binary SHA required')
    need(a.input.is_absolute() and a.build.is_absolute() and a.output.is_absolute(),'absolute paths required')
    if not a.execute:print(json.dumps({'arguments_valid':True,'executed':False}));return
    need(__debug__,'python -O is prohibited')
    with open('/tmp/strata-direct-ff34-ce.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(a)
if __name__=='__main__':main()
