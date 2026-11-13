#!/usr/bin/env python3
"""Root-only one-shot synthetic timing screen. Default is CPU-only validation; never builds kernels."""
import argparse, fcntl, hashlib, json, os, re, signal, subprocess, sys, time, uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
PINNED={'native_support.py':'850a18e95a5a2e13013f778ecb7b807809def5e62154fe9ff91acfe63b8f05bf','qsa_probe_safety.py':'749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,label):
    if not ok:raise RuntimeError(label)
for name,digest in PINNED.items():need(sha(HERE/name)==digest,'helper changed: '+name)
import native_support as N
import qsa_probe_safety as S
import analyze
BASE='5abe1289b549a9fa4ad7da00398fd81179a6d3b414bd2b3c44b749d5e315f377'
CAND='3f0c5ecb285232b1e338bf0aa51ebe729eefaa547388c14ebbd488167105ecf9'
def validate(a):
    need(__debug__,'python -O forbidden')
    need(all(p.is_absolute() for p in [a.driver,a.output]),'absolute paths required')
    need(re.fullmatch('[0-9a-f]{64}',a.driver_sha),'driver hash required')
    need(sha(a.driver)==a.driver_sha,'driver changed')
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        h,name=line.split();need('/' not in name and name not in ['.','..'],'manifest filename');need(sha(HERE/name)==h,'package changed: '+name)
    need(sha(HERE/'resource-gate.json')=='b4da67f44d01b0a9af96358ba897640d5e6906afff3658838010c03829e52659','independent resource gate changed')
    gate=json.loads((HERE/'resource-gate.json').read_text())
    need(gate['gate']=='PASS_RESOURCE_AND_STATIC_ISA_ONLY' and gate['candidate_sha256']==CAND and gate['baseline_sha256']==BASE,'resource gate must PASS for these ELFs')
    need(len(gate['kernels'])==2 and all(k['resource_gate']=='PASS' for k in gate['kernels']),'both kernel resource gates must PASS')
    need(sha(HERE/'baseline.elf')==BASE and sha(HERE/'candidate.elf')==CAND,'ELF identity')
    need(a.leave_api_stopped,'must leave API stopped')
    need(sha(HERE/'exhaustive-audit.json')=='a12bd1f1953115c1c9a9252f35f75bfadc6c6db5e3e20298a0debf389afc04a7','exhaustive raw parity prerequisite changed')
def execute(a):
    validate(a);need(not a.output.exists(),'fresh output required')
    need(not N.call(['docker','ps','-q']).strip(),'all containers/API must already be off')
    ds=N.devices();before=N.gpu(ds);a.output.mkdir(mode=0o700);out=a.output
    name='strata-hc-screen-'+uuid.uuid4().hex[:12];p=None;owned=False;samples=[];failure=None
    summary={'scope':'One-shot synthetic timing screen; five descriptors, three patterns, two cache regimes; no model gain or bound','started_utc':N.stamp(),'baseline_elf_sha256':BASE,'candidate_elf_sha256':CAND,'driver_sha256':a.driver_sha,'source_sha256':sha(HERE/'screen.cpp'),'manifest_sha256':sha(HERE/'SHA256SUMS'),'image':N.IMAGE,'planned_samples':360,'planned_launches':752,'automatic_retries':0,'gpu_ordinal':0,'api_actions':False,'power_changes':False,'passed':False,'container':name}
    def stop(signum,frame):raise InterruptedError('requested stop')
    old={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        summary['gpu_before']=N.cooldown(ds,out)
        need(not N.call(['docker','ps','-q']).strip(),'another container became active')
        validate(a)
        cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','-e','HIP_VISIBLE_DEVICES=0','-e','ROCR_VISIBLE_DEVICES=0','--entrypoint','/probe/driver','-v',str(a.driver)+':/probe/driver:ro','-v',str(HERE)+':/input:ro','-v',str(out)+':/output:rw',N.IMAGE,'/input/baseline.elf','/input/candidate.elf','/output']
        summary['command']=cmd;N.write(out/'summary.json',summary)
        with (out/'raw.jsonl').open('x') as stdout,(out/'stderr.txt').open('x') as stderr:
            started=time.monotonic();owned=True;p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr)
            while p.poll() is None:
                need(time.monotonic()-started<300,'300-second driver timeout')
                need(not (out/'cancel').exists(),'cancellation marker')
                samples.append({'gpus':N.gpu(ds)})
                active=N.call(['docker','ps','--format','{{.Names}}'],timeout=2).split();need(all(x==name for x in active),'another container became active')
                time.sleep(.2)
            summary['exit_code']=p.returncode
            need(time.monotonic()-started<300,'300-second driver deadline exceeded')
        need(p.returncode==0,'driver exit nonzero')
        receipt=analyze.check(out);N.write(out/'screen-analysis.json',receipt)
        summary['passed']=True
    except BaseException as e:failure=e;summary.update(error_type=type(e).__name__,error=str(e))
    finally:
        for s in old:signal.signal(s,signal.SIG_IGN)
        errors=[]
        try:
            if owned:S.cleanup(name)
        except BaseException as e:errors.append(str(e))
        try:
            if p is not None:
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            need(not N.call(['docker','ps','-q']).strip(),'unexpected container after cleanup')
            after=N.gpu(ds);summary['gpu_after']=after
            need([(r['pci'],r['mode'],r['power1_cap']) for r in before]==[(r['pci'],r['mode'],r['power1_cap']) for r in after],'power configuration changed')
        except BaseException as e:errors.append(str(e))
        summary['cleanup']={'passed':not errors,'errors':errors,'api_left_stopped':not errors}
        summary['exit_code']=p.returncode if p else None
        summary['raw_hashes']={f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and (f.suffix=='.bin' or f.name in ['raw.jsonl','stderr.txt'])}
        summary['finished_utc']=N.stamp();summary['passed']=summary['passed'] and not errors
        N.write(out/'thermal.json',samples);N.write(out/'summary.json',summary)
        for s,h in old.items():signal.signal(s,h)
    need(not errors,'cleanup failed')
    if failure:raise failure
    print(json.dumps({'passed':True,'samples':360,'launches':752,'model_performance_claim':False}))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--driver',type=Path,required=True);ap.add_argument('--driver-sha',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--leave-api-stopped',action='store_true');a=ap.parse_args()
    validate(a)
    if not a.execute:print('CPU validation passed; no GPU/container operation');return
    with open('/tmp/strata-direct-ff34-ce.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()

