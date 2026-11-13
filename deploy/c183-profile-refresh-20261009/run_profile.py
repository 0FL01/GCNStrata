#!/usr/bin/env python3
"""Root-only c183 control/profile diagnostic. Default validates syntax, never launches GPU work."""
import argparse
import fcntl
import hashlib
import json
import re
import signal
from pathlib import Path
import native_support as N
import native_arm as A
import qsa_probe_safety as S

HERE = Path(__file__).resolve().parent
BINARY_SHA = 'c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049'
FIXTURE_SHA = '63ce3d4042027a00b2193feab143f1d9443b5c31a863a962ec9543ccb54d9695'
ARCHIVE_SHA = '135bfd9e51c94d9596db56d69db956f0397ff088097b28950002b98c3b97af9b'
STAMP_SOURCE_SHA = '1f7a650452510af9892b9636534c8086fd80578ddee1733022e926f3285ae15c'
STAMP_OBJECT_SHA = '50a3e527914aa51e6c85a6b6aa358625b4398bd1a9fb9ab06742c735194486c5'
PINNED = {
    'native_support.py':'850a18e95a5a2e13013f778ecb7b807809def5e62154fe9ff91acfe63b8f05bf',
    'qsa_probe_safety.py':'749392111a9646aa7b7c03ae0cea8c30253922c23d9e9330ad7415222e929e45',
    'native_arm.py':'d344633b9a1730f971d50d19a13f7a2d6eee4ffa4694ef42c4af4a2b0800c355',
    'baseline.manifest.json':'fb71583d2f8d6d52bba57db43d7d9c489bf2a1648b3543473183126d9f92e8b2',
    'historical.result.json':'7021b6532ee7ade54a22af07669208cf71db818fd3d1cbad1a7632eaf4ec3233',
}
TIMING = ['STRATA_PREFILL_TIMING','STRATA_DECODE_TIMING','STRATA_SPLIT_TIMING','STRATA_VERIFY_PROFILE']
COUNTERS = ['accepted','offered','cache_hits','lookups','ram_blobs','file_blobs','file_read_mb',
            'offloaded_experts','lookup_chain_accepted','lookup_chain_offered','native_suffix_accepted',
            'native_suffix_offered','decode_windows','native_done_field_count','suffix_drafts']

def inputs(a):
    for name,digest in PINNED.items(): N.need(N.sha(HERE/name)==digest,'pinned input changed: '+name)
    base=json.loads((HERE/'baseline.manifest.json').read_text())
    N.need(base['release_binary_sha256']==BINARY_SHA and base['fixture_sha256']==FIXTURE_SHA,'baseline identity')
    N.need(all(k not in base['flags'] for k in TIMING+['STRATA_SH_STREAM']),'unexpected baseline route/timing')
    N.need(base['sampling']=={'temperature':0,'top_p':1,'top_k':1,'seed':12345},'sampling identity')
    N.need(N.sha(a.binary)==BINARY_SHA,'frozen c183 binary mismatch')
    N.need(N.sha(a.fixture)==FIXTURE_SHA,'original fixture bytes required')
    ids=json.loads(a.fixture.read_text())
    N.need(isinstance(ids,list) and len(ids)==65536 and all(type(x) is int and 0<=x<248320 for x in ids),'fixture ID domain/count')
    N.need(N.sha(a.timestamp_receipt)==a.timestamp_receipt_sha,'timestamp receipt binding')
    receipt=json.loads(a.timestamp_receipt.read_text());q=receipt['clock_query']
    N.need(receipt.get('binary_sha256')==BINARY_SHA and receipt.get('kernel_archive_sha256')==ARCHIVE_SHA,'timestamp build identity')
    N.need(receipt.get('timestamp_source_sha256')==STAMP_SOURCE_SHA,'timestamp source identity')
    N.need(receipt.get('timestamp_object_sha256')==STAMP_OBJECT_SHA,'timestamp object identity')
    N.need(receipt.get('archive_member_matches_object') is True and receipt.get('producer_isa_verified') is True,'actual producer provenance required')
    N.need(receipt.get('producer_ns_per_tick')==40,'actual producer conversion')
    N.need(receipt.get('query_source_sha256')==N.sha(HERE/'query_wallclock.cpp'),'clock query source binding')
    N.need(q.get('init_rc')==q.get('count_rc')==0 and q.get('device_count')==2,'clock runtime query failed')
    ds=q.get('devices',[]);N.need(len(ds)==2 and sorted(r['index'] for r in ds)==[0,1],'clock devices')
    for d in ds:
        N.need(d.get('attribute_rc')==d.get('pci_rc')==d.get('props_rc')==0,'clock device query failed')
        N.need(d.get('wall_clock_rate_khz')==25000 and d.get('gcn_arch','').split(':')[0]=='gfx906','clock rate/architecture')
    N.need({r['pci'].lower() for r in ds}=={'0000:05:00.0','0000:08:00.0'},'clock physical devices')
    return base,ids,receipt

def diagnostic(stderr):
    lines=stderr.splitlines()
    pp=[x for x in lines if x.startswith('strata prefill timing:') or x.startswith('strata prefill timing (peer GPU, ms):')]
    decode=[x for x in lines if x.startswith('strata decode timing:')]
    stages=[x for x in lines if x.startswith('strata decode GPU stages')]
    split=[x for x in lines if re.match(r'^strata serve: stage \d+:',x)]
    N.need(pp and len(decode)==1 and len(stages)==2 and len(split)==2,'required timing reports absent/duplicated')
    N.need(stages[0].startswith('strata decode GPU stages (ms/window):') and stages[1].startswith('strata decode GPU stages, stage 1 (ms/window):'),'two profile stage identities')
    m=re.match(r'^strata decode timing: (\d+) windows,',decode[0]);N.need(m and int(m[1])>0,'decode diagnostic windows')
    windows=int(m[1])
    for line in stages:
        m=re.search(r'\| total ([0-9.]+) ms/window over (\d+) windows$',line)
        N.need(m and float(m[1])>0 and int(m[2])==windows,'GPU stage window coverage')
    return {'prefill_raw':pp,'decode_raw':decode,'gpu_stages_raw':stages,'split_raw':split,
            'diagnostic_decode_windows':windows,'scope':'instrumented main-fork-serialized stream intervals, not exclusive kernel costs',
            'additive_whole_model_interpretation':False}

def compare(control,profile):
    N.need(control['output_ids']==profile['output_ids'],'control/profile output ID mismatch')
    for k in N.TOPOLOGY_KEYS: N.need(control['engine_info'][k]==profile['engine_info'][k],'topology changed: '+k)
    return {'all_output_ids_equal':True,'topology_equal':True,
            'counter_values':{k:{'control':control.get(k),'profile':profile.get(k)} for k in COUNTERS},
            'native_phase_time_change_pct':{k:100*(profile[k]/control[k]-1) for k in ['prompt_ms','decode_ms','request_wall_s']},
            'control_one_pct_throughput_budget_ms':{k:control[k]/101 for k in ['prompt_ms','decode_ms']},
            'interpretation':'single ordered diagnostic pair; changes combine timing instrumentation, forced main-fork serialization, adaptive scheduling, and run variability; no optimization gain claim',
            'pure_instrumentation_overhead_established':False,'performance_gain_claim':False}

def execute(a):
    N.need(a.leave_api_stopped,'API must remain stopped')
    N.need(a.binary.resolve().is_relative_to(N.R) and a.output.resolve().is_relative_to(N.R),'binary/output must be under existing R mount')
    N.need(not a.output.exists(),'fresh output required')
    base,ids,receipt=inputs(a)
    N.need(not N.call(['docker','ps','-q']).strip(),'all containers/API must already be stopped')
    # Inspect only relevant image defaults, never dump the complete environment.
    env=json.loads(N.call(['docker','image','inspect',N.IMAGE,'--format','{{json .Config.Env}}'])) or []
    inherited=dict(x.split('=',1) for x in env if '=' in x and x.split('=',1)[0].startswith('STRATA_'))
    N.need(all(k in base['flags'] for k in inherited),'unexpected inherited STRATA route/timing flags')
    ds=N.devices();before=N.gpu(ds);a.output.mkdir(mode=0o700);out=a.output
    summary={'started_utc':N.stamp(),'scope':'current c183 altered-scheduling attribution diagnostic',
             'binary_sha256':BINARY_SHA,'fixture_sha256':FIXTURE_SHA,'image':N.IMAGE,
             'timestamp_receipt_sha256':a.timestamp_receipt_sha,'timestamp_receipt':receipt,
             'image_inherited_strata_flags':inherited,'profile_flags':TIMING,'main_shared_fork':{'control':True,'profile':False},
             'planned_processes':2,'requests_per_process':1,'requested_outputs':1024,'automatic_retries':0,
             'results':[],'completed':False,'api_actions':False,'power_changes':False,'promoted':False,'cleanup':{'passed':False}}
    failure=None
    def stop(signum,frame): raise InterruptedError('requested stop')
    old={s:signal.signal(s,stop) for s in [signal.SIGINT,signal.SIGTERM]}
    try:
        for tag,profile in [('control',False),('profile',True)]:
            N.need(N.sha(a.binary)==BINARY_SHA and N.sha(a.fixture)==FIXTURE_SHA,'binary/fixture changed before arm')
            flags=dict(base['flags']);flags.update({k:'1' for k in TIMING} if profile else {})
            r={'binary':str(a.binary),'binary_sha256':BINARY_SHA,'release_artifacts':{'frozen_binary':str(a.binary)},'flags':flags,'native_args':base['native_args']}
            spec={'fixture_sha256':FIXTURE_SHA,'requested_outputs':1024}
            result=A.run_arm(tag,r,spec,ids,ds,out)
            result['timing_scope']='diagnostic control/profile; no optimization gain claim'
            err=(out/'private'/(tag+'.stderr')).read_text()
            if profile: summary['profile_reports']=diagnostic(err)
            else:
                N.need(not any(x.startswith(('strata prefill timing:','strata decode timing:','strata decode GPU stages','strata serve: stage ')) for x in err.splitlines()),'control unexpectedly instrumented')
                historical=json.loads((HERE/'historical.result.json').read_text())
                result['historical_c183_ids_equal']=result['output_ids']==historical['output_ids']
                N.need(result['historical_c183_ids_equal'],'fresh control differs from historical c183 IDs')
            summary['results'].append(result);N.write(out/'summary.json',summary)
        summary['comparison']=compare(*summary['results']);summary['completed']=True
    except BaseException as e:
        failure=e;summary.update(error_type=type(e).__name__,error=str(e))
    finally:
        for sig in old: signal.signal(sig,signal.SIG_IGN)
        try:
            for tag in ['control','profile']: S.cleanup('strata-fit17-correctness-'+tag)
            N.need(not N.call(['docker','ps','-q']).strip(),'unexpected container after cleanup')
            after=N.gpu(ds)
            N.need([(r['pci'],r['mode'],r['power1_cap']) for r in before]==[(r['pci'],r['mode'],r['power1_cap']) for r in after],'power configuration changed')
            summary['cleanup']={'passed':True,'api_left_stopped':True,'gpu_after':after}
        except BaseException as e: summary['cleanup']={'passed':False,'error':str(e)}
        summary['finished_utc']=N.stamp();N.write(out/'summary.json',summary)
        for sig,handler in old.items(): signal.signal(sig,handler)
    N.need(summary['cleanup']['passed'],'cleanup failed')
    if failure: raise failure
    print(json.dumps({'completed':True,'all_output_ids_equal':True,'performance_gain_claim':False}))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['binary','fixture','timestamp-receipt','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--timestamp-receipt-sha',required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--leave-api-stopped',action='store_true')
    a=ap.parse_args();N.need(__debug__,'python -O forbidden')
    N.need(all(getattr(a,k).is_absolute() for k in ['binary','fixture','timestamp_receipt','output']),'absolute paths required')
    N.need(re.fullmatch('[0-9a-f]{64}',a.timestamp_receipt_sha),'explicit timestamp receipt hash required')
    N.need(a.leave_api_stopped,'offline only')
    if not a.execute: print('arguments valid; no execution');return
    with open('/tmp/strata-direct-ff34-ce.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__': main()
