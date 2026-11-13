#!/usr/bin/env python3
"""Fixed one-shot 64K/1024 ABBA. Offline admission by default. No automatic promotion."""
import argparse,fcntl,json,math,os,signal,sys,time
from pathlib import Path
import native_support as N
import native_arm as A
import qsa_probe_safety as S
import reviewed_direct_reference as C
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE/'qualification'))
import run_qualification as Q
BASE=Q.BASE
FIXTURE='63ce3d4042027a00b2193feab143f1d9443b5c31a863a962ec9543ccb54d9695'
ORDER=['A1','B1','B2','A2']
PRIMARY='STRATA_HC_WNEG_PRIMARY';QUALIFY='STRATA_HC_WNEG_QUALIFY'
POLICY=dict(order=ORDER,outputs=1024,prompt_tokens=65536,seed=12345,arm_deadline_seconds=600,shutdown_deadline_seconds=60,cooldown_deadline_seconds=900,overall_schedule_deadline_seconds=7200,pp_mean_target_pct=1.0,pp_both_pairs_positive=True,tg_min_mean_gain_pct=0.0,tg_min_pair_gain_pct=0.0,automatic_retries=0,promoted=False)

def load(path):return json.loads(path.read_text(),object_pairs_hook=Q.P.base.unique)
def bound(path,digest):return Q.read_bound(path,digest)

def check_qualification(q,binding):
 N.need(q.get('completed') is True and q.get('cleanup',{}).get('passed') is True and q['cleanup'].get('api_left_stopped') is True,'qualification completion/cleanup')
 N.need(q.get('bindings')==binding,'qualification exact build/binary/review bindings')
 N.need(q.get('planned_processes')==1 and q.get('requested_outputs')==1 and q.get('automatic_retries')==0,'qualification fixed schedule')
 gate=q.get('qualification',{})
 proof=load(HERE/'qualification/baseline-proof.json')
 expected_gate=dict(output_ids=proof['output_ids'],keys=proof['keys'],original_routes=proof['routes'],primary_keys=108,primary_candidates=324,primary_learned=11,primary_hits=22,primary_physical_guards_inferred=302,peer_keys=84,peer_attempts=252,cache_scope_devices=[0],performance_claim=False)
 for k,v in expected_gate.items():N.need(gate.get(k)==v,'exact qualified payload mismatch: '+k)
 N.need(gate.get('gate')=='PASS_PRIMARY_CORRECTNESS_AND_REAL_HIT_POPULATION' and gate.get('primary_candidates')==324 and gate.get('primary_hits')==22 and gate.get('primary_learned')==11 and gate.get('cache_scope_devices')==[0],'qualification correctness and real hit population')
 n=q.get('native',{});N.need(n.get('passed') is True and n.get('prompt_tokens')==n.get('prompt_read')==12288 and n.get('prompt_reused')==0 and n.get('actual_outputs')==1,'qualification native coverage')
 N.need(n.get('output_ids')==gate.get('output_ids') and len(n['output_ids'])==1,'qualification output identity')

def inputs(a):
 for name,h in load(HERE/'input-pins.json').items():N.need(N.sha(HERE/name)==h,'package input changed: '+name)
 plan,_,binding=Q.inputs(a) # Full actual build, linked TLS, device equality, source and independent qualification admission.
 q=bound(a.qualification_receipt,a.qualification_receipt_sha);check_qualification(q,binding)
 base=load(HERE/'baseline.manifest.json');N.need(base['release_binary_sha256']==BASE and base['fixture_sha256']==FIXTURE,'baseline identity')
 N.need(base['sampling']==dict(temperature=0,top_p=1,top_k=1,seed=12345),'sampling')
 N.need(PRIMARY not in base['flags'] and QUALIFY not in base['flags'],'unmodified baseline environment')
 N.need(base['native_args']==plan['native_args'] and dict(base['flags'],**{PRIMARY:'1',QUALIFY:'1'})==plan['environment'],'exact qualification/default profile')
 N.need(N.sha(HERE/'prompt-65536.ids.json')==FIXTURE,'frozen fixture')
 ids=load(HERE/'prompt-65536.ids.json');N.need(len(ids)==65536 and all(type(i)is int and 0<=i<248320 for i in ids),'fixture domain')
 expected=dict(binding,qualification_receipt_sha256=a.qualification_receipt_sha,benchmark_runner_sha256=N.sha(HERE/'run_abba.py'),benchmark_input_pins_sha256=N.sha(HERE/'input-pins.json'),baseline_sha256=BASE,fixture_sha256=FIXTURE,baseline_flags=base['flags'],candidate_flags=dict(base['flags'],**{PRIMARY:'1'}),native_args=base['native_args'],policy=POLICY)
 review=bound(a.benchmark_review,a.benchmark_review_sha)
 N.need(review.get('passed') is True and review.get('source_and_cpu_review_passed') is True and review.get('actual_build_and_qualification_passed') is True and review.get('bindings')==expected,'final runtime benchmark admission missing or differently bound')
 return base,ids,expected

def verify_result(r,history):
 C.matches_historical_profile(r,history)
 N.need(r['prompt_read']==65536 and r['prompt_reused']==0 and len(r['output_ids'])==1024,'cold complete ID coverage')
 for key in C.COUNTER_KEYS+['native_done_field_count','suffix_drafts']:N.need(key in r,'counter missing: '+key)
 for key in ['accepted','offered','cache_hits','lookups']:
  N.need(type(r[key]) is int and r[key]>=0,'counter domain: '+key)
 N.need(r['accepted']<=r['offered'],'accepted exceeds offered')
 for key in ['prompt_ms','decode_ms']:N.need(type(r[key]) in (int,float) and math.isfinite(r[key]) and r[key]>0,'timing domain')
 r.update(PP=65536000/r['prompt_ms'],TG=1024000/r['decode_ms'])

def verify_raw(out,tag,r):
 lines=(out/'private'/f'{tag}.stdout').read_text().splitlines()
 N.need(sum(x.startswith('READY ') for x in lines)==sum(x.startswith('DONE ') for x in lines)==1,'exact READY/DONE coverage')
 N.need(not any(x.startswith('ERR ') for x in lines),'raw native error')
 N.need([int(x.split()[1]) for x in lines if x.startswith('T ')]==r['output_ids'],'raw complete 1024 IDs')
 N.need(lines.count('RESUME 0')==lines.count('REUSED 0')==1,'cold native protocol')

def compare(results):
 N.need([r['tag'] for r in results]==ORDER,'fixed ABBA coverage')
 for r in results:verify_result(r,results[0])
 report=C.compare(results)
 report['method']='one fixed 64K/1024 baseline/candidate/candidate/baseline sequence; PP hypothesis, TG control; descriptive only'
 report['constant_workload_claim']=False
 report['counter_values']={r['tag']:{k:r.get(k) for k in C.COUNTER_KEYS+['native_done_field_count','suffix_drafts']} for r in results}
 report['work_matched_speed_claim']=False
 for k in ['PP','TG']:
  report[k]['baseline_range']=[min(results[i][k] for i in (0,3)),max(results[i][k] for i in (0,3))]
  report[k]['candidate_range']=[min(results[i][k] for i in (1,2)),max(results[i][k] for i in (1,2))]
 report['pp_screen_passed']=report['PP']['gain_pct']>=1 and min(report['PP']['paired_gain_pct'])>0
 report['tg_control_passed']=report['TG']['gain_pct']>=0 and min(report['TG']['paired_gain_pct'])>=0
 report['counter_drift']=not report['all_counters_equal']
 report['screen_passed']=report['pp_screen_passed'] and report['tg_control_passed']
 report['interpretation']='ID-matched practical endpoint only; adaptive work differences are retained; four runs do not establish statistical significance or isolated guard-cost savings'
 report['promoted']=False
 return report

def execute(a):
 N.need(a.leave_api_stopped,'API stays stopped')
 N.need(all(p.resolve().is_relative_to(N.R) for p in (a.binary,a.baseline,a.output)),'existing R mount only')
 N.need(not a.output.exists(),'fresh evidence directory')
 base,ids,binding=inputs(a)
 N.need(not N.call(['docker','ps','-q']).strip(),'all containers/API must already be stopped')
 env=json.loads(N.call(['docker','image','inspect',N.IMAGE,'--format','{{json .Config.Env}}'])) or []
 inherited=dict(x.split('=',1) for x in env if '=' in x and x.split('=',1)[0].startswith(('STRATA_','HIP_','ROCR_','HSA_')))
 N.need(all(k in base['flags'] for k in inherited),'unexpected inherited route/geometry/instrumentation')
 ds=N.devices();before=N.gpu(ds);out=a.output;out.mkdir(mode=0o700)
 summary=dict(started_utc=N.stamp(),bindings=binding,policy=POLICY,image=N.IMAGE,image_inherited_flags=inherited,results=[],completed=False,cleanup={'passed':False},performance_claim=False,promoted=False,api_actions=False,power_changes=False)
 N.write(out/'summary.json',summary);failure=None;started=time.monotonic()
 def stop(sig,frame):raise InterruptedError('stop signal or fixed overall deadline')
 old={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM)}
 signal.alarm(POLICY['overall_schedule_deadline_seconds'])
 try:
  # Separate permanent marker; never removes or overwrites existing qualification/campaign markers.
  marker=N.R/'logs/perf-campaign-20261009'/('hc-primary-abba-'+a.binary_sha+'.once')
  fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  with os.fdopen(fd,'w') as f:f.write(str(out)+'\n');f.flush();os.fsync(f.fileno())
  history=load(HERE/'historical.result.json')
  for tag in ORDER:
   N.need(time.monotonic()-started<POLICY['overall_schedule_deadline_seconds'],'fixed schedule deadline')
   N.need(N.sha(a.binary)==a.binary_sha and N.sha(a.baseline)==BASE,'binary changed')
   flags=dict(base['flags']);candidate=tag.startswith('B')
   if candidate:flags[PRIMARY]='1'
   arm=dict(binary=str(a.binary if candidate else a.baseline),binary_sha256=a.binary_sha if candidate else BASE,release_artifacts=binding,flags=flags,native_args=base['native_args'])
   r=A.run_arm(tag,arm,dict(fixture_sha256=FIXTURE,requested_outputs=1024),ids,ds,out)
   summary['results'].append(r);N.write(out/'summary.json',summary)
   termination=load(out/'private'/f'{tag}.termination.json')
   N.need(termination['native_completed'] is True and termination['native_exit']==0 and termination['reader_finished'] is True and termination['reader_error'] is None and termination['native_cleanup_error'] is None,'complete process and evidence')
   stderr=(out/'private'/f'{tag}.stderr').read_text()
   N.need(not any(x in stderr for x in ('hc-wneg:','hc-wneg-key:','hc_observer_','strata prefill timing:','strata decode timing:','strata decode GPU stages')),'unexpected qualification/observer timing output')
   verify_result(r,history);verify_raw(out,tag,r);N.write(out/'summary.json',summary)
  summary['comparison']=compare(summary['results']);summary['completed']=True
 except BaseException as e:failure=e;summary.update(error_type=type(e).__name__,error=str(e))
 finally:
  signal.alarm(0)
  for sig in old:signal.signal(sig,signal.SIG_IGN)
  errors=[];after=None
  for tag in ORDER:
   try:S.cleanup('strata-primary-cache-abba-'+tag.lower())
   except BaseException as e:errors.append(dict(target=tag,error=str(e)))
  try:
   N.need(not N.call(['docker','ps','-q']).strip(),'container remains after cleanup');after=N.gpu(ds)
   N.need([(r['pci'],r['mode'],r['power1_cap']) for r in before]==[(r['pci'],r['mode'],r['power1_cap']) for r in after],'power configuration changed')
  except BaseException as e:errors.append(dict(target='final_state',error=str(e)))
  summary['cleanup']=dict(passed=not errors,api_left_stopped=not errors,gpu_after=after,errors=errors);summary['finished_utc']=N.stamp();N.write(out/'summary.json',summary)
  for sig,h in old.items():signal.signal(sig,h)
 N.need(summary['cleanup']['passed'],'cleanup failed')
 if failure:raise failure
 print(json.dumps(dict(completed=True,promoted=False,screen_passed=summary['comparison']['screen_passed'])))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 paths=['binary','baseline','build-receipt','inner-receipt','device-gate','review-receipt','qualification-receipt','benchmark-review','output']
 for n in paths:ap.add_argument('--'+n,type=Path,required=True)
 for n in ['binary-sha','build-receipt-sha','inner-receipt-sha','device-gate-sha','review-receipt-sha','qualification-receipt-sha','benchmark-review-sha']:ap.add_argument('--'+n,required=True)
 ap.add_argument('--execute',action='store_true');ap.add_argument('--leave-api-stopped',action='store_true');a=ap.parse_args();N.need(__debug__,'python -O forbidden')
 N.need(all(getattr(a,n.replace('-','_')).is_absolute() for n in paths),'absolute paths required')
 if not a.execute:inputs(a);print('offline admission passed; no execution');return
 with open('/tmp/strata-direct-ff34-ce.lock','w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()
