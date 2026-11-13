#!/usr/bin/env python3
"""Root invocation only. Default is offline receipt validation; no GPU work."""
import argparse,fcntl,json,os,re,signal
from pathlib import Path
import native_support as N
import native_arm as A
import qsa_probe_safety as S
import parse_cost as P
import device_compare as D
HERE=Path(__file__).resolve().parent
BASE='c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049'
V4_HOST='5696899be076d084e8b1896efffed5ad1981c484b79913a5486fc729f864ad87'
V4_INNER='1b483e54874fdc44b8497209068daff93eb46b798e4a1d96038f7c728b4fda27'
TAG='hc-cost'

def read_bound(path,digest):
 N.need(N.digest(digest) and N.sha(path)==digest,'receipt hash mismatch: '+path.name)
 return json.loads(path.read_text(),object_pairs_hook=P.base.unique)

def inputs(a):
 pins=json.loads((HERE/'input-pins.json').read_text())
 for name,h in pins.items():N.need(N.sha(HERE/name)==h,'pinned package input changed: '+name)
 plan=json.loads((HERE/'probe-plan.json').read_text());base=json.loads((HERE/'baseline.manifest.json').read_text())
 N.need(plan['environment']==dict(base['flags'],STRATA_HC_COUNTS_ONLY_OBSERVER='1') and plan['native_args']==base['native_args'],'exact original environment/native arguments')
 N.need(N.digest(a.binary_sha) and a.binary_sha!=BASE and N.sha(a.binary)==a.binary_sha,'explicit observer binary identity')
 N.need(N.sha(a.baseline)==BASE,'frozen baseline identity')
 ids=json.loads((HERE/'prompt-12288.ids.json').read_text());N.need(N.sha(HERE/'prompt-12288.ids.json')==plan['prefix_sha256'],'prefix bytes')
 N.need(len(ids)==12288 and all(type(i)is int and 0<=i<248320 for i in ids),'prefix count/domain')
 host=read_bound(a.build_receipt,a.build_receipt_sha);inner=read_bound(a.inner_receipt,a.inner_receipt_sha);gate=read_bound(a.device_gate,a.device_gate_sha)
 N.need(host['passed'] is True and host['status']=='PASS_COST_OBSERVER_BUILD_DEVICE_EQUALITY' and type(host['returncode'])is int and host['returncode']==0 and host['cleanup']['passed'] is True,'host build gate')
 N.need(host['inner_receipt_sha256']==a.inner_receipt_sha and host['artifact_hashes']['scratch-build/strata']==a.binary_sha,'host receipt binding')
 N.need(inner['passed'] is True and inner['status']=='PASS_HOST_ONLY_OBSERVER_BUILD_AND_DEVICE_EQUALITY' and inner['observer_binary_sha256']==a.binary_sha and inner['device_equality_sha256']==a.device_gate_sha,'inner build gate')
 v=inner['v4_inputs'];N.need(v['binary_sha256']==BASE and v['host_sha256']==V4_HOST and v['inner_sha256']==V4_INNER,'exact v4 provenance')
 N.need(gate['passed'] is True and gate['complete_device_payload_bytes_equal'] is True and gate['complete_module_kernel_metadata_equal'] is True,'full device payload gate')
 N.need(gate['baseline']['binary_sha256']==BASE and gate['observer']['binary_sha256']==a.binary_sha,'device gate identities')
 N.need(len(gate['baseline']['modules'])==len(gate['observer']['modules'])==55,'full module inventory')
 actual=D.compare(a.baseline,a.binary);N.need(actual==gate,'fresh complete device payload gate differs from receipt')
 review=read_bound(a.review_receipt,a.review_receipt_sha)
 expected=dict(binary_sha256=a.binary_sha,build_receipt_sha256=a.build_receipt_sha,inner_receipt_sha256=a.inner_receipt_sha,device_gate_sha256=a.device_gate_sha,runner_sha256=N.sha(HERE/'run_cost.py'),input_pins_sha256=N.sha(HERE/'input-pins.json'))
 N.need(review.get('passed') is True and review.get('source_and_cpu_review_passed') is True and review.get('build_provenance_and_device_gate_passed') is True and review.get('bindings')==expected,'independent review missing or differently bound')
 return plan,ids,expected

def execute(a):
 N.need(a.leave_api_stopped,'API stays stopped')
 N.need(a.binary.resolve().is_relative_to(N.R) and a.baseline.resolve().is_relative_to(N.R) and a.output.resolve().is_relative_to(N.R),'existing R mount only')
 N.need(not a.output.exists(),'fresh evidence directory required')
 plan,ids,binding=inputs(a)
 N.need(not N.call(['docker','ps','-q']).strip(),'containers/API must already be stopped')
 env=json.loads(N.call(['docker','image','inspect',N.IMAGE,'--format','{{json .Config.Env}}'])) or []
 inherited=dict(x.split('=',1) for x in env if '=' in x and x.split('=',1)[0].startswith(('STRATA_','HIP_','ROCR_','HSA_')))
 N.need(all(k in plan['environment'] for k in inherited),'unexpected inherited route/geometry flags')
 ds=N.devices();before=N.gpu(ds);a.output.mkdir(mode=0o700);out=a.output
 summary=dict(started_utc=N.stamp(),bindings=binding,planned_processes=1,requests_per_process=1,requested_outputs=1,automatic_retries=0,performance_claim=False,completed=False,cleanup={'passed':False},api_actions=False,power_changes=False)
 N.write(out/'summary.json',summary);failure=None
 def stop(sig,frame):raise InterruptedError('requested stop')
 old={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
 try:
  # Persistent within the campaign, keyed by observer bytes. No retry with a new output path.
  marker=N.R/'logs/perf-campaign-20261009'/('hc-cost-'+a.binary_sha+'.once')
  fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  with os.fdopen(fd,'w') as f:f.write(str(out)+'\n');f.flush();os.fsync(f.fileno())
  N.need(N.sha(a.binary)==a.binary_sha,'binary changed before arm')
  arm=dict(binary=str(a.binary),binary_sha256=a.binary_sha,release_artifacts=binding,flags=plan['environment'],native_args=plan['native_args'])
  summary['native']=A.run_arm(TAG,arm,dict(fixture_sha256=plan['prefix_sha256'],requested_outputs=1),ids,ds,out)
  termination=json.loads((out/'private'/f'{TAG}.termination.json').read_text())
  N.need(termination['native_exit']==0 and termination['native_completed'] is True and termination['reader_finished'] is True and termination['reader_error'] is None and termination['native_cleanup_error'] is None,'complete process lifetime required before destructor parse')
  raw=out/'private'/f'{TAG}.stderr';summary['raw_stderr_sha256']=N.sha(raw)
  summary['cost']=P.parse(raw.read_text());N.write(out/'cost.json',summary['cost'])
  summary['completed']=True
 except BaseException as e:failure=e;summary.update(error_type=type(e).__name__,error=str(e))
 finally:
  for sig in old:signal.signal(sig,signal.SIG_IGN)
  errors=[];after=None
  try:S.cleanup('strata-fit17-correctness-'+TAG)
  except BaseException as e:errors.append(dict(target='owned_container',error=str(e)))
  try:
   N.need(not N.call(['docker','ps','-q']).strip(),'container remains after cleanup');after=N.gpu(ds)
   N.need([(r['pci'],r['mode'],r['power1_cap']) for r in before]==[(r['pci'],r['mode'],r['power1_cap']) for r in after],'power configuration changed')
  except BaseException as e:errors.append(dict(target='final_state',error=str(e)))
  summary['cleanup']=dict(passed=not errors,api_left_stopped=not errors,gpu_after=after,errors=errors);summary['finished_utc']=N.stamp();N.write(out/'summary.json',summary)
  for sig,h in old.items():signal.signal(sig,h)
 N.need(summary['cleanup']['passed'],'cleanup failed')
 if failure:raise failure
 print(json.dumps(dict(completed=True,performance_claim=False,gate=summary['cost']['gate'])))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for n in ['binary','baseline','build-receipt','inner-receipt','device-gate','review-receipt','output']:ap.add_argument('--'+n,type=Path,required=True)
 for n in ['binary-sha','build-receipt-sha','inner-receipt-sha','device-gate-sha','review-receipt-sha']:ap.add_argument('--'+n,required=True)
 ap.add_argument('--execute',action='store_true');ap.add_argument('--leave-api-stopped',action='store_true');a=ap.parse_args();N.need(__debug__,'python -O forbidden')
 N.need(all(getattr(a,n).is_absolute() for n in ['binary','baseline','build_receipt','inner_receipt','device_gate','review_receipt','output']),'absolute paths required')
 if not a.execute:inputs(a);print('offline admission passed; no execution');return
 with open('/tmp/strata-direct-ff34-ce.lock','w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()
