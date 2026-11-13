#!/usr/bin/env python3
"""Explicit safe full-engine build: ff34 six-file overlay plus scalar FIT17 only.
Default is spec validation; --execute modifies and restores seven build-tree files.
No GPU devices are passed to the build container. No API operation or promotion.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import re
import shutil
import signal
import subprocess
from pathlib import Path

R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16')
O=R/'logs/perf-campaign-20261009/qsa-topk-fit17-model'
BASE=R/'releases/gfx906-qualified-20261009-reduce12'
BASE_SHA='ff348b38ad9b531e8df701311b1ae736f1378a6aaef322eeaa86e78f685917d9'
IMAGE='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
SELECT='src/kernels/cuda/qsa_select.cu'
SIX={
 'include/strata/core/verify.hpp':'5a959164cbaed0feee14463ea036c0002816993a21b2f987d6dc57566fceef13',
 'include/strata/kernels/iq_kernels.hpp':'00e41a0b6d841cf6c532e012b113f5f0e4e9aa6b9e0d59a077b0e0bce2f1897a',
 'src/core/verify.cpp':'e0562925ea41f430193df5c326c7ddec743a59b1e43bf5d2545040475053c292',
 'src/kernels/cuda/iq_kernels.cu':'3aefd6d98a44fc5d5954be4122debf310d2515afd016773f021ebc0b33deebbb',
 'src/program/generate.cpp':'e7a5a754af2155c779aad5f2232ba1c0052ebbd2a4795d40885eff571b62221c',
 'src/kernels/cuda/qsa_decode_attn.cu':'278f6080871552da95f18ea5e5af6a3e2a4208bf4d330368a6f054d895e91faf',
}

def need(ok,label):
 if not ok:raise RuntimeError(label)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
 t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(path)
def digest(x):return isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) is not None

def validate_spec(s):
 need(s['base_binary_sha256']==BASE_SHA and s['base_source_sha256']==SIX,'ff34 source identity')
 need(s['image']==IMAGE and s['target']=='strata' and s['jobs']==4,'pinned build configuration')
 need(s['patch_path']==str(O/'candidate.patch'),'patch destination')
 for k in ['patch_sha256','select_original_sha256','select_candidate_sha256']:need(digest(s[k]),'required source/patch digest')
 need(s['feature']=='scalar FIT17 only' and s['flag']=='STRATA_GFX906_TOPK_FIT17' and s['default']=='off','scalar feature scope')

def execute(s):
 need(not subprocess.check_output(['docker','ps','-q']).strip(),'another container active; no API or other build may be running')
 need(sha(BASE/'strata')==BASE_SHA,'qualified base binary changed')
 for f,h in SIX.items():need(sha(BASE/'modified-source'/f)==h,'qualified source mismatch '+f)
 need(sha(W/SELECT)==s['select_original_sha256'],'qsa_select original mismatch')
 patch=Path(s['patch_path']);need(sha(patch)==s['patch_sha256'],'patch hash')
 stat=subprocess.check_output(['git','apply','--numstat',str(patch)],cwd=W,text=True).splitlines()
 need(len(stat)==1 and stat[0].split('\t')[-1]==SELECT,'patch must touch only qsa_select.cu')
 subprocess.run(['git','apply','--check',str(patch)],cwd=W,check=True,stdout=subprocess.DEVNULL)
 for name in ['before','compiled-source','build.json','strata','qsa_select.cu.o','build.log']:need(not (O/name).exists(),'fresh build outputs required: '+name)
 O.mkdir(parents=True,exist_ok=True)
 files=list(SIX)+[SELECT];before={}
 # Complete every snapshot before the first build-tree write.
 for f in files:
  dest=O/'before'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(W/f,dest);before[f]=sha(dest)
  need(before[f]==sha(W/f),'snapshot verification '+f)
 d={'started_utc':stamp(),'base_binary_sha256':BASE_SHA,'base_source_sha256':SIX,'patch_sha256':s['patch_sha256'],'before':before,'feature':'scalar FIT17 only','api_touched':False,'exit_code':None,'sources_restored':False}
 cmd=['docker','run','--rm','--network','none','--name','strata-qsa-topk-fit17-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',IMAGE,'-c','cmake --build build-906 --target strata -j4']
 d['argv']=cmd;p=None;error=None
 def interrupted(signum,frame):raise InterruptedError('build stop requested')
 old={sig:signal.signal(sig,interrupted) for sig in [signal.SIGINT,signal.SIGTERM]}
 try:
  for f in SIX:shutil.copy2(BASE/'modified-source'/f,W/f);(W/f).touch()
  subprocess.run(['git','apply','--check',str(patch)],cwd=W,check=True,stdout=subprocess.DEVNULL)
  subprocess.run(['git','apply',str(patch)],cwd=W,check=True,stdout=subprocess.DEVNULL)
  need(sha(W/SELECT)==s['select_candidate_sha256'],'patched source digest');(W/SELECT).touch()
  d['compiled_sources']={}
  for f in files:
   dest=O/'compiled-source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(W/f,dest);d['compiled_sources'][f]=sha(dest)
  need({f:d['compiled_sources'][f] for f in SIX}==SIX,'qualified host/kernel changes lost')
  with (O/'build.log').open('x') as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);d['exit_code']=p.wait(timeout=1800)
  need(d['exit_code']==0,'full-engine build failed')
  shutil.copy2(W/'build-906/strata',O/'strata');d['binary_sha256']=sha(O/'strata')
  obj=W/'build-906/CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_select.cu.o'
  shutil.copy2(obj,O/'qsa_select.cu.o');d['select_object_sha256']=sha(O/'qsa_select.cu.o')
 except BaseException as e:
  error=e;d['error_type']=type(e).__name__
 finally:
  for sig in old:signal.signal(sig,signal.SIG_IGN)
  # Prove that the compiler stopped before restoring source bytes. Docker failure
  # must not cause an active compiler to race with source restoration.
  cleanup_error=False
  if p is not None and p.poll() is None:
   try:subprocess.run(['docker','stop','--time','30','strata-qsa-topk-fit17-build'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60,check=True)
   except (OSError,subprocess.SubprocessError):
    try:subprocess.run(['docker','kill','strata-qsa-topk-fit17-build'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30,check=True)
    except (OSError,subprocess.SubprocessError):cleanup_error=True
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:p.kill();p.wait();cleanup_error=True
  try:
   active=subprocess.check_output(['docker','ps','-q','--filter','name=^/strata-qsa-topk-fit17-build$'],timeout=30).strip()
   if active:cleanup_error=True
  except (OSError,subprocess.SubprocessError):cleanup_error=True
  d['compiler_confirmed_stopped']=not cleanup_error
  failures=[]
  if cleanup_error:
   # Snapshots remain under before/ for explicit recovery after compiler shutdown.
   failures=list(files)
  else:
   for f in files:
    try:shutil.copy2(O/'before'/f,W/f);(W/f).touch();need(sha(W/f)==before[f],'restore digest')
    except BaseException:failures.append(f)
  d['sources_restored']=not failures;d['restore_failed_files']=failures;d['finished_utc']=stamp();write(O/'build.json',d)
  for sig,handler in old.items():signal.signal(sig,handler)
  print(json.dumps({k:v for k,v in d.items() if k!='argv'}),flush=True)
 need(d['sources_restored'],'source restoration failed')
 if error is not None:raise error

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args();s=json.loads(a.spec.read_text());validate_spec(s)
 if not a.execute:print(json.dumps({'spec_valid':True,'executed':False}));return
 with open('/tmp/strata-qsa-fit17-qualification.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(s)
if __name__=='__main__':main()
