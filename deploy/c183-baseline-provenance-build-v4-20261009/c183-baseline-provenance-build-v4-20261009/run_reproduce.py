#!/usr/bin/env python3
"""Root opt-in CPU build container; original source is read-only, no GPU/network."""
import argparse,json,os,signal,subprocess,time,uuid
from pathlib import Path
import build_inside as B
HERE=Path(__file__).resolve().parent
IMAGE='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
SOURCE=Path('/home/radneon/strata-rnd-hc-f16')
def command(source,out,name):
 cmd=['docker','run','--rm','--pull=never','--name',name,'--user',str(os.getuid())+':'+str(os.getgid()),'--network','none','--read-only','--cap-drop=ALL','--security-opt','no-new-privileges','--pids-limit','512','--cpus','4','--memory','12g','--tmpfs','/tmp:rw,nosuid,nodev,size=2g','-e','TMPDIR=/tmp','-e','HOME=/tmp','-e','PYTHONDONTWRITEBYTECODE=1','-v',str(source)+':/src:ro','-v',str(source)+':/original:ro','-v',str(out/'scratch-build')+':/src/build-906:rw','-v',str(HERE)+':/probe:ro','-v',str(out)+':/out:rw','-w','/src/build-906','--entrypoint','python3',IMAGE,'/probe/build_inside.py']
 for relative,(filename,digest) in B.BASELINE.items():cmd[cmd.index('-w'):cmd.index('-w')]=['-v',str(HERE/'baseline'/filename)+':/src/'+relative+':ro']
 return cmd
def output_identity(output):
 B.need(os.getuid()==1000,'expected host uid1000');B.need(os.getuid()==os.geteuid() and os.getgid()==os.getegid(),'real/effective host identity must agree')
 st=output.stat();identity=dict(uid=os.getuid(),gid=os.getgid(),output_uid=st.st_uid,output_gid=st.st_gid,output_mode=oct(st.st_mode & 0o7777))
 B.need(st.st_uid==identity['uid'] and st.st_gid==identity['gid'] and (st.st_mode & 0o7777)==0o700,'output owner/mode must match explicit container uid:gid and 0700')
 return identity
def validate_paths(output):
 B.need(__debug__,'python -O forbidden');B.need(output.is_absolute(),'absolute output required');source=SOURCE.resolve();out=output.resolve();B.need(not out.exists(),'fresh output required');B.need(out!=source and source not in out.parents,'output must be outside original source/build tree');B.need(out!=HERE and HERE not in out.parents,'output must be outside immutable package')
def validate_package():
 B.validate_baseline()
 for line in (HERE/'SHA256SUMS').read_text().splitlines():
  digest,name=line.split();p=Path(name);B.need(not p.is_absolute() and '..' not in p.parts,'manifest path');B.need(B.sha(HERE/p)==digest,'package changed: '+name)
def validate(output):
 validate_paths(output);validate_package();return B.validate_metadata(SOURCE)
def execute(output):
 validate_paths(output);initial=None;name='strata-c183-provenance-'+uuid.uuid4().hex[:12];output.mkdir(mode=0o700);cmd=command(SOURCE,output,name)
 receipt=dict(passed=False,status='RUNNING',command=cmd,source=str(SOURCE),image=IMAGE,timeout_seconds=1000,network=False,gpu_devices=False,automatic_retries=0,container=name);B.write(output/'host-receipt.json',receipt)
 p=None;failure=None;owned=False;closure_receipt=None
 def stop(signum,frame):raise InterruptedError('requested stop')
 old={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
 try:
  receipt['output_identity']=output_identity(output);B.write(output/'host-receipt.json',receipt)
  validate_package();initial=B.validate_metadata(SOURCE);receipt['source_preflight']=initial;B.write(output/'host-receipt.json',receipt)
  closure_receipt=B.copy_build(SOURCE,output/'scratch-build');receipt['scratch_closure']=closure_receipt;B.write(output/'host-receipt.json',receipt)
  inspect_cmd=['docker','image','inspect','--format','{{.Id}}',IMAGE];inspection=subprocess.run(inspect_cmd,capture_output=True,text=True,timeout=15);receipt['image_inspect']=dict(argv=inspect_cmd,returncode=inspection.returncode,stdout=inspection.stdout,stderr=inspection.stderr);B.need(inspection.returncode==0 and inspection.stdout.strip()==IMAGE,'pinned image not locally available; never pull')
  with (output/'container.stdout').open('xb') as out,(output/'container.stderr').open('xb') as err:
   owned=True;p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True);receipt['returncode']=p.wait(timeout=1000)
  B.need(p.returncode==0,'container build failed')
  inner=json.loads((output/'reproduction/receipt.json').read_text());B.need(inner.get('passed') is True and inner.get('status')=='PASS_ORIGINAL_OBJECT_AND_FROZEN_BINARY_REPRODUCTION','inner reproduction gate failed')
  receipt.update(passed=True,status='PASS_ORIGINAL_SOURCE_PROVENANCE',inner_receipt_sha256=B.sha(output/'reproduction/receipt.json'))
 except BaseException as e:failure=e;receipt.update(error_type=type(e).__name__,error=str(e),status='FAILED_STOP_FOR_ANALYSIS')
 finally:
  for s in old:signal.signal(s,signal.SIG_IGN)
  cleanup=[]
  if owned:
   try:
    cleanup_cmd=['docker','rm','-f',name];cleaned=subprocess.run(cleanup_cmd,capture_output=True,text=True,timeout=20);receipt['container_cleanup']=dict(argv=cleanup_cmd,returncode=cleaned.returncode,stdout=cleaned.stdout,stderr=cleaned.stderr)
    if cleaned.returncode and 'No such container' not in cleaned.stderr:cleanup.append('owned container cleanup failed')
   except BaseException as e:cleanup.append(str(e))
  if p is not None:
   try:
    if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
    receipt['returncode']=p.wait(timeout=10)
   except BaseException as e:cleanup.append(str(e))
  try:
   if closure_receipt is not None:B.verify_closure(SOURCE,closure_receipt)
   if initial is not None:receipt['source_postflight']=B.validate_metadata(SOURCE);B.need(receipt['source_postflight']==initial,'original input changed')
  except BaseException as e:cleanup.append(str(e))
  receipt['cleanup']=dict(passed=not cleanup,errors=cleanup);receipt['passed']=receipt['passed'] and not cleanup
  if cleanup:receipt['status']='FAILED_STOP_FOR_ANALYSIS'
  receipt['artifact_hashes']={str(f.relative_to(output)):B.sha(f) for f in output.rglob('*') if f.is_file() and f.name not in ('host-receipt.json','host-receipt.json.tmp')};B.write(output/'host-receipt.json',receipt)
  for s,h in old.items():signal.signal(s,h)
 if failure:raise failure
 B.need(not cleanup,'cleanup/input integrity failure');print(json.dumps(dict(passed=True,receipt=str(output/'host-receipt.json'),model_run=False,gpu_run=False)))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 if a.execute:execute(a.output)
 else:print(json.dumps(dict(validation=validate(a.output),executed=False)))
if __name__=='__main__':main()
