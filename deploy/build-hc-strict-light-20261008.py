from pathlib import Path
import subprocess,json,hashlib,shutil,datetime
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261008/hc-strict-light'
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-hc-strict-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake --build build-906 --target strata gfx906_hc_guard_probe -j4']
d={'argv':cmd,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'patch_sha256':hashlib.sha256((O/'candidate.patch').read_bytes()).hexdigest(),'source_hashes':{n:hashlib.sha256((W/n).read_bytes()).hexdigest() for n in ['src/prefill/gemm.cu','bench/gfx906_hc_guard_probe.cpp']}}
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
d['exit_code']=p.returncode
if p.returncode==0:
 for name in ['strata','gfx906_hc_guard_probe']:
  shutil.copy2(W/'build-906'/name,O/name);d[name+'_sha256']=hashlib.sha256((O/name).read_bytes()).hexdigest()
d['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(O/'build.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True);assert p.returncode==0
