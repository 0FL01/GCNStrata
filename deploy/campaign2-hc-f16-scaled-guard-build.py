from pathlib import Path
import subprocess,json,hashlib,datetime,shutil
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261007/hc-f16-scaled-guard'
cmd=['docker','run','--rm','--network','none','--name','strata-hc-scaled-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c','cmake -S . -B build-906 && cmake --build build-906 --target strata gfx906_hc_guard_probe -j4']
with (O/'build.log').open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
out={'build_exit':p.returncode,'source_base':'80e0dc997defc975495414c7707ee290bf1f4ad9','patches_sha256':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['scaled.patch','scaled-string-fix.patch','scaled-always-guard.patch','guard-probe-cmake.patch']},'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if p.returncode==0:
 dst=R/'logs/perf-campaign-20261007/strata-hc-f16-scaled-guard'
 assert not dst.exists();shutil.copy2(W/'build-906/strata',dst);out['binary_sha256']=hashlib.sha256(dst.read_bytes()).hexdigest()
(O/'build.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
