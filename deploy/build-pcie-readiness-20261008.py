from pathlib import Path
import subprocess,shutil,json,hashlib
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261008/pcie-readiness'
files=['include/strata/core/verify.hpp','include/strata/kernels/verify_kernels.hpp','src/core/verify.cpp','src/kernels/cuda/verify_kernels.cu']
for f in files:
 dest=O/'before'/f;dest.parent.mkdir(parents=True,exist_ok=True)
 assert not dest.exists();shutil.copy2(W/f,dest)
subprocess.run(['git','apply','--check',str(O/'diagnostic.patch')],cwd=W,check=True)
subprocess.run(['git','apply',str(O/'diagnostic.patch')],cwd=W,check=True)
cmd=['docker','run','--rm','--network','none','--name','strata-pcie-readiness-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c','cmake --build build-906 --target strata -j4']
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
d={'exit_code':p.returncode,'patch_sha256':hashlib.sha256((O/'diagnostic.patch').read_bytes()).hexdigest()}
if p.returncode==0:
 shutil.copy2(W/'build-906/strata',O/'strata');d['binary_sha256']=hashlib.sha256((O/'strata').read_bytes()).hexdigest()
(O/'build.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True);assert p.returncode==0
