from pathlib import Path
import subprocess,shutil,json,hashlib,datetime
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261009/qsa-reduce12-model'
P=R/'logs/perf-campaign-20261009/primary-commit-overlap'
files=['include/strata/core/verify.hpp','include/strata/kernels/iq_kernels.hpp','src/core/verify.cpp','src/kernels/cuda/iq_kernels.cu','src/program/generate.cpp']
q='src/kernels/cuda/qsa_decode_attn.cu'
assert hashlib.sha256((O/'candidate.patch').read_bytes()).hexdigest()=='ff08b97d7c71d74e46c5776cb5705280eff7df94f361f3865fb6527f4c60c942'
assert hashlib.sha256((R/'releases/gfx906-qualified-20261009-commit/strata').read_bytes()).hexdigest()=='8900243a832bde09818b7fc0fa922ad7d430891b3d378e10a370ba7a9fa0a1bf'
before={}
for f in files+[q]:
 dest=O/'before'/f;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copy2(W/f,dest);before[f]=hashlib.sha256(dest.read_bytes()).hexdigest()
assert before[q]=='c2dad92763333743b6c76cf05948c6fdcf706e28a8884353f78f8abd3de5cfa3'
d={'patch_sha256':hashlib.sha256((O/'candidate.patch').read_bytes()).hexdigest(),'base_binary_sha256':'8900243a832bde09818b7fc0fa922ad7d430891b3d378e10a370ba7a9fa0a1bf','before':before}
try:
 for f in files:
  shutil.copy2(P/'compiled-source'/f,W/f);(W/f).touch()
 subprocess.run(['git','apply','--check',str(O/'candidate.patch')],cwd=W,check=True)
 subprocess.run(['git','apply',str(O/'candidate.patch')],cwd=W,check=True)
 assert hashlib.sha256((W/q).read_bytes()).hexdigest()=='278f6080871552da95f18ea5e5af6a3e2a4208bf4d330368a6f054d895e91faf'
 d['compiled_sources']={}
 for f in files+[q]:
  dest=O/'compiled-source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(W/f,dest);d['compiled_sources'][f]=hashlib.sha256(dest.read_bytes()).hexdigest()
 cmd=['docker','run','--rm','--network','none','--name','strata-qsa-reduce12-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c','cmake --build build-906 --target strata -j4']
 d['argv']=cmd
 with (O/'build.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 d['exit_code']=r.returncode
 if r.returncode==0:
  shutil.copy2(W/'build-906/strata',O/'strata');d['binary_sha256']=hashlib.sha256((O/'strata').read_bytes()).hexdigest()
  shutil.copy2(W/'build-906/CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_decode_attn.cu.o',O/'qsa_decode_attn.o')
 assert r.returncode==0
finally:
 for f in files+[q]:shutil.copy2(O/'before'/f,W/f);(W/f).touch()
 d['sources_restored']=all(hashlib.sha256((W/f).read_bytes()).hexdigest()==before[f] for f in files+[q])
 d['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (O/'build.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True);assert d['sources_restored']
