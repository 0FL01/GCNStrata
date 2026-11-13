from pathlib import Path
import subprocess,json,hashlib,datetime
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-pr-topk-20261008');O=R/'logs/topk-upstream-pr-20261008'
assert subprocess.check_output(['git','-C',str(R/'third_party/llama.cpp'),'rev-parse','HEAD'],text=True).strip()=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
script='cmake -S . -B build-906 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_HIP_GFX906=ON -DSTRATA_PORTABLE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DSTRATA_GGML_DIR=/src/third_party/llama.cpp && cmake --build build-906 --target strata qsa_topk_parity -j4'
cmd=['docker','run','--rm','--network','none','--name','strata-topk-upstream-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c',script]
m={'upstream':'6674a0065fb96bacde33e3eb10f91a1df86f95f2','argv':cmd,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'production_untouched':True,'candidate_patch_sha256':'744cdfc5be9ae9efbb7baedf49c18a474f23e62883599e6cf8a9381dac41f012','common_build_prerequisite':'one event alias from upstream PR1396, shared baseline/candidate','common_patch_sha256':'198be9ad4f6b00b4f3e7150ce6d7cb278996cfb90338548b904befad4d9594b6'}
(O/'candidate-build.json').write_text(json.dumps(m,indent=2))
with (O/'candidate-build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
m['exit_code']=p.returncode
if p.returncode==0:
 import shutil
 for name in ['strata','qsa_topk_parity']:
  src=W/'build-906'/name;shutil.copy2(src,O/('candidate-'+name));m[name+'_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
m['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(O/'candidate-build.json').write_text(json.dumps(m,indent=2));print(json.dumps(m),flush=True)
assert p.returncode==0
