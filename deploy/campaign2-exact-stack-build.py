from pathlib import Path
import subprocess,json,hashlib,time
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-exact-stack');O=R/'logs/perf-campaign-20261007/exact-stack-build'
O.mkdir(exist_ok=False)
assert not subprocess.check_output(['docker','ps','-q']).strip()
patch=R/'logs/perf-campaign-20261007/entry/tracked.patch'
subprocess.run(['git','apply','--check',str(patch)],cwd=W,check=True)
subprocess.run(['git','apply',str(patch)],cwd=W,check=True)
manifest={'source_commit':'b6731bf','source_tar_sha256':hashlib.sha256((W/'source.tar').read_bytes()).hexdigest(),'deployment_patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'baseline_commit':'82f4473fb56fd8cf7355814c9512e21d67f4e0d2'}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
script='cmake -S . -B build-906 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_HIP_GFX906=ON -DSTRATA_PORTABLE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DSTRATA_GGML_DIR=/src/third_party/llama.cpp && cmake --build build-906 --target strata q2_unpack_spread_check iq_avx2_parity qsa_topk_parity native_expert_bench native_grouped_parity -j4'
cmd=['docker','run','--rm','--network','none','--name','strata-exact-stack-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c',script]
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
manifest['build_exit']=p.returncode;(O/'manifest.json').write_text(json.dumps(manifest,indent=2));assert p.returncode==0
manifest['binary_sha256']=hashlib.sha256((W/'build-906/strata').read_bytes()).hexdigest()
(O/'complete.json').write_text(json.dumps(manifest,indent=2));print('Clean exact stack build complete',flush=True)
