from pathlib import Path
import subprocess,json,hashlib,shutil
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-pr-qsa-swizzle-20261008');O=R/'logs/perf-campaign-20261008/qsa-query-swizzle/clean-pr-build';O.mkdir(exist_ok=False)
script='cmake -S . -B build-906 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_GGML_DIR=/src/third_party/llama.cpp -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx906 && cmake --build build-906 --target gfx906_qsa_query_swizzle_probe -j4 && python3 bench/test_gfx906_qsa_query_swizzle_guard.py'
cmd=['docker','run','--rm','--network','none','--name','strata-qsa-clean-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c',script]
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
d={'exit_code':p.returncode,'upstream_base':'fb58e0dbc8399662c0e47c76578c6e878b14f6cf','source_sha256':hashlib.sha256((W/'src/kernels/cuda/qsa_decode_attn.cu').read_bytes()).hexdigest(),'argv':cmd}
if p.returncode==0:
 shutil.copy2(W/'build-906/gfx906_qsa_query_swizzle_probe',O/'gfx906_qsa_query_swizzle_probe')
 d['probe_sha256']=hashlib.sha256((O/'gfx906_qsa_query_swizzle_probe').read_bytes()).hexdigest()
(O/'build.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True);assert p.returncode==0
