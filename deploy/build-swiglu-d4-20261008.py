from pathlib import Path
import subprocess,json,hashlib,shutil
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-swiglu-d4-probe-20261008');O=R/'logs/perf-campaign-20261008/swiglu-d4';O.mkdir(exist_ok=False)
script='/opt/rocm/llvm/bin/clang++ --version > /src/compiler-version.txt; cat /opt/rocm/.info/version > /src/rocm-version.txt; cmake -S . -B build-906 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_GGML_DIR=/src/third_party/llama.cpp -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx906 && cmake --build build-906 --target gfx906_swiglu_quant_probe -j2'
cmd=['docker','run','--rm','--network','none','--name','strata-swiglu-d4-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c',script]
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
d={'exit_code':p.returncode,'source_commit':'895801834ad8ba0acf7bb432565dd18372de00a6','argv':cmd}
if p.returncode==0:
 shutil.copy2(W/'build-906/gfx906_swiglu_quant_probe',O/'gfx906_swiglu_quant_probe');d['probe_sha256']=hashlib.sha256((O/'gfx906_swiglu_quant_probe').read_bytes()).hexdigest()
for name in ['compiler-version.txt','rocm-version.txt']:
 if (W/name).exists():shutil.copy2(W/name,O/name)
(O/'build.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True);assert p.returncode==0
