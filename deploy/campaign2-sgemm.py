from pathlib import Path
import subprocess,json,time
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007';W=Path('/home/radneon/strata-rnd-sgemm')
for _ in range(720):
 p=O/'mode16-64k/summary.json'
 if p.exists():
  j=json.loads(p.read_text());assert len(j['results'])==4 and all(x['passed'] for x in j['results']) and j['production_restore']['passed'];break
 time.sleep(5)
else:raise RuntimeError('64K series completion receipt absent')
subprocess.run(['docker','stop','--timeout','60','strata-api'],check=True)
assert not subprocess.check_output(['docker','ps','-q']).strip()
subprocess.run(['git','worktree','add','--detach',str(W),'82f4473'],cwd=R,check=True)
compat=subprocess.check_output(['git','diff','--','include/strata/platform/hip_compat/strata_hip.h','src/core/vmm.cpp','src/kernels/cuda/fused_gr.cu'],cwd=R)
subprocess.run(['git','apply','-'],input=compat,cwd=W,check=True)
subprocess.run(['git','apply',str(O/'sgemm-candidate.patch')],cwd=W,check=True)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-sgemm-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake -S . -B build-906 -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_GGML_DIR=/src/third_party/llama.cpp -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE=ON && cmake --build build-906 --target gfx906_gemm_screen gfx906_sgemm_parity strata -j4']
with (O/'sgemm-build.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(O/'sgemm-build-state.json').write_text(json.dumps({'exit':r.returncode}))
assert r.returncode==0
P=O/'sgemm-micro';P.mkdir()
results=[]
for gpu in [0,1]:
 for arm in [0,1,1,0]:
  i=len(results);env='STRATA_RDNA2_SGEMM='+str(arm)
  for target in ['gfx906_sgemm_parity','gfx906_gemm_screen']:
   tag=str(i)+'-g'+str(gpu)+'-arm'+str(arm)+'-'+target
   cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/'+target,'-v',str(W)+':/src','-e',env,'-e','HIP_VISIBLE_DEVICES='+str(gpu),image]
   with (P/(tag+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
   results.append({'tag':tag,'exit':r.returncode});(P/'results.json').write_text(json.dumps(results,indent=2))
   assert r.returncode==0,tag
(P/'complete.json').write_text(json.dumps({'passed':True,'invocations':len(results)}))
print('SGEMM component tests completed',flush=True)
