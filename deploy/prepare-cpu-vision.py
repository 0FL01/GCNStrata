from pathlib import Path
import urllib.request,json,subprocess,time,datetime,concurrent.futures,os
R=Path('/home/radneon/strata');out=R/'logs/cpu-vision-20261006';out.mkdir(exist_ok=False)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
repo='ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF';name='mmproj-Qwen3.8-Flash-Next-BF16.gguf'
def download():
 t=time.monotonic();print('DOWNLOAD start',flush=True)
 with urllib.request.urlopen('https://huggingface.co/api/models/'+repo,timeout=60) as f:meta=json.load(f)
 revision=meta['sha'];url='https://huggingface.co/'+repo+'/resolve/'+revision+'/'+name
 dest=R/'runtime/vision'/name;dest.parent.mkdir(exist_ok=True);part=Path(str(dest)+'.part')
 assert not dest.exists() and not part.exists()
 count=0;last=0
 with urllib.request.urlopen(url,timeout=120) as src,part.open('xb') as dst:
  while True:
   block=src.read(4*1024*1024)
   if not block:break
   dst.write(block);count+=len(block)
   if count-last>=128*1024*1024:print('DOWNLOAD bytes='+str(count),flush=True);last=count
 assert count==907543008,(count,'unexpected mmproj size')
 with part.open('rb') as f:assert f.read(4)==b'GGUF'
 part.rename(dest)
 result={'repo':repo,'revision':revision,'file':name,'bytes':count,'url':url,'path':str(dest),'elapsed_s':time.monotonic()-t,'gguf_magic_verified':True}
 (out/'download.json').write_text(json.dumps(result,indent=2));print('DOWNLOAD complete '+json.dumps(result),flush=True)
 return result
script='''set -eu
cmake -S /src/tools/vision -B /src/build-vision-cpu -DLLAMA_DIR=/src/third_party/llama.cpp -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DSTRATA_PORTABLE=ON -DSTRATA_VISION_CUDA=OFF -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_SYCL=OFF -DGGML_METAL=OFF -DGGML_RPC=OFF -DGGML_OPENCL=OFF
cmake --build /src/build-vision-cpu --target strata-vision -j4
ldd /src/build-vision-cpu/bin/strata-vision
'''
(R/'deploy/build-cpu-vision.sh').write_text('#!/bin/sh\nset -eu\ncd /home/radneon/strata\ndocker run --rm --name strata-vision-cpu-build --entrypoint /bin/sh -v "$PWD:/src" '+image+" -c '"+script+"'\n")
def build():
 t=time.monotonic();print('BUILD start',flush=True)
 cmd=['docker','run','--rm','--name','strata-vision-cpu-build','--entrypoint','/bin/sh','-v',str(R)+':/src',image,'-c',script]
 with (out/'build.log').open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 result={'native_exit':p.returncode,'elapsed_s':time.monotonic()-t,'argv':cmd};(out/'build.json').write_text(json.dumps(result,indent=2));assert p.returncode==0
 cache=(R/'build-vision-cpu/CMakeCache.txt').read_text()
 for flag in ['GGML_HIP','GGML_CUDA','GGML_VULKAN','GGML_SYCL','GGML_METAL','GGML_RPC','GGML_OPENCL']:
  assert flag+':BOOL=OFF' in cache,flag
 assert (R/'build-vision-cpu/bin/strata-vision').is_file()
 print('BUILD complete '+json.dumps({'native_exit':0,'elapsed_s':result['elapsed_s'],'gpu_backends_disabled':True}),flush=True)
 return result
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  a=pool.submit(download);b=pool.submit(build);a.result();b.result()
 result={'native_exit':0,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config_changed':False,'service_restarted':False}
except BaseException as e:
 result={'native_exit':1,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error_type':type(e).__name__,'error':str(e)[:500]}
(out/'prepare-exit.json').write_text(json.dumps(result));print(json.dumps(result),flush=True)
raise SystemExit(result['native_exit'])
