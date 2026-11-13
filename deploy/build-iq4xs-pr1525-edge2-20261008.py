from pathlib import Path
import subprocess,json,hashlib,shutil,os
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261008/iq4xs-flat-dequant/pr1525-edge'
records=[]
for variant in ['baseline','candidate']:
 shutil.copyfile(O/(variant+'.cu'),W/'src/kernels/cuda/dequant_bf16.cu')
 os.utime(W/'src/kernels/cuda/dequant_bf16.cu',None)
 targets='gfx906_iq4xs_edge_probe gfx906_iq4xs_dequant_probe'+(' strata' if variant=='candidate' else '')
 cmd=['docker','run','--rm','--network','none','--name','strata-iq4xs-edge-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c','cmake -S . -B build-906 && cmake --build build-906 --target '+targets+' -j4']
 with (O/(variant+'.build2.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 d={'variant':variant,'exit_code':p.returncode,'source_sha256':hashlib.sha256((O/(variant+'.cu')).read_bytes()).hexdigest(),'binaries':{}}
 if p.returncode==0:
  for name in targets.split():
   dest=O/(variant+'-'+name);shutil.copy2(W/'build-906'/name,dest)
   d['binaries'][name]=hashlib.sha256(dest.read_bytes()).hexdigest()
 records.append(d);(O/'build2.json').write_text(json.dumps(records,indent=2));print(json.dumps(d),flush=True);assert p.returncode==0
