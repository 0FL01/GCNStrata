from pathlib import Path
import subprocess,json,hashlib,shutil
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261008/hc-mmq-combined'
patch=(O/'candidate.patch').read_bytes()
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-hc-mmq-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake -S . -B build-906 && cmake --build build-906 --target strata gfx906_mmq_tile_probe -j4']
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
d={'argv':cmd,'exit_code':p.returncode,'patch_sha256':hashlib.sha256(patch).hexdigest()}
if p.returncode==0:
 for name in ['strata','gfx906_mmq_tile_probe']:
  shutil.copy2(W/'build-906'/name,O/name);d[name+'_sha256']=hashlib.sha256((O/name).read_bytes()).hexdigest()
(O/'build.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True);assert p.returncode==0
