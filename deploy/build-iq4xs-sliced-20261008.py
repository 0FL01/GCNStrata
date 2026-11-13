from pathlib import Path
import subprocess,json,hashlib,shutil
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261008/iq4xs-flat-dequant/sliced'
O.mkdir(exist_ok=False)
cmd=['docker','run','--rm','--network','none','--name','strata-iq4xs-sliced-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c','cmake --build build-906 --target gfx906_iq4xs_dequant_probe strata -j4']
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
d={'exit_code':p.returncode,'argv':cmd}
if p.returncode==0:
 for name in ['gfx906_iq4xs_dequant_probe','strata']:
  shutil.copy2(W/'build-906'/name,O/name)
  d[name+'_sha256']=hashlib.sha256((O/name).read_bytes()).hexdigest()
(O/'build.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True);assert p.returncode==0
