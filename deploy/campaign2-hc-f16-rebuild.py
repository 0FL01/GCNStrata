from pathlib import Path
import subprocess,json,hashlib
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-hc-f16');O=R/'logs/perf-campaign-20261007/hc-f16-build'
assert not subprocess.check_output(['docker','ps','-q']).strip()
(O/'complete.json').rename(O/'complete-initial.json')
d=json.loads((O/'complete-initial.json').read_text())
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-hc-f16-rebuild','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake --build build-906 --target strata gfx906_hc_f16_probe -j4']
with (O/'rebuild.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0
d['archive_source_commit']=d['source_commit'];d['source_commit']='f9065d1b24090d644493d63008aef7ab59bf1f73'
d['source_delta_sha256']=hashlib.sha256((O/'review.patch').read_bytes()).hexdigest()
d['binary_sha256']=hashlib.sha256((W/'build-906/strata').read_bytes()).hexdigest()
(O/'complete.json').write_text(json.dumps(d,indent=2));print('HC FP16 reviewed prototype rebuilt',flush=True)
