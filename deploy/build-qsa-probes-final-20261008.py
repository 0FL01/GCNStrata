from pathlib import Path
import subprocess,json,shutil,hashlib
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/qsa-query-swizzle'
for name,wpath in [('hardened-build','/home/radneon/strata-rnd-hc-f16'),('clean-pr-build','/home/radneon/strata-pr-qsa-swizzle-20261008')]:
 W=Path(wpath);D=O/name
 cmd=['docker','run','--rm','--network','none','--name','strata-qsa-probe-final-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c','cmake --build build-906 --target gfx906_qsa_query_swizzle_probe -j4 && python3 bench/test_gfx906_qsa_query_swizzle_guard.py']
 with (D/'build-final.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 d={'exit_code':p.returncode,'probe_source_sha256':hashlib.sha256((W/'bench/gfx906_qsa_query_swizzle_probe.cpp').read_bytes()).hexdigest(),'kernel_source_sha256':hashlib.sha256((W/'src/kernels/cuda/qsa_decode_attn.cu').read_bytes()).hexdigest(),'build':name}
 if p.returncode==0:
  shutil.copy2(W/'build-906/gfx906_qsa_query_swizzle_probe',D/'gfx906_qsa_query_swizzle_probe')
  d['probe_sha256']=hashlib.sha256((D/'gfx906_qsa_query_swizzle_probe').read_bytes()).hexdigest()
  if (D/'strata').exists():d['strata_sha256']=hashlib.sha256((D/'strata').read_bytes()).hexdigest()
 (D/'build-final.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True);assert p.returncode==0
