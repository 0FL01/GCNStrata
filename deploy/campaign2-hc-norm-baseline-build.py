from pathlib import Path
import subprocess,json,hashlib,shutil,datetime
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-exact-stack');O=R/'logs/perf-campaign-20261007/hc-norm-reuse'
assert not subprocess.check_output(['docker','ps','-q']).strip()
cmd=['docker','run','--rm','--network','none','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c','cmake --build build-906 --target fused_gr_bench gr_parity -j4']
with (O/'baseline-build.log').open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
d={'build_exit':p.returncode,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
assert p.returncode==0
for name in ['fused_gr_bench','gr_parity']:
 dst=O/('baseline-'+name);assert not dst.exists();shutil.copy2(W/'build-906'/name,dst);d[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
(O/'baseline-build.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
