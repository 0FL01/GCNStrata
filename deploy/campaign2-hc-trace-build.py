from pathlib import Path
import subprocess,json,time,hashlib,shutil
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007';W=Path('/home/radneon/strata-rnd-sgemm')
for _ in range(720):
 p=O/'hc-warm/summary.json'
 if p.exists():
  j=json.loads(p.read_text());assert len(j['results'])==4 and all(x['passed'] for x in j['results']) and j['production_restore']['passed'];break
 time.sleep(5)
else:raise RuntimeError('warm request series incomplete')
subprocess.run(['docker','stop','--timeout','60','strata-api'],check=True)
assert not subprocess.check_output(['docker','ps','-q']).strip()
subprocess.run(['git','apply','--check',str(O/'hc-trace.patch')],cwd=W,check=True)
subprocess.run(['git','apply',str(O/'hc-trace.patch')],cwd=W,check=True)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake --build build-906 --target strata -j4']
with (O/'hc-trace-build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copy2(W/'build-906/strata',O/'strata-hc-trace')
(O/'hc-trace-build.json').write_text(json.dumps({'passed':True,'binary_sha256':hashlib.sha256((O/'strata-hc-trace').read_bytes()).hexdigest()}))
print('HC trace binary ready',flush=True)
