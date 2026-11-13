from pathlib import Path
import subprocess,json,hashlib
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007/topk-model-screen'
assert json.loads((R/'logs/perf-campaign-20261007/topk-reg-expanded/complete.json').read_text())['cases']==48
assert not subprocess.check_output(['docker','ps','-q']).strip()
baseline='3fdb12152c5e148c7af241cc4cde0e4f71494f657bba48f6ca9bb363b90757f6'
assert hashlib.sha256((R/'build-906/strata').read_bytes()).hexdigest()==baseline
assert hashlib.sha256((R/'logs/perf-campaign-20261007/entry/strata').read_bytes()).hexdigest()==baseline
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
script="trap 'cp /src/logs/perf-campaign-20261007/entry/strata /src/build-906/strata' EXIT; cmake --build build-906 --target strata -j4 && cp build-906/strata logs/perf-campaign-20261007/strata-topk-reg"
cmd=['docker','run','--rm','--network','none','--name','strata-topk-engine-build','--entrypoint','/bin/sh','-v',str(R)+':/src','-w','/src',image,'-c',script]
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert hashlib.sha256((R/'build-906/strata').read_bytes()).hexdigest()==baseline
assert p.returncode==0
(O/'build-state.json').write_text(json.dumps({'passed':True,'binary_sha256':hashlib.sha256((R/'logs/perf-campaign-20261007/strata-topk-reg').read_bytes()).hexdigest(),'production_binary_restored':True}))
p=subprocess.run(['python3',str(R/'deploy/bench-campaign2-topk-model.py')])
assert p.returncode==0
