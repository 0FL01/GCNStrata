from pathlib import Path
import subprocess,json
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007/topk-reg-expanded'
assert not subprocess.check_output(['docker','ps','-q']).strip()
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-topk-build','--entrypoint','/bin/sh','-v',str(R)+':/src','-w','/src',image,'-c','cmake -S . -B build-906 && cmake --build build-906 --target qsa_topk_parity -j4']
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0
rows=[]
for gpu in [0,1]:
 for n,q,flag,count in [(4096, 1, 1, 0), (4096, 4, 1, 0), (4096, 6, 1, 0), (8192, 1, 1, 0), (8192, 4, 1, 0), (8192, 6, 1, 0), (16384, 1, 1, 0), (16384, 4, 1, 0), (16384, 6, 1, 0), (32768, 1, 1, 0), (32768, 4, 1, 0), (32768, 6, 1, 0), (65536, 1, 1, 0), (65536, 4, 1, 0), (65536, 6, 1, 0), (131072, 1, 1, 0), (131072, 4, 1, 0), (131072, 6, 1, 0), (200000, 1, 1, 0), (200000, 4, 1, 0), (200000, 6, 1, 0), (204800, 1, 1, 0), (204800, 4, 1, 0), (204800, 6, 1, 0)]:
  tag=f'g{gpu}-n{n}-q{q}-flag{flag}-count{count}'
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/qsa_topk_parity','-v',str(R)+':/src','-e',f'HIP_VISIBLE_DEVICES={gpu}','-e',f'STRATA_GFX906_TOPK_REG={flag}','-e','STRATA_TOPK_TRACE=1',image,str(n),str(q),'30','204800',str(count)]
  with (O/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  rows.append({'tag':tag,'exit':p.returncode});(O/'results.json').write_text(json.dumps(rows));assert p.returncode==0,tag
(O/'complete.json').write_text(json.dumps({'passed':True,'cases':len(rows)}))
