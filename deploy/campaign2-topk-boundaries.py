from pathlib import Path
import subprocess,json
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007/topk-boundaries'
assert not subprocess.check_output(['docker','ps','-q']).strip()
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-topk-build','--entrypoint','/bin/sh','-v',str(R)+':/src','-w','/src',image,'-c','cmake -S . -B build-906 && cmake --build build-906 --target qsa_topk_parity -j4']
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0
rows=[]
for gpu in [0,1]:
 for n,q,cap,flag,old in [(4096, 8, 204800, 2, 0), (65536, 8, 204800, 2, 0), (200000, 8, 204800, 2, 0), (65536, 9, 204800, 2, 0), (135160, 8, 135160, 2, 0), (135164, 8, 135164, 2, 0), (270328, 8, 270328, 2, 0), (270332, 8, 270332, 2, 0), (65536, 4, 204800, 2, 1), (65536, 4, 204800, 0, 0), (65536, 4, 204800, 3, 0)]:
  tag=f'g{gpu}-n{n}-q{q}-cap{cap}-flag{flag}-old{old}'
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/qsa_topk_parity','-v',str(R)+':/src','-e',f'HIP_VISIBLE_DEVICES={gpu}','-e',f'STRATA_GFX906_TOPK_REG={flag}','-e','STRATA_TOPK_TRACE=1',image,str(n),str(q),'10',str(cap),'0']
  if old:cmd[cmd.index(image):cmd.index(image)]=['-e','STRATA_TOPK_OLD=1']
  with (O/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  rows.append({'tag':tag,'exit':p.returncode});(O/'results.json').write_text(json.dumps(rows));assert p.returncode==0,tag
(O/'complete.json').write_text(json.dumps({'passed':True,'cases':len(rows)}))

cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/bin/sh','-v',str(R)+':/src','-w','/src/build-906',image,'-c','ctest --output-on-failure -R ^qsa_topk_gfx906_guarded$']
with (O/'ctest.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0
(O/'ctest-complete.json').write_text(json.dumps({'passed':True}))
