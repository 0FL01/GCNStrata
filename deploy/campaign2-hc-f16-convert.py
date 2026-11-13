from pathlib import Path
import subprocess,json,time
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-sgemm');O=R/'logs/perf-campaign-20261007/hc-f16-convert'
d=json.loads((R/'logs/perf-campaign-20261007/config-capacity-200k/summary.json').read_text());assert all(x['passed'] for x in d['results'])
assert not subprocess.check_output(['docker','ps','-q']).strip()
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-hc-f16-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake -S . -B build-906 && cmake -S . -B build-906 && cmake --build build-906 --target gfx906_hc_f16_convert -j4']
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0
rows=[]
for gpu in [0,1]:
 for arm in [0,1]:
  tag=f'g{gpu}-sgemm{arm}'
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--entrypoint','/src/build-906/gfx906_hc_f16_convert','-v',str(W)+':/src','-e','STRATA_RDNA2_SGEMM=0','-e','STRATA_GFX906_HC_SGEMM=1','-e','STRATA_GFX906_HC_SGEMM_BORROW=2','-e','STRATA_HIP_PROMPT_F16=0','-e',f'HIP_VISIBLE_DEVICES={gpu}',image]
  if arm:cmd[-1:-1]=['-e','BENCH_HC_F16=1']
  with (O/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  rows.append({'tag':tag,'exit':p.returncode});(O/'results.json').write_text(json.dumps(rows));assert p.returncode==0
(O/'complete.json').write_text(json.dumps({'passed':True,'note':'FP16 conversion included every call; synthetic BF16 source and FP64 sample errors, no model quality claim'}))
print('HC FP16 screen complete',flush=True)
