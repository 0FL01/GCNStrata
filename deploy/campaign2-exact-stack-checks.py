from pathlib import Path
import subprocess,json,hashlib
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-rnd-exact-stack');O=R/'logs/perf-campaign-20261007/exact-stack-checks'
O.mkdir(exist_ok=False)
receipt=json.loads((R/'logs/perf-campaign-20261007/exact-stack-build/complete.json').read_text())
assert hashlib.sha256((W/'build-906/strata').read_bytes()).hexdigest()==receipt['binary_sha256']
assert not subprocess.check_output(['docker','ps','-q']).strip()
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
rows=[]
def execute(tag,cmd,timeout=600):
 with (O/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 rows.append({'tag':tag,'exit':p.returncode});(O/'results.json').write_text(json.dumps(rows,indent=2));assert p.returncode==0,tag
base=['docker','run','--rm','--network','none','-v',str(W)+':/src','-w','/src/build-906']
execute('cpu-ctest',base+['--entrypoint','/bin/sh',image,'-c',"ctest --output-on-failure -R '^(q2_unpack_spread_check|iq_avx2_parity|iq_avx2_parity_q2_spread)$'"])
model='/sata/shefowl-qwen38-hybrid/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0-00001-of-00002.gguf'
for gpu in [0,1]:
 b=base+['--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','-e',f'HIP_VISIBLE_DEVICES={gpu}']
 execute(f'g{gpu}-grouped-topk',b+['--entrypoint','/bin/sh','-e','STRATA_EXP_MODE=16','-e','STRATA_HIP_SWIGLU_FUSED=1',image,'-c',"./native_grouped_parity && ctest --output-on-failure -R '^qsa_topk_gfx906_guarded$'"])
 for g in [1,8,32]:
  for t in [1,2,4,8]:
   execute(f'g{gpu}-G{g}-T{t}',b+['--entrypoint','/src/build-906/native_expert_bench','-v','/home/radneon/models:/sata:ro','-e','STRATA_HIP_SWIGLU_FUSED=1',image,model,'0,23,47',str(g),str(t),'15','16','50'])
(O/'complete.json').write_text(json.dumps({'passed':True,'native_real_weight_cases':72,'binary_sha256':receipt['binary_sha256'],'note':'Fused h scratch is unwritten and not an independent SwiGLU oracle; gate/up, active Q8 and expert output comparisons are meaningful.'},indent=2))
print('Clean stack component qualification complete',flush=True)
