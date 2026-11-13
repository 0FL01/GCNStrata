from pathlib import Path
import subprocess,json,time,urllib.request,hashlib
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-pr-topk-20261008');O=R/'logs/perf-campaign-20261008/counted-topk-boundary'
O.mkdir(exist_ok=False)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
cmd=['docker','run','--rm','--network','none','--name','strata-counted-topk-build','--entrypoint','/bin/sh','-v',str(W)+':/src','-v',str(R/'third_party/llama.cpp')+':/src/third_party/llama.cpp:ro','-w','/src',image,'-c','cmake --build build-906 --target qsa_select_bench -j4']
with (O/'build.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(O/'build.json').write_text(json.dumps({'argv':cmd,'exit_code':p.returncode},indent=2));assert p.returncode==0
rows=[];restore={}
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 for d in Path('/sys/class/drm').glob('card[0-9]*/device'):
  if (d/'mem_info_vram_total').exists():assert (d/'power_dpm_force_performance_level').read_text().strip()=='high'
 for tag,dev,exe,argv in [('gpu0-bench',0,'qsa_select_bench',['135168','256','100','204800']),('gpu0-parity',0,'qsa_topk_parity',['135168','256','100','204800','1']),('gpu1-bench',1,'qsa_select_bench',['135168','256','100','204800']),('gpu1-parity',1,'qsa_topk_parity',['135168','256','100','204800','1']),('gpu0-repeat',0,'qsa_select_bench',['135168','256','100','204800'])]:
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-counted-topk-diagnostic','--entrypoint','/src/build-906/'+exe,'-e','HIP_VISIBLE_DEVICES='+str(dev),'-e','STRATA_GFX906_TOPK_REG=2','-v',str(W)+':/src:ro',image]+argv
  log=O/(tag+'.log')
  with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  row={'tag':tag,'gpu':dev,'exit_code':p.returncode,'binary_sha256':hashlib.sha256((W/'build-906'/exe).read_bytes()).hexdigest(),'argv':argv,'log':log.read_text()}
  rows.append(row);print(json.dumps(row),flush=True);assert p.returncode==0
finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True,'health':h};break
  except (OSError,ValueError):pass
  time.sleep(5)
 (O/'summary.json').write_text(json.dumps({'results':rows,'production_restore':restore,'scope':'one actual nq256 boundary; diagnostic only, no dispatch change'},indent=2));assert restore.get('passed')
print('COUNTED TOPK BOUNDARY DIAGNOSTIC COMPLETE',flush=True)
