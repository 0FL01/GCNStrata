from pathlib import Path
import subprocess,json,time,urllib.request,hashlib,re
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/qsa-score-grid-order';M=O/'micro'
assert json.loads((O/'build.json').read_text())['exit_code']==0;M.mkdir(exist_ok=False)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
rows=[];budgets=[];restore={};first=None;decision={}
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 for d in Path('/sys/class/drm').glob('card[0-9]*/device'):
  if (d/'mem_info_vram_total').exists():
   assert (d/'power_dpm_force_performance_level').read_text().strip()=='high'
   assert all(int(p.read_text())==190000000 for p in (d/'hwmon').glob('hwmon*/power1_cap'))
 for tag,mode in [('A1',0),('B1',1),('B2',1),('A2',0)]:
  dump=M/(tag+'.bin');log=M/(tag+'.log')
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-score-grid-micro','--entrypoint','/src/logs/perf-campaign-20261008/qsa-score-grid-order/qsa_select_bench','-e','HIP_VISIBLE_DEVICES=0','-e','STRATA_GFX906_TOPK_REG=2','-e','STRATA_GFX906_SCORE_QUERY_FAST='+str(mode),'-e','STRATA_SCORE_QUERY_FAST_TRACE=1','-e','STRATA_QSA_SCORE_DUMP=/src/'+str(dump.relative_to(R)),'-v',str(R)+':/src',image,'135168','256','50','204800']
  with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  t=log.read_text();assert p.returncode==0,t
  assert ('mode='+('query-fast' if mode else 'key-fast')+' nq=256 reach=33793 stride=51202') in t,'wrong scorer dispatch'
  content=dump.read_bytes()
  if first is None:first=content
  assert content==first,'full score/ID/input-header bytes differ'
  timings=re.search(r'warp scorer ([0-9.]+) ms; scorer\+current top-k ([0-9.]+) ms',t);assert timings,t
  row={'tag':tag,'mode':mode,'scorer_ms':float(timings.group(1)),'combined_ms':float(timings.group(2)),'dump_bytes':len(content),'dump_sha256':hashlib.sha256(content).hexdigest(),'log':t}
  rows.append(row);print(json.dumps(row),flush=True)
 a=[r for r in rows if r['mode']==0];b=[r for r in rows if r['mode']==1]
 old=sum(x['scorer_ms'] for x in a)/2;new=sum(x['scorer_ms'] for x in b)/2
 oldc=sum(x['combined_ms'] for x in a)/2;newc=sum(x['combined_ms'] for x in b)/2
 decision={'all_score_and_id_bytes_equal':True,'scorer_old_ms':old,'scorer_new_ms':new,'scorer_throughput_gain_pct':100*(old/new-1),'combined_old_ms':oldc,'combined_new_ms':newc,'combined_throughput_gain_pct':100*(oldc/newc-1),'clear_signal':oldc/newc>=1.20}
 print(json.dumps(decision),flush=True)
 # Separate baseline-only conversion budget; sequential, never concurrent with scorer timing.
 for dev in [0,1]:
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-dense-dequant-budget','--entrypoint','/src/logs/perf-campaign-20261008/qsa-score-grid-order/gfx906_dense_dequant_budget','-e','HIP_VISIBLE_DEVICES='+str(dev),'-v',str(R)+':/src',image,'--device','0','--warmups','5','--reps','20']
  log=M/('dequant-gpu%d.log'%dev)
  with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  row={'gpu':dev,'exit_code':p.returncode,'log':log.read_text()};budgets.append(row);print(json.dumps(row),flush=True);assert p.returncode==0
finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True,'health':h};break
  except (OSError,ValueError):pass
  time.sleep(5)
 (M/'summary.json').write_text(json.dumps({'results':rows,'decision':decision,'dense_dequant_budget':budgets,'production_restore':restore},indent=2));assert restore.get('passed')
print('SCORE GRID AND DEQUANT DIAGNOSTICS COMPLETE',flush=True)
