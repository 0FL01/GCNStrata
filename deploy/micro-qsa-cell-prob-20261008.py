from pathlib import Path
import subprocess,json,hashlib,time,urllib.request,re
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/qsa-cell-prob';M=O/'micro'
assert json.loads((O/'build.json').read_text())['exit_code']==0
M.mkdir(exist_ok=False)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
rows=[];restore={};first=None;input_hash=None;decision={}
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip(),'another GPU/build container active'
 for d in Path('/sys/class/drm').glob('card[0-9]*/device'):
  if (d/'mem_info_vram_total').exists():
   assert (d/'power_dpm_force_performance_level').read_text().strip()=='high'
   assert all(int(p.read_text())==190000000 for p in (d/'hwmon').glob('hwmon*/power1_cap'))
 for tag,mode in [('A1',0),('B1',1),('B2',1),('A2',0)]:
  output=M/(tag+'.bin')
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-qsa-cell-micro','--entrypoint','/src/logs/perf-campaign-20261008/qsa-cell-prob/gfx906_qsa_cell_prob_probe','-e','HIP_VISIBLE_DEVICES=0','-e','STRATA_ATTN_LANECELL=0','-e','STRATA_GFX906_ATTN_CELL_PROB='+str(mode),'-e','STRATA_ATTN_CELL_PROB_TRACE=1','-v',str(R)+':/src',image,'32','65536','20','/src/'+str(output.relative_to(R))]
  log=M/(tag+'.log')
  with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  t=log.read_text();assert p.returncode==0,t
  assert ('active=%d nq=32 cap=2051 kv_mode=1'%mode) in t,'requested route did not activate'
  h=re.search(r'INPUT_FNV1A64=([0-9a-f]+)',t).group(1)
  content=output.read_bytes();assert len(content)==32*24*256*4
  if first is None:first=content;input_hash=h
  assert content==first and h==input_hash,'full output or input mismatch'
  row={'tag':tag,'mode':mode,'exit_code':p.returncode,'median_ms':float(re.search(r'median_ms=([0-9.]+)',t).group(1)),'input_hash':h,'output_sha256':hashlib.sha256(content).hexdigest(),'log':t}
  rows.append(row);print(json.dumps(row),flush=True)
 old=sum(x['median_ms'] for x in rows if x['mode']==0)/2
 new=sum(x['median_ms'] for x in rows if x['mode']==1)/2
 decision={'mean_old_median_ms':old,'mean_new_median_ms':new,'component_throughput_gain_pct':100*(old/new-1),'all_output_bytes_equal':True,'expand_parity':old/new>=1.10,'estimated_whole_prompt_saving_pct_under_profile_model':13.57*(1-new/old)}
finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True,'health':h};break
  except (OSError,ValueError):pass
  time.sleep(5)
 (M/'summary.json').write_text(json.dumps({'results':rows,'decision':decision,'production_restore':restore},indent=2))
 assert restore.get('passed')
print(json.dumps(decision),flush=True)
