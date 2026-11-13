from pathlib import Path
import subprocess,json,time,urllib.request,hashlib
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/q5-head-r4';M=O/'micro'
assert json.loads((O/'build.json').read_text())['exit_code']==0;M.mkdir(exist_ok=False)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
rows=[];restore={};first=None;first_q=None;decision={}
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 for tag,mode in [('A1',0),('B1',1),('B2',1),('A2',0)]:
  prefix='/src/'+str((M/tag).relative_to(R))
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-q5-head-micro','--entrypoint','/src/logs/perf-campaign-20261008/q5-head-r4/gfx906_q5_head_row_probe','-e','HIP_VISIBLE_DEVICES=0','-e','STRATA_MMVQ_WAVE=1','-e','STRATA_GFX906_Q5_HEAD_R4='+str(mode),'-e','STRATA_GFX906_Q5_HEAD_TRACE=1','-v',str(R)+':/src',image,'--cols','4','--out',prefix,'--warmup','5','--samples','30','--repeats','2','--graph-nodes','8']
  if tag!='A1':cmd+=['--q8-input','/src/'+str((M/'A1.q8_1.bin').relative_to(R))]
  log=M/(tag+'.log')
  with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  text=log.read_text();assert p.returncode==0,text
  assert ('ncols=4 R=%d CTAs=%d'%((4 if mode else 1),(15520 if mode else 62080))) in text,'wrong head dispatch'
  meta=json.loads((M/(tag+'.header.json')).read_text());raw=(M/(tag+'.f32')).read_bytes();q=(M/(tag+'.q8_1.bin')).read_bytes()
  if first is None:first=raw;first_q=q;input_hash=meta['input_fnv1a64']
  assert raw==first and q==first_q and meta['input_fnv1a64']==input_hash,'projection output/input mismatch'
  row={'tag':tag,'mode':mode,'median_ms':meta['projection_ms_median'],'output_bytes':len(raw),'output_sha256':hashlib.sha256(raw).hexdigest(),'input_hash':input_hash,'timing_mode':meta['timing_mode'],'log':text}
  rows.append(row);print(json.dumps(row),flush=True)
 old=sum(x['median_ms'] for x in rows if x['mode']==0)/2
 new=sum(x['median_ms'] for x in rows if x['mode']==1)/2
 decision={'full_output_bytes_equal':True,'old_projection_ms':old,'new_projection_ms':new,'latency_reduction_pct':100*(1-new/old),'throughput_gain_pct':100*(old/new-1),'expand_parity':new<=old*.75}
finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True,'health':h};break
  except (OSError,ValueError):pass
  time.sleep(5)
 (M/'summary.json').write_text(json.dumps({'results':rows,'decision':decision,'production_restore':restore},indent=2));assert restore.get('passed')
print(json.dumps(decision),flush=True)
