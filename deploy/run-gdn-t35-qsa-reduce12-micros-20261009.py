from pathlib import Path
import subprocess,json,time,urllib.request,hashlib,datetime,re
R=Path('/home/radneon/strata');Q=R/'logs/perf-campaign-20261009/qsa-reduce12-probe';G=R/'logs/perf-campaign-20261009/gdn-ab-token-probe'
cfg=R/'configs/shefowl-hybrid-200k.json';cfgsha=hashlib.sha256(cfg.read_bytes()).hexdigest()
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
names=['strata-qsa-reduce12-micro','strata-gdn-ab-t35'];results=[];coverage=[];restore={};decision={}
def gpu():
 rows=[]
 for d in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
  if not (d/'mem_info_vram_total').exists():continue
  row={'pci':d.resolve().name,'mode':(d/'power_dpm_force_performance_level').read_text().strip()}
  assert row['mode']=='high'
  for h in (d/'hwmon').glob('hwmon*'):
   p=h/'power1_cap'
   if p.exists():assert int(p.read_text())==190000000
   for i in [1,2,3]:
    p=h/f'temp{i}_input';c=h/f'temp{i}_crit'
    if p.exists():
     row[f'temp{i}']=int(p.read_text())
     if c.exists():assert row[f'temp{i}']<int(c.read_text())-10000
  rows.append(row)
 assert len(rows)==2
 return rows
def run(cmd,root,tag):
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 start=time.monotonic();samples=[]
 with (root/(tag+'.stdout')).open('x') as out,(root/(tag+'.stderr')).open('x') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err)
  while p.poll() is None:
   samples.append({'elapsed_s':time.monotonic()-start,'gpu':gpu()})
   if time.monotonic()-start>180:raise TimeoutError(tag)
   time.sleep(1)
 (root/(tag+'.sensors.json')).write_text(json.dumps(samples))
 assert p.returncode==0,(tag,p.returncode,(root/(tag+'.stderr')).read_text()[-2000:])
 return (root/(tag+'.stdout')).read_text(),(root/(tag+'.stderr')).read_text()
def common(name,root,visible,entry):
 return ['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','-e','HIP_VISIBLE_DEVICES='+visible,'-v',str(root)+':/work','--entrypoint',entry]
def qsa(dev,arm,flag,nq=32,ctx=65536,masked=False):
 tag=f'd{dev}-{arm}-n{nq}-c{ctx}';base=Q/tag
 cmd=common(names[0],Q,str(dev),'/work/gfx906_qsa_reduce12_probe')+['-e','STRATA_GFX906_ATTN_QUERY_SWIZZLE=1','-e','STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE=1','-e','STRATA_ATTN_LANECELL=0','-e','STRATA_GFX906_ATTN_REDUCE12='+str(flag),image,str(nq),str(ctx),str(50 if nq==32 else 5),'/work/'+tag+'.bin']
 if masked:cmd.append('--masked-pages')
 out,err=run(cmd,Q,tag)
 assert 'actual-full-specialization='+str(flag) in err,(tag,err)
 assert 'attn-query-swizzle: active=1 ' in err,(tag,err)
 assert 'raw_score_parity=PASS' in out and 'PASS finite_written_guards_and_repeat' in out
 m=re.search(r'full_chunk_merge_median_ms=([0-9.]+) score_only_median_ms=([0-9.]+)',out);assert m
 row={'device':dev,'arm':arm,'candidate':bool(flag),'nq':nq,'context':ctx,'masked_pages':masked,'full_ms':float(m[1]),'score_only_ms':float(m[2]),'full_sha256':hashlib.sha256(Path(str(base)+'.bin').read_bytes()).hexdigest(),'scores_sha256':hashlib.sha256(Path(str(base)+'.bin.scores.bin').read_bytes()).hexdigest(),'actual_route_trace':True,'exit_code':0}
 results.append(row);print(json.dumps(row),flush=True);return row
try:
 assert json.loads((G/'build-t35.json').read_text())['exit_code']==0
 assert json.loads((Q/'build-linked.json').read_text())['exit_code']==0
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 for dev in [0,1]:
  out,err=run(common(names[1],G,'0,1','/work/gdn_ab_token_probe_t35')+[image,str(dev)],G,f't35-device{dev}')
  row=json.loads(out);assert len(row['results'])==2
  coverage.append(row);print(json.dumps({'gdn_t35':row}),flush=True)
 (G/'coverage-t35.json').write_text(json.dumps(coverage,indent=2))
 for dev in [0,1]:
  rows=[qsa(dev,a,b) for a,b in [('A1',0),('B1',1),('B2',1),('A2',0)]]
  assert len(set(x['full_sha256'] for x in rows))==1 and len(set(x['scores_sha256'] for x in rows))==1,'QSA ABBA byte mismatch'
  a=(rows[0]['full_ms']+rows[3]['full_ms'])/2;b=(rows[1]['full_ms']+rows[2]['full_ms'])/2
  decision[str(dev)]={'full_latency_reduction_pct':100*(1-b/a),'passes_component_gate':100*(1-b/a)>=12}
 assert len(set(x['full_sha256'] for x in results))==1 and len(set(x['scores_sha256'] for x in results))==1,'cross GPU byte mismatch'
 decision['both_pass']=all(decision[str(dev)]['passes_component_gate'] for dev in [0,1])
 if decision['both_pass']:
  for dev in [0,1]:
   for nq,ctx,masked in [(1,4,False),(33,4096,True)]:
    a=qsa(dev,'edge-A',0,nq,ctx,masked);b=qsa(dev,'edge-B',1,nq,ctx,masked)
    assert a['full_sha256']==b['full_sha256'] and a['scores_sha256']==b['scores_sha256'],'edge byte mismatch'
 print(json.dumps({'decision':decision}),flush=True)
finally:
 for name in names:
  if subprocess.check_output(['docker','ps','-q','--filter','name=^/'+name+'$']).strip():
   subprocess.run(['docker','stop','--time','30',name],check=False)
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True,'health':h};break
  except (OSError,ValueError):pass
  time.sleep(5)
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==cfgsha
 (Q/'summary.json').write_text(json.dumps({'results':results,'gdn_t35':coverage,'decision':decision,'production_restore':restore,'diagnostic_component_only':True,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
 assert restore.get('passed')
print('SEQUENTIAL MICRO SCREEN COMPLETE',flush=True)
