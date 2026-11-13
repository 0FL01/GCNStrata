from pathlib import Path
import subprocess,json,time,datetime,hashlib,queue,threading,urllib.request,os,traceback
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/hc-mmq-combined/split26-placement';cfgpath=R/'configs/shefowl-hybrid-200k.json';original=cfgpath.read_bytes();cfg=json.loads(original)
for flag,value in {'--pack':'/src/runtime/shefowl-hybrid','--native':'/sata/shefowl-qwen38-hybrid/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0-00001-of-00002.gguf','--ple-gguf':'/sata/shefowl-qwen38-hybrid/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0-00002-of-00002.gguf','--expert-profile':'/src/data/expert-profile.bin'}.items():
 cfg['args'][cfg['args'].index(flag)+1]=value

assert (O/'fixture-manifest.json').exists();out=O/'runs';out.mkdir(exist_ok=False)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
assert cfg.get('vision',{}).get('gpu') is False
(O/'production-config-before-benchmark.json').write_bytes(original)
restore={};results=[]
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def mem():
 return {a.rstrip(':'):int(b)*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapTotal:','SwapFree:')) for a,b,*_ in [line.split()]}
devices=[d.resolve() for d in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')) if (d/'mem_info_vram_total').exists()]
assert len(devices)==2 and {d.name for d in devices}=={'0000:05:00.0','0000:08:00.0'}
saved={str(d):(d/'power_dpm_force_performance_level').read_text().strip() for d in devices}
assert all(v=='high' for v in saved.values())
(O/'original-power-modes.json').write_text(json.dumps(saved,indent=2))
def set_perf(mode):
 assert mode in ['auto','high']
 for d in devices:
  subprocess.run(['sudo','-n','tee',str(d/'power_dpm_force_performance_level')],input=mode+chr(10),text=True,stdout=subprocess.DEVNULL,check=True)
  assert (d/'power_dpm_force_performance_level').read_text().strip()==mode
def gpu():
 rows=[]
 for d in devices:
  row={'pci':d.name,'mode':(d/'power_dpm_force_performance_level').read_text().strip()}
  for h in (d/'hwmon').glob('hwmon*'):
   for name in ['temp1_input','temp2_input','temp3_input','power1_input','power1_cap','freq1_input','freq2_input']:
    p=h/name
    if p.exists():row[name]=int(p.read_text())
   for i in [1,2,3]:
    crit=h/f'temp{i}_crit'
    if crit.exists() and f'temp{i}_input' in row:
     assert row[f'temp{i}_input'] < int(crit.read_text())-10000, 'thermal safety margin reached'
  assert row.get('power1_cap')==190000000, 'power limit changed unexpectedly'
  rows.append(row)
 return rows

def run(tag,kv,n,new,context,candidate,controlled):
 mode=16 if current_cpu else 15
 fused=candidate
 temperature=0 if controlled else 1
 top_p=1 if controlled else .95
 top_k=1 if controlled else 20
 seed=54321 if tag.startswith('ru') else 12345
 assert not subprocess.check_output(['docker','ps','-q']).strip(),'another container active'
 before=mem();assert before['MemAvailable']>=64*1024**3,'insufficient available RAM for pinnedPLE+expertarena'
 ids=json.loads((O/(('prompt-ru-' if tag.startswith('ru') else 'prompt-')+str(n)+'.ids.json')).read_text());assert len(ids)==n
 args=list(cfg['args']);args=[x for x in args if x!='--vision'];args[args.index('--kv')+1]=kv;args[args.index('--max-context')+1]=str(context)
 for flag,value in [('--expert-cache',str(current_primary)),('--mtp-window',str(16384 if current_cpu else 32768)),('--vram-reserve-later-mib',str(623 if current_cpu else 600))]:
  if flag in args:args[args.index(flag)+1]=value
  else:args += [flag,value]
 if controlled:args+=['--adapt-every','0']
 args+=['--layer-split',str(current_split),'--prompt-cache','0','--prompt-cache-every','0','--eos-ids','2147483647']
 engine=R/('logs/perf-campaign-20261008/hc-mmq-combined/strata' if current_combined else 'logs/perf-campaign-20261007/strata-clean-exact-stack')
 name='strata-bench-'+tag.lower()
 cmd=['docker','run','--rm','-i','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--ulimit','memlock=-1:-1','--entrypoint','/src/'+str(engine.relative_to(R)),'-e','STRATA_EXP_MODE='+str(mode),'-e','STRATA_MMVF_ROWS='+str(candidate),'-e','STRATA_HIP_SWIGLU_FUSED='+str(candidate),'-e','STRATA_GDN_HEAD='+str(candidate),'-e','STRATA_GDN_PP='+str(2*candidate),'-e','STRATA_GDN_CONVL2='+str(candidate),'-e','STRATA_GDN_NOY='+str(candidate),'-e','STRATA_TSUM=0','-e','STRATA_Q2_AVX2_SPREAD='+str(current_cpu),'-e','STRATA_GFX906_TOPK_REG='+str(2*current_cpu),'-e','STRATA_GR_UP_EXACT=0','-e','STRATA_MMVF_RPB=4','-e','STRATA_GFX906_HC_F16='+str(current_combined),'-e','STRATA_HC_F16_AUDIT=0','-e','STRATA_HC_F16_SCALE_PROBE=1','-e','STRATA_HC_F16_EXACT_INPUT=1','-e','STRATA_HC_F16_TRACE=1','-e','STRATA_PREFILL_MMQ=1','-e','STRATA_GFX906_MMQ_OPT_CAP='+str(current_cap),'-e','STRATA_MMQ_OPT_TRACE=1','-e','HIP_VISIBLE_DEVICES=0,1','-e','STRATA_STAGE_TRIM=1','-e','STRATA_ARENA_MMAP=1','-e','DO_NOT_TRACK=1','-e','HF_HUB_DISABLE_TELEMETRY=1','-v',str(R)+':/src','-v','/home/radneon/models-nvme:/models:ro','-v','/home/radneon/models:/sata:ro',image,'--serve']+args
 m={'tag':tag,'kv':kv,'prompt_tokens':n,'generate':new,'argv':cmd,'started_utc':stamp(),'source_revision':('b6731bf+entry-deployment-patches' if current_cpu else 'original-entry-production'),'local_patches':['gfx906-0140-return-and-stream-priority-build-fixes','isolated-Q2-down-bitspread-unpack-mode15','API-only/no-telemetry (not involved in native benchmark)'],'binary_sha256':hashlib.sha256(engine.read_bytes()).hexdigest(),'host_memory_before':before,'vision':'disabled in native text benchmark','prompt_reuse':False,'adapt_every':0 if controlled else 'default','control':'fixed placement greedy' if controlled else 'normal adaptive placement and production sampling','ple':'RAM locked required','resident_kv_tokens':32768,'mtp':'existing StrataQ2_0 draft experts withQ8_0 dense, spec4+suffix-default3','mtp_window':16384 if current_cpu else 32768,'power_profile':current_profile,'expert_mode':mode,'mmvf_rows':fused,'model_experts_per_layer':512,'context_capacity':context,'model_revision':'55568c1b41d2e381a59a281447fb10935c3c6d0a'}
 (out/(tag+'.manifest.json')).write_text(json.dumps(m,indent=2));print(json.dumps({'event':'start','tag':tag,'utc':stamp()}),flush=True)
 p=None;done=None;info={};tokens=[];samples=[];started=time.monotonic();request_start=None;ready_time=None;messages=queue.Queue()
 try:
  with (out/(tag+'.stderr')).open('x') as err,(out/(tag+'.stdout')).open('x') as stdout:
   p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
   def reader():
    try:
     for line in p.stdout:messages.put(line)
    finally:messages.put(None)
   t=threading.Thread(target=reader,daemon=True);t.start();last_sample=0
   while True:
    now=time.monotonic()
    if now-started>5400:raise TimeoutError('bounded native run exceeded90minutes')
    if (O/'cancel').exists():raise RuntimeError('benchmark cancellation marker')
    if now-last_sample>=5:samples.append({'elapsed_s':now-started,'gpus':gpu()});last_sample=now
    try:line=messages.get(timeout=5)
    except queue.Empty:continue
    if line is None:raise RuntimeError('engine ended before DONE')
    stdout.write(line);stdout.flush();line=line.strip()
    if line.startswith('ERR '):raise RuntimeError(line)
    if line.startswith('INFO '):info=dict(part.split('=',1) for part in line.split()[1:] if '=' in part)
    if line.startswith('READY '):
     assert ready_time is None and line.split()[1]==str(context),line
     ready_time=time.monotonic();m['load_wall_s']=ready_time-started;m['engine_info']=info;m['gpu_at_ready']=gpu();m['host_memory_at_ready']=mem()
     actual_free=vram_free()
     m['vram_free_at_ready']=actual_free
     assert all(v >= max(256*1024**2,baseline_free[k]-32*1024**2) for k,v in actual_free.items()), 'VRAM admission margin below baseline budget'
     if admission_only:
      assert int(info['context'])==context and info.get('kv')==kv
      p.stdin.write('QUIT\n');p.stdin.flush();p.stdin.close();code=p.wait(timeout=120);t.join(timeout=5);assert code==0
      result={'tag':tag,'passed':True,'admission_only':True,'engine_info':info,'vram_free_at_ready':actual_free}
      (out/(tag+'.result.json')).write_text(json.dumps(result,indent=2))
      print(json.dumps({'event':'admission_complete',**result}),flush=True)
      return result
     assert int(info['expert_slots'])==19078 and int(info['expert_slots_primary'])==current_primary, 'expert residency changed: '+str(info)
     assert info.get('kv')==kv and int(info['context'])==context and int(info['lookup'])==3 and int(info['mtp_max'])==4
     assert 'locked in RAM' in (out/(tag+'.stderr')).read_text(),'PLE did not lock in RAM'
     (out/(tag+'.manifest.json')).write_text(json.dumps(m,indent=2))
     request_start=time.monotonic();p.stdin.write('GEN '+str(new)+' temperature='+str(temperature)+' top_p='+str(top_p)+' top_k='+str(top_k)+' seed='+str(seed)+' '+','.join(map(str,ids))+'\n');p.stdin.flush()
    elif line.startswith('T '):tokens.append(int(line.split()[1]))
    elif line.startswith('DONE '):done=line.split();m['request_wall_s']=time.monotonic()-request_start;break
   p.stdin.write('QUIT\n');p.stdin.flush();p.stdin.close();code=p.wait(timeout=120);t.join(timeout=5)
   while not messages.empty():
    line=messages.get()
    if line is not None:stdout.write(line)
   assert code==0,code
  assert len(done)>=15 and int(done[1])==len(tokens)==new and int(done[2])==n and done[5]=='length'
  assert int(done[8])==0 and int(done[14])==n,'prompt was reused/not fully processed'
  assert all(0<=x<248320 for x in tokens)
  prompt_ms=float(done[3]);decode_ms=float(done[4]);assert prompt_ms>0 and decode_ms>0
  result={'tag':tag,'passed':True,'kv':kv,'prompt_tokens':n,'actual_outputs':new,'prompt_ms':prompt_ms,'decode_ms':decode_ms,'PP':n*1000/prompt_ms,'TG':new*1000/decode_ms,'engine_total_s':(prompt_ms+decode_ms)/1000,'request_wall_s':m['request_wall_s'],'load_wall_s':m['load_wall_s'],'accepted':int(done[6]),'offered':int(done[7]),'cache_hits':int(done[9]),'lookups':int(done[10]),'prompt_reused':int(done[8]),'prompt_read':int(done[14]),'output_ids':tokens,'engine_info':info,'sampling':{'temperature':temperature,'top_p':top_p,'top_k':top_k,'seed':seed},'metric_scope':'PP=reported full prompt count /native prompt_ms; last prompt token executes in first decode window. TG=actual outputs/native decode_ms. Native wall/load separate. No quality or equal-token claim across KV quantizations.'}
 except BaseException as e:
  result={'tag':tag,'passed':False,'kv':kv,'prompt_tokens':n,'error':type(e).__name__+': '+str(e),'actual_outputs':len(tokens)}
  print(json.dumps({'event':'case_failed',**result}),flush=True)
 finally:
  if p is not None and p.poll() is None:
   subprocess.run(['docker','stop','--time','30',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  m.update(finished_utc=stamp(),elapsed_s=time.monotonic()-started,native_exit=p.returncode if p else None,host_memory_after=mem())
  (out/(tag+'.manifest.json')).write_text(json.dumps(m,indent=2));(out/(tag+'.gpu-samples.json')).write_text(json.dumps(samples))
 (out/(tag+'.result.json')).write_text(json.dumps(result,indent=2));print(json.dumps({'event':'case_complete',**{k:v for k,v in result.items() if k!='output_ids'}}),flush=True);return result



import re
assert json.loads((O.parent/'build.json').read_text())['exit_code']==0
current_cpu=1;current_profile='high';restore={};micro=[]
def vram_free():
 return {d.name:int((d/'mem_info_vram_total').read_text())-int((d/'mem_info_vram_used').read_text()) for d in devices}
baseline_free=vram_free()
(O/'admission-budget.json').write_text(json.dumps({'baseline_free':baseline_free,'minimum_free_mib':256,'maximum_baseline_headroom_loss_mib':32,'reserve_unchanged':True},indent=2))
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 current_combined=1;current_cap=32;current_split=26;current_primary=9900;admission_only=True
 admission=run('admit-split26','int8',65536,0,204800,1,True)
 results.append(admission);assert admission['passed'] and admission.get('admission_only'),admission
 delta=19078-int(admission['engine_info']['expert_slots'])
 assert 0<=delta<=256,'Unexpected cache budget delta; no reserve reduction is permitted'
 current_primary=9900+delta;admission_only=False
 reference=next(r for r in json.loads((O.parent/'qualify-resume/summary.json').read_text())['results'] if r['tag']=='code65536-combined')
 candidate=run('code65536-split26','int8',65536,1024,204800,1,True)
 results.append(candidate);assert candidate['passed'],candidate
 assert candidate['output_ids']==reference['output_ids'],'placement changed output IDs; no further test'
 decision={'scope':'split plus changed expert placement, not fixed-placement isolation','primary_slots':current_primary,'total_slots':19078,'reserve_later_mib':623,'historical_PP_gain_pct':100*(candidate['PP']/reference['PP']-1),'historical_TG_gain_pct':100*(candidate['TG']/reference['TG']-1),'reference_time_separated':True}
 if candidate['PP']>reference['PP']*1.015 and candidate['TG']>=reference['TG']*.99:
  current_split=27;current_primary=9900
  control=run('code65536-control27','int8',65536,1024,204800,1,True)
  results.append(control);assert control['passed'],control
  assert control['output_ids']==candidate['output_ids'],'matched control output mismatch'
  decision.update(matched_PP_gain_pct=100*(candidate['PP']/control['PP']-1),matched_TG_gain_pct=100*(candidate['TG']/control['TG']-1),matched_control_completed=True)
 else:decision['matched_control_completed']=False
 (O/'decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision),flush=True)

finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True,'health':h};break
  except (OSError,ValueError):pass
  time.sleep(5)
 (O/'summary.json').write_text(json.dumps({'results':results,'micro':micro,'production_restore':restore,'screen_only':True,'finished_utc':stamp()},indent=2));assert restore.get('passed')
print('HC MMQ COMBINED QUALIFICATION COMPLETE',flush=True)
