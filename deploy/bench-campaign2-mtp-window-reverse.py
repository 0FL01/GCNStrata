from pathlib import Path
import subprocess,json,time,datetime,hashlib,queue,threading,urllib.request,os,traceback
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261007/mtp-window-reverse';cfgpath=R/'configs/shefowl-hybrid-200k.json';original=cfgpath.read_bytes();cfg=json.loads(original)
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
def gpu(): return []
def run(tag,kv,n,new,context,candidate,controlled):
 mode=16
 fused=candidate
 temperature=0 if controlled else 1
 top_p=1 if controlled else .95
 top_k=1 if controlled else 20
 seed=54321 if tag.startswith('ru') else 12345
 assert not subprocess.check_output(['docker','ps','-q']).strip(),'another container active'
 before=mem();assert before['MemAvailable']>=64*1024**3,'insufficient available RAM for pinnedPLE+expertarena'
 ids=json.loads((O/(('prompt-ru-' if tag.startswith('ru') else 'prompt-')+str(n)+'.ids.json')).read_text());assert len(ids)==n
 args=list(cfg['args']);args=[x for x in args if x!='--vision'];args[args.index('--kv')+1]=kv;args[args.index('--max-context')+1]=str(context)
 for flag,value in [('--expert-cache','9900'),('--mtp-window',str(16384 if current_cpu else 32768)),('--vram-reserve-later-mib',str(623 if current_cpu else 600))]:
  if flag in args:args[args.index(flag)+1]=value
  else:args += [flag,value]
 if controlled:args+=['--adapt-every','0']
 args+=['--layer-split',str(cfg['layer_split']),'--prompt-cache','0','--prompt-cache-every','0','--eos-ids','2147483647']
 name='strata-bench-'+tag.lower()
 cmd=['docker','run','--rm','-i','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--ulimit','memlock=-1:-1','--entrypoint','/src/logs/perf-campaign-20261007/strata-hc-up-selected','-e','STRATA_EXP_MODE='+str(mode),'-e','STRATA_MMVF_ROWS='+str(candidate),'-e','STRATA_HIP_SWIGLU_FUSED='+str(candidate),'-e','STRATA_GDN_HEAD='+str(candidate),'-e','STRATA_GDN_PP='+str(2*candidate),'-e','STRATA_GDN_CONVL2='+str(candidate),'-e','STRATA_GDN_NOY='+str(candidate),'-e','STRATA_TSUM=0','-e','STRATA_Q2_AVX2_SPREAD=1','-e','STRATA_GR_UP_EXACT=0','-e','STRATA_MMVF_RPB=4','-e','HIP_VISIBLE_DEVICES=0,1','-e','STRATA_STAGE_TRIM=1','-e','STRATA_ARENA_MMAP=1','-e','DO_NOT_TRACK=1','-e','HF_HUB_DISABLE_TELEMETRY=1','-v',str(R)+':/src','-v','/home/radneon/models-nvme:/models:ro','-v','/home/radneon/models:/sata:ro',image,'--serve']+args
 m={'tag':tag,'kv':kv,'prompt_tokens':n,'generate':new,'argv':cmd,'started_utc':stamp(),'source_revision':'82f4473+mode16+CPUspread; drafter-window-only','local_patches':['gfx906-0140-return-and-stream-priority-build-fixes','isolated-Q2-down-bitspread-unpack-mode15','API-only/no-telemetry (not involved in native benchmark)'],'binary_sha256':hashlib.sha256((R/'logs/perf-campaign-20261007/strata-hc-up-selected').read_bytes()).hexdigest(),'host_memory_before':before,'vision':'disabled in native text benchmark','prompt_reuse':False,'adapt_every':0 if controlled else 'default','control':'fixed placement greedy' if controlled else 'normal adaptive placement and production sampling','ple':'RAM locked required','resident_kv_tokens':32768,'mtp':'existing StrataQ2_0 draft experts withQ8_0 dense, spec4+suffix-default3','mtp_window':16384 if current_cpu else 32768,'expert_mode':mode,'mmvf_rows':fused,'model_experts_per_layer':512,'context_capacity':context,'model_revision':'55568c1b41d2e381a59a281447fb10935c3c6d0a'}
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
     assert int(info['expert_slots'])==19078 and int(info['expert_slots_primary'])==9900, 'expert residency changed: '+str(info)
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

subprocess.run(['docker','stop','--timeout','60','strata-api'],check=True)
assert not subprocess.check_output(['docker','ps','-q']).strip()
try:
 context=204800
 for arm,current_cpu in [('B2',1),('A2',0)]:
  tag='code65536-'+arm
  r=run(tag,'int8',65536,512,context,1,True)
  results.append(r)
  assert r['passed'],r
finally:
 (O/'summary.json').write_text(json.dumps({'results':results,'production_left_stopped_for_RnD':True,'finished_utc':stamp()},indent=2))
print('MTP-window screen complete',flush=True)
