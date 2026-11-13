from pathlib import Path
import os,json,urllib.request,urllib.error,base64,time,re
O=Path('/src/logs/perf-campaign-20261007/exact-stack-api');key=os.environ['STRATA_API_KEY'];base='http://127.0.0.1:8080'
def call(path,payload=None,auth=True):
 headers={'Authorization':'Bearer '+key} if auth else {}
 data=None
 if payload is not None:data=json.dumps(payload).encode();headers['Content-Type']='application/json'
 try:
  with urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers=headers),timeout=600) as f:return f.status,f.read()
 except urllib.error.HTTPError as e:return e.code,e.read()
code,raw=call('/health',auth=False);health=json.loads(raw);assert code==200 and health['images'] and health['loaded'] and health['api_key']
assert call('/v1/models',auth=False)[0]==401 and call('/',auth=False)[0]==404
png=(O/'red-blue.png').read_bytes();data='data:image/png;base64,'+base64.b64encode(png).decode()
payload={'model':health['model'],'messages':[{'role':'user','content':[{'type':'text','text':'What colors are the two rectangles from left to right? Reply with only the two English color names, in that order.'},{'type':'image_url','image_url':{'url':data}}]}],'max_tokens':40,'temperature':0,'reasoning_effort':'none','stream':False}
results=[]
for trial in range(2):
 start=time.monotonic();code,raw=call('/v1/chat/completions',payload);elapsed=time.monotonic()-start
 response=json.loads(raw);assert code==200,response
 content=response['choices'][0]['message']['content'];text=content.lower();assert 'red' in text and 'blue' in text and text.index('red')<text.index('blue'),content
 result={'trial':trial,'elapsed_s':elapsed,'response':response};results.append(result);print(json.dumps({'trial':trial,'elapsed_s':elapsed,'content':content,'usage':response.get('usage')}),flush=True)
# Text-only requests continue to work after an image request.
code,raw=call('/v1/chat/completions',{'model':health['model'],'messages':[{'role':'user','content':'Compute 17 + 25. Reply with just the decimal integer.'}],'max_tokens':16,'temperature':0,'reasoning_effort':'none'})
answer=json.loads(raw);assert code==200 and '42' in answer['choices'][0]['message']['content']
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if not args or not args[0].endswith(b'/strata-vision'):continue
  maps=(p/'maps').read_text();bad=[x for x in maps.splitlines() if re.search(r'libamdhip|libhsa-runtime|libcuda|libcudart|libggml-(cuda|hip|vulkan)',x)]
  gpu_fds=[]
  for fd in (p/'fd').iterdir():
   try:target=os.readlink(fd)
   except OSError:continue
   if target.startswith(('/dev/kfd','/dev/dri','/dev/nvidia')):gpu_fds.append(target)
  status={}
  for line in (p/'status').read_text().splitlines():
   if line.startswith(('VmRSS:','VmHWM:','Threads:')):
    k,v=line.split(':',1);status[k]=v.strip()
  assert not bad and not gpu_fds,(bad,gpu_fds)
  processes.append({'pid':int(p.name),'argv':[x.decode() for x in args if x],'memory':status,'gpu_libraries':bad,'gpu_device_fds':gpu_fds})
 except (FileNotFoundError,ProcessLookupError):continue
assert len(processes)==1
result={'passed':True,'health':health,'image_trials':results,'text_smoke':answer,'vision_processes':processes,'unauthenticated_models':401,'ui':404,'scope':'synthetic1024x1024 red-left/blue-right image twice and text reuse; CPU backend/mappings/fds checked, not broad vision accuracy'}
(O/'api-result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'passed':True,'vision_processes':processes}),flush=True)
