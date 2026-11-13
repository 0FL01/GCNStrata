import json,os,time,urllib.request,urllib.error
from pathlib import Path
base='http://127.0.0.1:8080';key=os.environ['STRATA_API_KEY'];model='Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M'
results=[]
def req(payload):
 return urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
def plain(messages,max_tokens=128,**kw):
 p={'model':model,'messages':messages,'max_tokens':max_tokens,'reasoning_effort':'none','temperature':1,'top_p':.95,'top_k':20,'stream':False};p.update(kw)
 t=time.monotonic()
 with urllib.request.urlopen(req(p),timeout=600) as f:r=json.load(f)
 assert r['choices'][0]['message']['content'].strip()
 assert 0<r['usage']['completion_tokens']<=max_tokens,r
 results.append({'kind':'nonstream','elapsed_s':time.monotonic()-t,'request':p,'response':r})
 return r
messages=[{'role':'user','content':'Write a Python function named add that returns the sum of its two arguments. No explanation.'}]
p={'model':model,'messages':messages,'max_tokens':128,'temperature':0,'reasoning_effort':'none','stream':True,'stream_options':{'include_usage':True}}
events=[];done=False;t=time.monotonic()
with urllib.request.urlopen(req(p),timeout=180) as f:
 for line in f:
  if not line.startswith(b'data: '):continue
  data=line[6:].strip()
  if data==b'[DONE]':done=True;break
  events.append(json.loads(data))
assert done and events
text=''.join(c.get('delta',{}).get('content','') for e in events for c in e.get('choices',[]))
assert 'def add' in text,text
results.append({'kind':'stream','elapsed_s':time.monotonic()-t,'content':text,'events':events,'done':done})
filler='\n'.join('value_'+str(i)+' = '+str(i)+'  # deterministic filler' for i in range(350))
r=plain([{'role':'user','content':filler+'\nExplain in three short sentences what these assignments do.'}],128)
assert r['usage']['prompt_tokens']>=4000,r['usage']
# Disconnect an active SSE response, then verify that another request succeeds.
p={'model':model,'messages':[{'role':'user','content':'Write a detailed guide to implementing a hash table in Python.'}],'max_tokens':512,'temperature':1,'reasoning_effort':'none','stream':True}
seen=False
with urllib.request.urlopen(req(p),timeout=180) as f:
 for line in f:
  if line.startswith(b'data: ') and line[6:].strip()!=b'[DONE]':
   e=json.loads(line[6:])
   if any(c.get('delta',{}).get('content') for c in e.get('choices',[])):
    seen=True;break
assert seen
r=plain([{'role':'user','content':'Reply with just the word READY.'}],32,temperature=0)
assert 'READY' in r['choices'][0]['message']['content']
results.append({'kind':'disconnect_then_reuse','passed':True})
p={'model':model,'messages':[{'role':'user','content':'Hi'}],'max_tokens':200000,'stream':False}
try:
 urllib.request.urlopen(req(p),timeout=15)
 raise AssertionError('oversized output accepted')
except urllib.error.HTTPError as e:assert e.code==400,e.code
results.append({'kind':'oversized_request','http_status':400})
Path('/src/logs/live-extended-result.json').write_text(json.dumps({'passed':True,'results':results},ensure_ascii=False,indent=2))
print(json.dumps({'passed':True,'checks':[x['kind'] for x in results],'long_prompt_tokens':results[1]['response']['usage']['prompt_tokens']}))

