import json,os,time,urllib.request,urllib.error
from pathlib import Path
base='http://127.0.0.1:8080';key=os.environ['STRATA_API_KEY']
def call(path,payload=None,auth=True,timeout=600):
 headers={'Authorization':'Bearer '+key} if auth else {}
 data=None
 if payload is not None:data=json.dumps(payload).encode();headers['Content-Type']='application/json'
 req=urllib.request.Request(base+path,data=data,headers=headers)
 try:
  with urllib.request.urlopen(req,timeout=timeout) as f:return f.status,f.read()
 except urllib.error.HTTPError as e:return e.code,e.read()
code,raw=call('/health',auth=False,timeout=5);health=json.loads(raw)
assert code==200 and health['loaded'] and health['api_key'] and health['max_context']==131072
assert call('/v1/models',auth=False,timeout=5)[0]==401
assert call('/',auth=False,timeout=5)[0]==404
code,raw=call('/v1/models');models=json.loads(raw);assert code==200
model='Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M';assert models['data'][0]['id']==model
prompt={'model':model,'messages':[{'role':'user','content':'Compute 17 + 25. Reply with just the decimal integer.'}],'max_tokens':32,'temperature':0,'reasoning_effort':'none','stream':False}
t=time.monotonic();code,raw=call('/v1/chat/completions',prompt);elapsed=time.monotonic()-t
response=json.loads(raw);assert code==200,response
content=response['choices'][0]['message']['content'];assert isinstance(content,str) and content.strip(),response
assert '42' in content,(content,'simple arithmetic smoke failed')
result={'passed':True,'health':health,'unauthorized_status':401,'ui_status':404,'model':model,'elapsed_s':elapsed,'request':prompt,'response':response,'scope':'one arithmetic API smoke, not broad numerical qualification'}
label=os.environ.get('STRATA_TEST_LABEL','deployment')
Path('/src/logs/'+label+'-api-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'passed':True,'elapsed_s':elapsed,'content':content,'usage':response.get('usage'),'timings':response.get('timings')}))

