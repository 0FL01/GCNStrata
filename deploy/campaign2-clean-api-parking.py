from pathlib import Path
import json,os,time,urllib.request
O=Path('/src/logs/perf-campaign-20261007/exact-stack-api')
def call(messages):
 body={'model':'current','messages':messages,'max_tokens':16,'temperature':0,'reasoning_effort':'none'}
 req=urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['STRATA_API_KEY']})
 t=time.monotonic()
 with urllib.request.urlopen(req,timeout=300) as f:d=json.load(f)
 return d,time.monotonic()-t
a=[{'role':'user','content':'A synthetic cache test. Ignore the data values below and reply with only 42. Data:'+chr(10)+chr(10).join('record %d: value alpha beta gamma delta.'%i for i in range(500))+chr(10)+'Reply with only 42.'}]
rows=[]
for tag,m in [('cold',a),('warm',a),('auxiliary',[{'role':'user','content':'Reply with only TEST.'}]),('restored',a)]:
 d,elapsed=call(m);u=d.get('usage',{});content=d['choices'][0]['message'].get('content','')
 if tag!='auxiliary':assert '42' in content,(tag,content)
 cached=u.get('prompt_tokens_details',{}).get('cached_tokens',u.get('cached_tokens',0))
 rows.append({'tag':tag,'elapsed_s':elapsed,'content':content,'usage':u,'cached_tokens':cached})
assert rows[-1]['cached_tokens']>0,rows
out={'passed':True,'rows':rows,'scope':'synthetic A/title/A RAM parking regression, not a client replay'}
(O/'parking.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
