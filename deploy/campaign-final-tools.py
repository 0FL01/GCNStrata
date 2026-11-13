import os,json,urllib.request,time
from pathlib import Path
O=Path('/src/logs/perf-campaign-20261006/final-api')
def call(payload):
 r=urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+os.environ['STRATA_API_KEY'],'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=240) as f:return json.load(f)
model='Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0'
tool={'type':'function','function':{'name':'add_numbers','description':'Add two integer numbers.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}
messages=[{'role':'user','content':'Use add_numbers to calculate 17 plus 25. Call the tool once with a=17 and b=25.'}]
p={'model':model,'messages':messages,'tools':[tool],'tool_choice':{'type':'function','function':{'name':'add_numbers'}},'temperature':0,'max_tokens':1024,'reasoning_budget_tokens':128}
a=call(p);c=a['choices'][0];assert c['finish_reason']=='tool_calls',a
tc=c['message']['tool_calls'];assert len(tc)==1 and tc[0]['function']['name']=='add_numbers';assert json.loads(tc[0]['function']['arguments'])=={'a':17,'b':25}
messages+=[c['message'],{'role':'tool','tool_call_id':tc[0]['id'],'content':'42'},{'role':'user','content':'State only the resulting decimal integer.'}]
b=call({'model':model,'messages':messages,'tools':[tool],'tool_choice':'none','temperature':0,'max_tokens':256,'reasoning_budget_tokens':64})
assert '42' in b['choices'][0]['message'].get('content',''),b
result={'passed':True,'forced_named_call':a,'tool_result_followup':b,'scope':'synthetic tool roundtrip; original user incident not reproduced'}
(O/'tool-test.json').write_text(json.dumps(result,indent=2));print('tool_roundtrip_passed',flush=True)
