from pathlib import Path
import sys,json,collections
R=Path('/home/radneon/strata');sys.path.insert(0,str(R/'tools'));import gguf_reader as G
out=R/'logs/quant-comparison-64k-20261006';out.mkdir(exist_ok=True)
paths=[Path('/home/radneon/models-nvme/qwen38-keep1-Q4_0.gguf'),Path('/home/radneon/models-nvme/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf')]
results=[]
for path in paths:
 g=G.GGUFFile(path);ts={t.name:t for t in g.tensors}
 item={'path':str(path),'bytes':path.stat().st_size,'tensor_count':len(ts),'types':dict(collections.Counter(t.type_name for t in g.tensors)),'experts':[],'ple_metadata':{k:v for k,v in g.metadata.items() if '.ple.' in k},'sample_names':list(ts)[:12]}
 for t in g.tensors:
  if '_exps.weight' in t.name:item['experts'].append({'name':t.name,'type':t.type_name,'shape':t.shape})
 if 'per_layer_token_embd.weight' in ts:
  t=ts['per_layer_token_embd.weight'];item['ple']={'type':t.type_name,'shape':t.shape,'bytes':t.expected_bytes()}
 results.append(item)
(out/'source-inventory.json').write_text(json.dumps(results,indent=2))
for item in results:
 print(json.dumps({k:v for k,v in item.items() if k!='experts'}));print('EXPERT_TYPES '+json.dumps(dict(collections.Counter(x['type'] for x in item['experts']))))
