from pathlib import Path
import json,numpy as np
R=Path('/src');P=R/'runtime/shefowl-hybrid';O=R/'logs/perf-campaign-20261007/hc-f16-range'
O.mkdir(exist_ok=False)
raw=(np.arange(65536,dtype=np.uint32)<<16);x=raw.view(np.float32)
with np.errstate(over='ignore',invalid='ignore',under='ignore'):
 h=x.astype(np.float16).astype(np.float32)
finite=np.isfinite(x);changed=(h.view(np.uint32)!=raw)&finite
overflow=finite&~np.isfinite(h);zero=(x!=0)&finite&(h==0)
data=np.memmap(P/'dense.bin',mode='r',dtype=np.uint16)
rows=[];total=np.zeros(65536,dtype=np.uint64)
for line in (P/'index.txt').read_text().splitlines():
 if line.startswith('#'):continue
 f=line.split()
 if len(f)<9 or 'hc' not in f[0] or not (f[0].endswith('down.weight') or f[0].endswith('up.weight')):continue
 assert int(f[2])==4
 off=int(f[3]);size=int(f[4]);assert size==6553600 and off%2==0
 v=data[off//2:(off+size)//2];assert v.size==size//2
 hist=np.bincount(v,minlength=65536).astype(np.uint64);total+=hist
 used=(hist>0)&finite
 error=(h[used].astype(np.float64)-x[used].astype(np.float64))
 denom=np.sum(hist[used]*x[used].astype(np.float64)**2)
 row={'tensor':f[0],'elements':int(hist.sum()),'changed_on_fp16_conversion':int(hist[changed].sum()),'overflow':int(hist[overflow].sum()),'underflow_to_zero':int(hist[zero].sum()),'nonfinite_source':int(hist[~finite].sum()),'min':float(x[used].min()),'max':float(x[used].max()),'relative_l2_conversion_error':float(np.sqrt(np.sum(hist[used]*error**2)/denom))}
 rows.append(row)
summary={'tensors':len(rows),'elements':int(total.sum()),'changed':int(total[changed].sum()),'overflow':int(total[overflow].sum()),'underflow_to_zero':int(total[zero].sum()),'nonfinite_source':int(total[~finite].sum()),'scope':'all BF16 HC up/down weights in native dense pack; does not inspect activations or establish model-quality equivalence'}
(O/'weights.json').write_text(json.dumps({'summary':summary,'tensors':rows},indent=2))
print(json.dumps(summary),flush=True)
