from pathlib import Path
import numpy as np,json,hashlib
O=Path('/src/logs/perf-campaign-20261007/hc-f16-distribution');rows=[]
def metrics(a,b):
 a=a.astype(np.float64);b=b.astype(np.float64);d=b-a
 la=a-(np.max(a)+np.log(np.exp(a-np.max(a)).sum()))
 lb=b-(np.max(b)+np.log(np.exp(b-np.max(b)).sum()))
 pa=np.exp(la);pb=np.exp(lb);topa=np.argsort(a)[-10:][::-1];topb=np.argsort(b)[-10:][::-1]
 return {'bitwise_equal':bool(np.array_equal(a,b)),'changed':int(np.count_nonzero(a!=b)),'max_abs':float(np.abs(d).max()),'relative_l2':float(np.linalg.norm(d)/np.linalg.norm(a)),'argmax_a':int(topa[0]),'argmax_b':int(topb[0]),'top2_margin_a':float(a[topa[0]]-a[topa[1]]),'top10_overlap':len(set(topa)&set(topb)),'KL_A_to_B':float(np.sum(pa*(la-lb))),'KL_B_to_A':float(np.sum(pb*(lb-la))),'total_variation':float(np.abs(pa-pb).sum()/2),'argmax_probability_a':float(pa[topa[0]]),'same_token_probability_b':float(pb[topa[0]])}
dt=np.dtype([('position','<i8'),('r','<f4',(10240,))])
for kind in ['code','ru']:
 logits={};res={}
 for arm in ['A1','A2','B1']:
  p=O/'runs'/f'{kind}4096-{arm}.logits.f32';assert p.stat().st_size==248320*4
  logits[arm]=np.fromfile(p,dtype='<f4');assert np.isfinite(logits[arm]).all()
  p=O/'runs'/f'{kind}4096-{arm}.residuals.f32';assert p.stat().st_size==64*dt.itemsize
  res[arm]=np.fromfile(p,dtype=dt);assert np.isfinite(res[arm]['r']).all()
  assert np.array_equal(res[arm]['position'],np.arange(0,4095,64))
 for arm in ['A2','B1']:
  ra=res['A1']['r'].astype(np.float64);rb=res[arm]['r'].astype(np.float64);rd=rb-ra
  row={'fixture':kind,'comparison':'A1/'+arm,'logits':metrics(logits['A1'],logits[arm]),'residuals':{'bitwise_equal':bool(np.array_equal(ra,rb)),'changed':int(np.count_nonzero(ra!=rb)),'max_abs':float(np.abs(rd).max()),'relative_l2':float(np.linalg.norm(rd)/np.linalg.norm(ra))}}
  rows.append(row)
result={'comparisons':rows,'vocab':248320,'residual_rows':64,'residual_width':10240,'scope':'Two4K fixtures, first-next-token distribution and sampled final residuals; not broad quality equivalence or perplexity.'}
(O/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
