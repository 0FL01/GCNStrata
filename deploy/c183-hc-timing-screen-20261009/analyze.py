#!/usr/bin/env python3
"""CPU recheck and predeclared synthetic-screen estimator; never a model bound."""
import array,hashlib,json,math,mmap,statistics,struct
from pathlib import Path
DESCS=((3276800,4096,0,1728),(41943040,512,5,810),(41932800,512,5,54),(1310720,512,5,810),(1310400,512,5,54),(16777216,512,0,0))
TARGET=1003.739604
def need(ok,label):
 if not ok:raise ValueError(label)
def unique(items):
 d={}
 for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
 return d
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def pattern_bytes(pattern,n):
 accepted=(0x3e80,0xbe80,0x3f00,0xbf00,0x3f80,0xbf80,0x4000,0xc000,0x4080,0xc080,0x4100,0xc100);reject=(1,0x7f7f,0x7f81,0x8000);period=(12,49152,4)[pattern]
 block=array.array('H',(reject[i%4] if pattern==2 else (reject[(i//4096)%4] if pattern==1 and i%4096==0 else accepted[i%12]) for i in range(period)))
 import sys
 if sys.byteorder!='little':block.byteswap()
 raw=block.tobytes();return b'\x5a\xa5'+raw*(n//period)+raw[:(n%period)*2]+b'\xa5\x5a'
def stats_expected(pattern,offset):return tuple(0xa55aa55a if j<1+offset or j>=17+offset else int(pattern!=0 and 2+offset<=j<=4+offset) for j in range(23))
def check_pair(paths,descriptor,pattern):
 n,scale,offset,_=DESCS[descriptor];words=n+2;size=28+words*4+92;expected=pattern_bytes(pattern,n)
 with Path(paths[0]).open('rb') as fa,Path(paths[1]).open('rb') as fb,mmap.mmap(fa.fileno(),0,access=mmap.ACCESS_READ) as a,mmap.mmap(fb.fileno(),0,access=mmap.ACCESS_READ) as b:
  for arm,m in enumerate((a,b)):
   need(len(m)==size,'preflight size');need(struct.unpack_from('<7I',m)==(0x48535431,descriptor,pattern,arm,n,scale,offset),'preflight header')
   need(m[28:28+words*2]==expected,'input changed or wrong generator')
   out_at=28+words*2;need(m[out_at:out_at+2]==b'\x5a\xa5' and m[out_at+words*2-2:out_at+words*2]==b'\xa5\x5a','output guards')
   need(struct.unpack_from('<23I',m,28+words*4)==stats_expected(pattern,offset),'stats flags/guards')
  for at in range(28+words*2,28+words*4,4*1024**2):need(a[at:min(at+4*1024**2,28+words*4)]==b[at:min(at+4*1024**2,28+words*4)],'full output parity')
 return {Path(p).name:sha(p) for p in paths}
def analyze_samples(samples):
 need(len(samples)==360,'sample coverage');cells=[];index=0
 for pattern in range(3):
  for regime in range(2):
   for descriptor in range(5):
    quartets=[];raw=[]
    for q in range(3):
     values=[]
     for pos,arm in enumerate((0,1,1,0)):
      row=samples[index];expected=dict(type='sample',index=index,descriptor=descriptor,pattern=pattern,regime=regime,quartet=q,position=pos,arm=arm,launches=34+2*index)
      need({k:v for k,v in row.items() if k!='ms'}==expected,'sample identity/order/count')
      ms=row.get('ms');need(type(ms) in (int,float) and math.isfinite(ms) and ms>0,'positive finite duration');values.append(ms);raw.append(row);index+=1
     baseline=(values[0]+values[3])/2;candidate=(values[1]+values[2])/2
     quartets.append(dict(baseline_ms=baseline,candidate_ms=candidate,signed_delta_ms=baseline-candidate,baseline_over_candidate=baseline/candidate))
    delta=statistics.median(q['signed_delta_ms'] for q in quartets)
    cells.append(dict(descriptor=descriptor,pattern=pattern,regime=regime,calls=DESCS[descriptor][3],raw_samples=raw,quartets=quartets,median_paired_delta_ms=delta,median_baseline_ms=statistics.median(q['baseline_ms'] for q in quartets),median_candidate_ms=statistics.median(q['candidate_ms'] for q in quartets),baseline_range_ms=[min(x['ms'] for x in raw if x['arm']==0),max(x['ms'] for x in raw if x['arm']==0)],candidate_range_ms=[min(x['ms'] for x in raw if x['arm']==1),max(x['ms'] for x in raw if x['arm']==1)]))
 scenarios=[]
 for p in range(3):
  for r in range(2):
   chosen=[c for c in cells if c['pattern']==p and c['regime']==r]
   signed=sum(c['calls']*c['median_paired_delta_ms'] for c in chosen);optimistic=sum(c['calls']*max(0,c['median_paired_delta_ms']) for c in chosen)
   scenarios.append(dict(pattern=p,regime=r,signed_weighted_delta_ms=signed,optimistic_clipped_weighted_delta_ms=optimistic))
 maximum=max(s['optimistic_clipped_weighted_delta_ms'] for s in scenarios)
 return dict(cells=cells,scenarios=scenarios,target_ms=TARGET,largest_scenario_optimistic_ms=maximum,gate='INVESTIGATION_ONLY' if maximum>TARGET else 'PARK_WITHOUT_MODEL_QUALIFICATION',estimator='Median of three within-ABBA paired mean differences per descriptor; maximum callcounts; negatives clipped only for optimistic sums; no cross-scenario mixing',scope='Synthetic single-kernel screening estimate, not whole-model gain or bound',cache_caveat='Warm or scrub-conditioned; scrub-conditioned is not guaranteed cold',timing_caveat='Single-launch HIP event brackets can include host submission gaps')
def check_event_results(samples,events):
 need(len(events)==len(samples)==360,'raw event coverage')
 for sample,event in zip(samples,events):
  bits=struct.unpack('<I',struct.pack('<f',sample['ms']))[0]
  need(type(event.get('ms_bits')) is int and type(event.get('rc')) is int and event==dict(type='event_result',index=sample['index'],ms_bits=bits,rc=0),'raw event bits/sample agreement')
def check(out):
 out=Path(out);rows=[json.loads(line,object_pairs_hook=unique) for line in (out/'raw.jsonl').read_text().splitlines()]
 need(rows[-1]==dict(type='complete',launches=752,samples=360,preflight_pairs=16,cells=30,passed=True),'completion')
 hip=[x for x in rows if x.get('type')=='hip_rc'];need(all(type(x.get('rc')) is int and x['rc']==0 and type(x.get('operation')) is str for x in hip),'HIP rc')
 need(sum(x['operation']=='timed_kernel_launch' for x in hip)==360,'timed launch count');need(sum(x['operation'].startswith('submit(') for x in hip)==392,'untimed launch count')
 need(not any(x.get('type')=='failure' for x in rows),'native failure')
 expected=[(d,p) for d in range(5) for p in range(3)]+[(5,0)]
 pre=[x for x in rows if x.get('type')=='preflight_complete'];need(pre==[dict(type='preflight_complete',descriptor=d,pattern=p,equal=True) for d,p in expected],'preflight coverage')
 marker=[i for i,x in enumerate(rows) if x.get('type')=='all_preflights_complete'];need(len(marker)==1 and rows[marker[0]]==dict(type='all_preflights_complete',launches=32),'preflight barrier')
 need(all(i<marker[0] for i,x in enumerate(rows) if x.get('type')=='preflight_complete') and all(i>marker[0] for i,x in enumerate(rows) if x.get('type')=='sample'),'preflight before timing')
 cells=[x for x in rows if x.get('type')=='cell_complete'];need(cells==[dict(type='cell_complete',descriptor=d,pattern=p,regime=r,equal=True) for p in range(3) for r in range(2) for d in range(5)],'post-cell full parity coverage')
 sample_rows=[x for x in rows if x.get('type')=='sample'];result=analyze_samples(sample_rows);hashes={}
 check_event_results(sample_rows,[x for x in rows if x.get('type')=='event_result'])
 stat_rows=[x for x in rows if x.get('type')=='sample_stats'];need(len(stat_rows)==360,'per-sample stats coverage')
 for sample,stats in zip(sample_rows,stat_rows):need(stats==dict(type='sample_stats',index=sample['index'],arm=sample['arm'],words=list(stats_expected(sample['pattern'],DESCS[sample['descriptor']][2]))),'per-sample stats identity/values')
 need(len(list(out.glob('preflight-*.bin')))==32,'preflight file coverage')
 for d,p in expected:hashes.update(check_pair([out/f'preflight-d{d}-p{p}-a{a}.bin' for a in range(2)],d,p))
 result.update(passed=True,launches=752,samples=360,preflight_pairs=16,raw_hashes=hashes,raw_jsonl_sha256=sha(out/'raw.jsonl'));return result
if __name__=='__main__':
 import sys
 print(json.dumps(check(sys.argv[1]),indent=2))
