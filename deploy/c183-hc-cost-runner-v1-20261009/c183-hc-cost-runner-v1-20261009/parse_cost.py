#!/usr/bin/env python3
"""Strict read-only guard-wall diagnostic; conditional screen, never a gain estimate."""
import copy,json,re
import parse_counts as base
need=base.need
THRESHOLD_NS=1003739604
BASELINE_W_BAD_ORDINALS=[1, 7, 15, 29, 32, 39, 46, 67, 73, 84, 100, 113, 129, 142, 168, 172, 173, 177, 178, 186]
FIELDS=set('ns T p0 W_mask X_mask known_W_bad_repeat terminal'.split())

def parse(text,require_prefix=True):
 rows=[]
 for line in text.splitlines():
  if 'hc_observer_' in line:need(re.match(r'^\s*\{\s*"type"\s*:\s*"hc_observer_',line) is not None,'malformed/reordered observer diagnostic')
  if re.match(r'^\s*\{\s*"type"\s*:\s*"hc_observer_',line):
   rows.append(json.loads(line,object_pairs_hook=base.unique))
 need(rows and len(rows)<=2073,'bounded observer rows')
 stripped=copy.deepcopy(rows)
 for row in stripped:
  if row.get('type')=='hc_observer_weight':
   need('guard_observations' in row,'missing raw durations');del row['guard_observations']
 result=base.parse('\n'.join(json.dumps(r,separators=(',',':')) for r in stripped),require_prefix)
 summaries={(r['context'],r['generation']):r for r in rows if r['type']=='hc_observer_summary'}
 stages={};total_observations=0
 for row in rows:
  if row['type']!='hc_observer_weight':continue
  s=summaries[row['context'],row['generation']];device=s['device'];obs=row['guard_observations']
  need(type(obs)is list and len(obs)<=3,'bounded raw durations')
  need(len(obs)==sum(row['eligible_by_T']),'missing/extra raw durations')
  need(all(row[k]==0 for k in ['pointer_changes','address_epoch','within_generation_W_mask_changes','repeats_changed_visible_generation']) and s['rebind_epoch']==0,'cost screen visible lifetime changed')
  hist={4095:[0]*64,4096:[0]*64};term={4095:[0]*13,4096:[0]*13};known=0;last=None
  stage=stages.setdefault(device,dict(device=device,observations=0,all_guard_wall_ns=0,W_bad_weights=0,first_hit_ns=0,full_repeat_ns=0,tail_repeat_ns=0,first_hit_count=0,full_repeat_count=0,tail_repeat_count=0))
  prior=-1
  for o in obs:
   need(type(o)is dict and set(o)==FIELDS,'raw duration schema')
   need(all(type(o[k])is int for k in FIELDS-{'known_W_bad_repeat'}),'raw integer types')
   need(type(o['known_W_bad_repeat'])is bool,'raw known repeat boolean')
   need(0<=o['ns']<2**63,'invalid/overflow raw duration')
   need((o['T'],o['p0']) in list(zip(s['chunk_T'],s['chunk_p0'])) and o['p0']>prior,'raw geometry/order')
   prior=o['p0'];w,x=o['W_mask'],o['X_mask'];need(0<=w<=7 and 0<=x<=7,'raw masks')
   expected=9 if (w|x)&3 else 10 if (w|x)&4 else 12
   need(o['terminal']==expected,'raw terminal decision')
   repeat=last is not None and w!=0 and last==w
   need(o['known_W_bad_repeat']==repeat,'raw known repeat reconciliation')
   known+=repeat;last=w;hist[o['T']][w*8+x]+=1;term[o['T']][expected]+=1
   stage['observations']+=1;total_observations+=1;stage['all_guard_wall_ns']+=o['ns']
   if w:
    if o['p0']==0:label='first_hit';stage['W_bad_weights']+=1
    elif repeat and o['p0']==4096:label='full_repeat'
    elif repeat and o['p0']==8192:label='tail_repeat'
    else:raise ValueError('nonstationary W bad population')
    stage[label+'_ns']+=o['ns'];stage[label+'_count']+=1
  need(known==row['W_bad_repeats_same_visible_generation'],'raw known repeat count')
  need(row['first_observations']==int(bool(obs)) and row['repeats_same_visible_generation']==max(0,len(obs)-1),'raw lifetime count')
  for t in (4095,4096):
   need(hist[t]==row[f'mask_hist_{t}'],'raw mask histogram')
   need(term[t][9:]==row[f'stages_{t}'][9:],'raw terminal histogram')
  if require_prefix:
   need(len(obs)==3 and [(o['T'],o['p0']) for o in obs]==[(4096,0),(4096,4096),(4095,8192)],'full timed coverage required')
   need(all(o['W_mask']==(4 if row['tensor_ordinal'] in BASELINE_W_BAD_ORDINALS else 0) for o in obs),'changed baseline W bad identity/mask population')
 if require_prefix:
  need(total_observations==576 and result['full_attempt_coverage'] and result['full_guard_identity_coverage'],'exact 576 observations')
  need([stages[d]['W_bad_weights'] for d in (0,1)]==[11,9],'baseline W bad stage population')
 for d,s in stages.items():
  need(s['first_hit_count']==s['full_repeat_count']==s['tail_repeat_count'],'stationary repeat population')
  s['conditional_64K_repeat_count']=15*s['W_bad_weights']
  s['conditional_64K_screen_ns']=14*s['full_repeat_ns']+s['tail_repeat_ns']
  s['conditional_64K_screen_ms']=s['conditional_64K_screen_ns']/1e6
 total=sum(s['conditional_64K_screen_ns'] for s in stages.values())
 result.update(gate='PARK_WITHOUT_CACHE_PATCH_OR_MODEL_QUALIFICATION' if total<THRESHOLD_NS else 'ATTRIBUTION_AND_LIFETIME_INVESTIGATION_ONLY',guard_wall_stages=[stages[d] for d in sorted(stages)],raw_observation_count=total_observations,conditional_64K_screen_sum_ns=total,conditional_64K_screen_sum_ms=total/1e6,screen_threshold_ns=THRESHOLD_NS,screen_assumptions='Same W-bad population, eligibility, visible and content lifetime, and per-observation costs. Per stage: 14 times measured full-repeat sum plus measured tail-repeat sum; 165 primary and 135 peer repeats.',screen_limit='Whole guard wall includes queued predecessor work and is not wholly removable. Stage sum is a deliberately generous overlap-double-counting screen; never add it to GPU timings. This is not a model bound or gain estimate. First-hit costs are reported separately and excluded from repeat screen.',performance_claim=False)
 return result
if __name__=='__main__':
 import sys
 from pathlib import Path
 print(json.dumps(parse(Path(sys.argv[1]).read_text()),indent=2))
