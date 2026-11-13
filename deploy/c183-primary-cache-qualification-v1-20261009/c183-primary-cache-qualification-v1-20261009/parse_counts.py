#!/usr/bin/env python3
"""Read-only strict observer reconciliation. No inference from attempts-minus-taken."""
import json
import re
SCHEMA={
 'hc_observer_lifecycle':set('type context generation bound thread_changes binding_changes ticket_mismatches inflight'.split()),
 'hc_observer_summary':set('type context generation device rebind_epoch valid attempts eligible guards original_attempts original_taken unmapped ambiguous overflows registry_overflows device_query_errors layer_begin layer_end chunk_T chunk_p0 stages'.split()),
 'hc_observer_weight':set('type context generation tensor_ordinal N K declarations address_epoch pointer_changes within_generation_W_mask_changes first_observations repeats_same_visible_generation W_bad_repeats_same_visible_generation repeats_changed_visible_generation attempts_by_T eligible_by_T stages_4095 stages_4096 mask_hist_4095 mask_hist_4096'.split()),
 'hc_observer_end':set('type context generation weight_rows'.split()),
 'hc_observer_registry_overflow':{'type'}}
STAGES=['no_scratch','small_scratch','ldy_small','ldy_large','Y_size_overflow','scratch_address_overflow','X_overlap','W_overlap','Y_overlap','range_reject','exact_reject','arena_only','taken']
def need(ok,label):
 if not ok:raise ValueError(label)
def unique(items):
 d={}
 for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
 return d
def counts(v,n):need(isinstance(v,list) and len(v)==n and all(type(x) is int and 0<=x<2**64 for x in v),'count array');return v
def parse(text,require_prefix=True):
 rows=[]
 for line in text.splitlines():
  if re.match(r'^\s*\{\s*"type"\s*:\s*"hc_observer_',line):rows.append(json.loads(line,object_pairs_hook=unique))
 need(rows and len(rows)<=8*(256+3)+1,'bounded observer rows');need(not any(r['type']=='hc_observer_registry_overflow' for r in rows),'registry overflow')
 for row in rows:
  need(row.get('type') in SCHEMA and set(row)==SCHEMA[row['type']],'observer row type/schema')
  for name,value in row.items():
   if name=='type':continue
   if name in ('valid','bound'):need(type(value)is bool,'exact boolean field')
   elif name in ('stages','chunk_T','chunk_p0','attempts_by_T','eligible_by_T','stages_4095','stages_4096','mask_hist_4095','mask_hist_4096'):need(isinstance(value,list) and all(type(x)is int and 0<=x<2**64 for x in value),'exact nonnegative list counters')
   else:need(type(value)is int and (-1 if name in ('device','layer_begin','layer_end') else 0)<=value<2**64,'exact integer scalar counter')
  need(1<=row['context']<=8 and 1<=row['generation']<=8,'context/generation bound')
 summaries=[r for r in rows if r['type']=='hc_observer_summary'];need(0<len(summaries)<=8,'context bound');need(len({(r['context'],r['generation']) for r in summaries})==len(summaries),'duplicate context')
 summary_keys={(r['context'],r['generation']) for r in summaries};need(all((r['context'],r['generation']) in summary_keys for r in rows),'orphan observer row')
 result=[];active_devices=[];all_mask_changes=0;all_Wbad_repeats=0;all_Wbad=0;all_full=True;all_guard_covered=True
 for s in summaries:
  key=(s['context'],s['generation']);group=[r for r in rows if (r.get('context'),r.get('generation'))==key];weights=[r for r in group if r['type']=='hc_observer_weight'];ends=[r for r in group if r['type']=='hc_observer_end'];life=[r for r in group if r['type']=='hc_observer_lifecycle']
  need(len(ends)==len(life)==1 and ends[0]['weight_rows']==len(weights)<=256,'terminal row coverage');need(s['valid'] is True,'native invalid context');need(all(s[k]==0 for k in ['unmapped','ambiguous','overflows','registry_overflows','device_query_errors']),'native observer errors')
  need(all(life[0][k]==0 for k in ['thread_changes','binding_changes','ticket_mismatches','inflight']),'lifecycle invalid')
  st=counts(s['stages'],13);need(sum(st)==s['attempts']==s['original_attempts'],'context attempts reconciliation');need(sum(st[:9])+s['eligible']==s['attempts'] and sum(st[9:])==s['eligible']==s['guards'],'context eligibility reconciliation');need(st[11]==0 and st[12]==s['original_taken'],'terminal taken/arena reconciliation')
  aggregate=[0]*13;attempts=eligible=0;classes=[0,0,0,0];ordinals=[];mask_changes=repeats=0;full=True;guard_covered=0
  for w in weights:
   ordinal=w['tensor_ordinal'];need(type(ordinal) is int and ordinal>0 and ordinal not in ordinals,'unique tensor ordinal');ordinals.append(ordinal)
   need((w['N'],w['K'])==((320,10240) if ordinal%2 else (10240,320)),'ordinal shape binding')
   wa=counts(w['attempts_by_T'],2);we=counts(w['eligible_by_T'],2);attempts+=sum(wa);eligible+=sum(we);obs=0
   for ti,t in enumerate((4095,4096)):
    stage=counts(w[f'stages_{t}'],13);hist=counts(w[f'mask_hist_{t}'],64);obs+=sum(hist)
    need(sum(stage)==wa[ti] and sum(stage[:9])+we[ti]==wa[ti] and sum(stage[9:])==we[ti]==sum(hist),'weight stage reconciliation')
    ranges=sum(v for i,v in enumerate(hist) if ((i//8)|(i%8))&3)
    exact=sum(v for i,v in enumerate(hist) if not (((i//8)|(i%8))&3) and (((i//8)|(i%8))&4))
    need(stage[9]==ranges and stage[10]==exact and stage[11]==0 and stage[12]==hist[0],'mask-to-terminal reconciliation')
    for i,v in enumerate(hist):classes[(2 if i//8 else 0)+(1 if i%8 else 0)]+=v
    aggregate=[a+b for a,b in zip(aggregate,stage)]
   need(w['first_observations']+w['repeats_same_visible_generation']+w['repeats_changed_visible_generation']==obs,'observation lifecycle reconciliation')
   guard_covered+=int(obs>0)
   need(0<=w['W_bad_repeats_same_visible_generation']<=w['repeats_same_visible_generation'],'known-bad repeats')
   mask_changes+=w['within_generation_W_mask_changes'];repeats+=w['W_bad_repeats_same_visible_generation']
   full &= wa==[1,2] and w['declarations']==3
  need(attempts==s['attempts'] and eligible==s['eligible'] and aggregate==st,'weight sums match context')
  if s['attempts'] or weights:
   active_devices.append(s['device']);need(life[0]['bound'] is True,'caller binding absent')
   if require_prefix:
    need(s['chunk_T']==[4096,4096,4095] and s['chunk_p0']==[0,4096,8192],'runtime chunk geometry')
    lb,le=(0,27) if s['device']==0 else (27,48) if s['device']==1 else (-1,-1)
    need((s['layer_begin'],s['layer_end'])==(lb,le),'stage layer coverage')
    need(set(ordinals)==set(range(4*lb+1,4*le+1)),'logical tensor declaration coverage')
  result.append(dict(context=s['context'],generation=s['generation'],device=s['device'],attempts=s['attempts'],eligible=s['eligible'],stage_counts=dict(zip(STAGES,st)),guard_classes=dict(zip(['both_good','X_only_bad','W_only_bad','both_bad'],classes)),logical_tensor_rows=len(weights),weight_rows_with_guard_observation=guard_covered,weight_rows_never_eligible=len(weights)-guard_covered,all_declared_weights_attempted_once_tail_twice_full=full,within_generation_W_mask_changes=mask_changes,known_W_bad_repeats_same_visible_generation=repeats,chunk_T=s['chunk_T'],chunk_p0=s['chunk_p0']))
  all_mask_changes+=mask_changes;all_Wbad_repeats+=repeats;all_Wbad+=classes[2]+classes[3];all_full &=full
  all_guard_covered &=guard_covered==len(weights)
 if require_prefix:need(sorted(active_devices)==[0,1],'exact two-stage active coverage')
 if all_mask_changes:gate='PARK_KEY_LIFECYCLE_UNSTABLE'
 elif all_Wbad_repeats==0:gate='PARK_NO_OBSERVED_REUSABLE_W_BAD'
 else:gate='COUNTS_ONLY_OPPORTUNITY_REQUIRES_COST_AND_LIFETIME_PROOF'
 return dict(passed=True,gate=gate,contexts=result,W_bad_observations=all_Wbad,known_W_bad_repeats_same_visible_generation=all_Wbad_repeats,full_attempt_coverage=all_full,full_guard_identity_coverage=all_guard_covered,coverage_limit='If any logical tensor bypassed HC or lacked an eligible guard observation, this is observed-prefix coverage only; no claim every tensor is W-good',lifecycle_limit='Visible init/rebind/pointer/model/session/stream/thread checks are not a proof of content immutability under in-place mutation',performance_claim=False)
if __name__=='__main__':
 import sys
 from pathlib import Path
 print(json.dumps(parse(Path(sys.argv[1]).read_text()),indent=2))
