"""Strict frozen-prefix correctness/hit parser. No speed interpretation."""
import json,re
from pathlib import Path
import parse_counts as base
need=base.need
HERE=Path(__file__).resolve().parent
SUMMARY=re.compile(r'strata hc-wneg: device=(\d+) candidates=(\d+) hits=(\d+) learned=(\d+) mismatches=(\d+)')
KEY=re.compile(r'strata hc-wneg-key: ordinal=(\d+) N=(\d+) K=(\d+) mask=(\d+) hits=(\d+)')
ROUTE=re.compile(r'strata hc-f16 route: attempts=(\d+) taken=(\d+)')
def parse(text,native,stdout_path):
 proof=json.loads((HERE/'baseline-proof.json').read_text(),object_pairs_hook=base.unique)
 summaries=[];keys=[];routes=[]
 for line in text.splitlines():
  if 'hc-wneg' in line:
   m=SUMMARY.fullmatch(line)
   if m:summaries.append(list(map(int,m.groups())));continue
   m=KEY.fullmatch(line);need(m is not None,'malformed qualification row')
   vals=list(map(int,m.groups()));need(all(0<=x<2**63 for x in vals),'qualification integer range')
   keys.append(dict(zip(['ordinal','N','K','mask','hits'],vals)))
  if 'hc-f16 route:' in line:
   m=ROUTE.fullmatch(line);need(m is not None,'malformed original route row');routes.append(list(map(int,m.groups())))
  need('hc_observer_' not in line,'unexpected observer instrumentation')
 need(summaries==[[0,324,22,11,0]],'exact primary-only scope and candidates/hits/learned/mismatches required')
 need(keys==proof['keys'],'exact unique ordered ordinal/N/K/mask/hit population')
 need(routes==proof['routes'],'original trace attempts/taken changed')
 need(native['output_ids']==proof['output_ids'],'baseline output mismatch')
 need(native['passed'] is True and native['prompt_tokens']==native['prompt_read']==12288 and native['prompt_reused']==0 and native['actual_outputs']==1,'cold full-prefix native result')
 lines=Path(stdout_path).read_text().splitlines()
 need(sum(x.startswith('READY ') for x in lines)==sum(x.startswith('DONE ') for x in lines)==sum(x.startswith('T ') for x in lines)==1,'exact native output coverage')
 need(lines.count('RESUME 0')==lines.count('REUSED 0')==1,'cold native protocol')
 need([int(x.split()[1]) for x in lines if x.startswith('T ')]==proof['output_ids'],'raw output mismatch')
 need([(int(x.split()[1]),int(x.split()[2])) for x in lines if x.startswith('PP ')]==[(4096,12288),(8192,12288),(12287,12288)],'frozen prefill chunk coverage')
 return dict(gate='PASS_PRIMARY_CORRECTNESS_AND_REAL_HIT_POPULATION',output_ids=proof['output_ids'],primary_keys=108,primary_candidates=324,primary_learned=11,primary_hits=22,primary_physical_guards_inferred=302,physical_guard_note='Inferred candidates minus hits from reviewed sole guard path; not a separately instrumented counter.',peer_keys=84,peer_attempts=252,cache_scope_devices=[0],keys=keys,original_routes=routes,performance_claim=False)
