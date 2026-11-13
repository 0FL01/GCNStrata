import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_abba as B
class Tests(unittest.TestCase):
 def results(self):
  h=B.load(B.HERE/'historical.result.json');out=[]
  for t in B.ORDER:
   r=copy.deepcopy(h);r['tag']=t
   if t.startswith('B'):r['prompt_ms']/=1.02
   out.append(r)
  return out
 def test_positive_screen(self):
  c=B.compare(self.results());self.assertTrue(c['screen_passed']);self.assertFalse(c['promoted']);self.assertFalse(c['work_matched_speed_claim']);self.assertEqual(len(c['PP']['paired_gain_pct']),2)
 def test_reverse_pair_required(self):
  r=self.results();r[2]['prompt_ms']=r[3]['prompt_ms']*1.01
  self.assertFalse(B.compare(r)['pp_screen_passed'])
 def test_tg_control(self):
  r=self.results();r[1]['decode_ms']*=1.001;self.assertFalse(B.compare(r)['tg_control_passed'])
 def test_drift_retained(self):
  r=self.results();r[1]['offered']+=1;c=B.compare(r);self.assertTrue(c['counter_drift']);self.assertEqual(c['paired_counter_deltas'][0]['candidate_minus_control']['offered'],1);self.assertFalse(c['work_matched_speed_claim'])
 def test_bad_results(self):
  for field,value in [('output_ids',[1]),('prompt_read',1),('actual_outputs',1023),('accepted',-1),('prompt_ms',float('nan'))]:
   with self.subTest(field=field):
    r=self.results();r[1][field]=value
    with self.assertRaises(RuntimeError):B.compare(r)
 def test_order(self):
  with self.assertRaises(RuntimeError):B.compare(list(reversed(self.results())))
 def test_qual_gate_mutations(self):
  n=dict(passed=True,prompt_tokens=12288,prompt_read=12288,prompt_reused=0,actual_outputs=1,output_ids=B.load(B.HERE/'qualification/baseline-proof.json')['output_ids']);g=dict(gate='PASS_PRIMARY_CORRECTNESS_AND_REAL_HIT_POPULATION',primary_candidates=324,primary_hits=22,primary_learned=11,cache_scope_devices=[0],output_ids=B.load(B.HERE/'qualification/baseline-proof.json')['output_ids'])
  proof=B.load(B.HERE/'qualification/baseline-proof.json');g.update(keys=proof['keys'],original_routes=proof['routes'],primary_keys=108,primary_physical_guards_inferred=302,peer_keys=84,peer_attempts=252,performance_claim=False)
  q=dict(completed=True,cleanup=dict(passed=True,api_left_stopped=True),bindings={'binary':'exact'},planned_processes=1,requested_outputs=1,automatic_retries=0,native=n,qualification=g);B.check_qualification(q,q['bindings'])
  for key,value in [('completed',False),('bindings',{}),('requested_outputs',2),('qualification',{}),('native',{})]:
   with self.subTest(key=key):
    bad=copy.deepcopy(q);bad[key]=value
    with self.assertRaises((RuntimeError,KeyError)):B.check_qualification(bad,q['bindings'])
  for key,value in [('output_ids',[42]),('keys',[]),('original_routes',[]),('primary_physical_guards_inferred',301),('peer_attempts',251)]:
   with self.subTest(payload=key):
    bad=copy.deepcopy(q);bad['qualification'][key]=value
    if key=='output_ids':bad['native']['output_ids']=value
    with self.assertRaises(RuntimeError):B.check_qualification(bad,q['bindings'])
 def test_mock_schedule_and_failure(self):
  for fail in [None,'B1']:
   with self.subTest(fail=fail),tempfile.TemporaryDirectory() as td:
    root=Path(td);(root/'logs/perf-campaign-20261009').mkdir(parents=True);binary=root/'candidate';baseline=root/'baseline';binary.write_bytes(b'c');baseline.write_bytes(b'a')
    a=SimpleNamespace(binary=binary,baseline=baseline,binary_sha='1'*64,output=root/'out',leave_api_stopped=True)
    base=B.load(B.HERE/'baseline.manifest.json');called=[]
    def arm(tag,r,s,ids,ds,out):
     called.append(tag);self.assertNotIn(B.QUALIFY,r['flags']);self.assertEqual(r['flags'].get(B.PRIMARY), '1' if tag.startswith('B') else None)
     if tag==fail:raise RuntimeError('injected arm failure')
     private=out/'private';private.mkdir(exist_ok=True);(private/f'{tag}.stderr').write_text('');(private/f'{tag}.stdout').write_text('READY 204800\nRESUME 0\nREUSED 0\n'+''.join('T '+str(i)+'\n' for i in self.results()[0]['output_ids'])+'DONE complete\n')
     B.N.write(private/f'{tag}.termination.json',dict(native_completed=True,native_exit=0,reader_finished=True,reader_error=None,native_cleanup_error=None))
     return next(r for r in self.results() if r['tag']==tag)
    def call(args,*rest):return '[]' if args[:3]==['docker','image','inspect'] else ''
    with patch.object(B.N,'R',root),patch.object(B,'inputs',return_value=(base,[1]*65536,{})),patch.object(B.N,'devices',return_value=[]),patch.object(B.N,'gpu',return_value=[]),patch.object(B.N,'call',side_effect=call),patch.object(B.N,'sha',side_effect=lambda p:B.BASE if p==baseline else '1'*64),patch.object(B.A,'run_arm',side_effect=arm),patch.object(B.S,'cleanup'):
     if fail:
      with self.assertRaises(RuntimeError):B.execute(a)
     else:B.execute(a)
     self.assertEqual(called,['A1','B1'] if fail else B.ORDER)
     summary=B.load(a.output/'summary.json');self.assertTrue(summary['cleanup']['passed']);self.assertEqual(summary['completed'],fail is None)
     a.output=root/'second'
     with self.assertRaises(FileExistsError):B.execute(a)
     self.assertEqual(called,['A1','B1'] if fail else B.ORDER)
if __name__=='__main__':unittest.main()
