import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_abba as B
class Tests(unittest.TestCase):
 def test_bound_admission_and_mutations(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);base=B.load(B.HERE/'baseline.manifest.json');plan=B.load(B.HERE/'qualification/probe-plan.json');binding={'binary_sha256':'1'*64,'build_receipt_sha256':'2'*64}
   native=dict(passed=True,prompt_tokens=12288,prompt_read=12288,prompt_reused=0,actual_outputs=1,output_ids=B.load(B.HERE/'qualification/baseline-proof.json')['output_ids'])
   gate=dict(gate='PASS_PRIMARY_CORRECTNESS_AND_REAL_HIT_POPULATION',primary_candidates=324,primary_hits=22,primary_learned=11,cache_scope_devices=[0],output_ids=B.load(B.HERE/'qualification/baseline-proof.json')['output_ids'])
   proof=B.load(B.HERE/'qualification/baseline-proof.json');gate.update(keys=proof['keys'],original_routes=proof['routes'],primary_keys=108,primary_physical_guards_inferred=302,peer_keys=84,peer_attempts=252,performance_claim=False)
   q=dict(completed=True,cleanup=dict(passed=True,api_left_stopped=True),bindings=binding,planned_processes=1,requested_outputs=1,automatic_retries=0,native=native,qualification=gate)
   qp=d/'q.json';B.N.write(qp,q);a=SimpleNamespace(qualification_receipt=qp,qualification_receipt_sha=B.N.sha(qp),benchmark_review=d/'review.json')
   expected=dict(binding,qualification_receipt_sha256=a.qualification_receipt_sha,benchmark_runner_sha256=B.N.sha(B.HERE/'run_abba.py'),benchmark_input_pins_sha256=B.N.sha(B.HERE/'input-pins.json'),baseline_sha256=B.BASE,fixture_sha256=B.FIXTURE,baseline_flags=base['flags'],candidate_flags=dict(base['flags'],**{B.PRIMARY:'1'}),native_args=base['native_args'],policy=B.POLICY)
   review=dict(passed=True,source_and_cpu_review_passed=True,actual_build_and_qualification_passed=True,bindings=expected)
   def save(r):B.N.write(a.benchmark_review,r);a.benchmark_review_sha=B.N.sha(a.benchmark_review)
   with patch.object(B.Q,'inputs',return_value=(plan,[],binding)):
    save(review);self.assertEqual(B.inputs(a)[2],expected)
    for key,value in [('passed',False),('actual_build_and_qualification_passed',False),('bindings',{})]:
     bad=copy.deepcopy(review);bad[key]=value;save(bad)
     with self.assertRaises(RuntimeError):B.inputs(a)
    save(review);a.qualification_receipt_sha='0'*64
    with self.assertRaises(RuntimeError):B.inputs(a)
 def test_raw_tail_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);(d/'private').mkdir();p=d/'private/A1.stdout';r=dict(output_ids=[1]*1024)
   good='READY 204800\nRESUME 0\nREUSED 0\n'+'T 1\n'*1024+'DONE complete\n';p.write_text(good);B.verify_raw(d,'A1',r)
   for tail in ['T 2\n','DONE second\n','ERR failure\n','READY 204800\n']:
    p.write_text(good+tail)
    with self.assertRaises(RuntimeError):B.verify_raw(d,'A1',r)
if __name__=='__main__':unittest.main()
