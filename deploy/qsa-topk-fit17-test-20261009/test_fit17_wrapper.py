#!/usr/bin/env python3
"""CPU-only parser/control-flow tests; never executes docker or accesses GPUs."""
import contextlib,io,json,re,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import run_middle_test as OLD
import run_fit17_test as W
HERE=Path(__file__).resolve().parent;LOGS=HERE/'test_logs'
oldmf=json.loads((LOGS/'reg33-transition-manifest.json').read_text())
newmf=json.loads((LOGS/'reg17-transition-manifest.json').read_text())

def trace(mode):
 return f'strata topk fit17 probe: variant={mode} max_blocks=17408 device_max_guard=1 hist2={int(mode==2)} arms='+('ref256/reg66' if mode==0 else 'ref256/reg17/reg66')+'\n'

def transformed(text):
 # Explicit synthetic adaptation of captured reg33 logs to new boundary values.
 for n in range(135164,135173):text=text.replace(str(n),str(n-65536))
 return text.replace('33792','17408').replace('33793','17409').replace('33794','17410').replace('middle33','middle17').replace(' mid=1 ',' fit17=1 ')

class Parsers(unittest.TestCase):
 def test_actual_old_real_and_new_trace_examples(self):
  out=(LOGS/'real-g0-A1.stdout').read_text();err=(LOGS/'real-g0-A1.stderr').read_text()
  original=OLD.parse_result(out,err,0,0,64,7)
  route=err.splitlines()[0]+'\n'
  for mode in [0,1,2]:
   self.assertEqual(W.parse_result(out,route+trace(mode),0,mode,64,7),original)
  for bad in [trace(1)+trace(1),trace(1).replace('hist2=0','hist2=1'),trace(1).replace('17408','17409')]:
   with self.assertRaises(RuntimeError):W.parse_result(out,route+bad,0,1,64,7)
  with self.assertRaises(RuntimeError):W.parse_result(out,route+trace(1),1,1,64,7)
  with self.assertRaises(RuntimeError):W.parse_result(out.replace('PARITY_PASS','PARITY_FAIL'),route+trace(1),0,1,64,7)
 def test_actual_old_transitions_and_synthetic_fit17(self):
  for t in [4,9]:
   old=next(f for f in oldmf['fixtures'] if f['nq']==t);new=next(f for f in newmf['fixtures'] if f['nq']==t)
   out=(LOGS/f'gpu0-mid1-t{t}.stdout').read_text();err=(LOGS/f'gpu0-mid1-t{t}.stderr').read_text()
   self.assertTrue(OLD.parse_transition(out,err,0,1,old))
   out=transformed(out);route=err.splitlines()[0]+'\n'
   for mode in [0,1,2]:
    adapted=out.replace('fit17=1 ',f'fit17={mode} ')
    self.assertTrue(W.parse_transition(adapted,route+(trace(mode) if t<=8 else ''),0,mode,new))
   with self.assertRaises(RuntimeError):W.parse_transition(out.replace('CASE_PASS','CASE_BAD',1),route+trace(1),0,1,new)
   with self.assertRaises(RuntimeError):W.parse_transition(out+out.splitlines()[0]+'\n',route+trace(1),0,1,new)
 def test_scoring_negative_does_not_raise(self):
  def rows(vals):return [dict(mode=m,median_ms_per_call=x) for m,x in zip([0,1,1,0],vals)]
  self.assertEqual(W.score_abba(rows([10,9,9,10]),0,1)['status'],'negative_candidate')
  self.assertTrue(W.score_abba(rows([10,6,6,10]),0,1)['first_gpu_gate_passed'])
  self.assertFalse(W.score_abba(rows([5,6,1,10]),0,1)['first_gpu_gate_passed'])
 def test_mocked_offline_order_and_negative_continuation(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);fixtures=root/'fixtures';fixtures.mkdir();(fixtures/'reg17-transition-manifest.json').write_text(json.dumps(newmf))
   a=types.SimpleNamespace(leave_api_stopped=True,build=root/'build',output=root/'output',transition_fixtures=fixtures,
      real0=root/'real0',real1=root/'real1',expected_binary_sha='a'*64,expected_transition_sha='b'*64)
   calls=[]
   def trans(a,ds,out,d,m,f):calls.append(('transition',d,m,f['nq']));return {'mode':m,'device':d,'nq':f['nq']}
   def real(a,ds,out,d,f,tag,m):calls.append(('real',d,m,tag));return dict(mode=m,median_ms_per_call={0:10,1:9,2:6}[m])
   power=[dict(pci='test',mode='high',power1_cap=190000000)]
   with patch.object(W.S,'R',root),patch.object(W,'gate',return_value={'allowed_modes':[1,2]}),patch.object(W.S,'call',return_value=''),patch.object(W.S,'devices',return_value=[]),patch.object(W.S,'gpu',return_value=power),patch.object(W.S,'cleanup'),patch.object(W,'run_transition',side_effect=trans),patch.object(W,'run_one',side_effect=real),contextlib.redirect_stdout(io.StringIO()):
    W.execute(a)
   self.assertEqual(len(calls),42);self.assertTrue(all(c[0]=='transition' for c in calls[:30]))
   self.assertEqual([(c[1],c[2]) for c in calls[30:]],[(0,0),(0,1),(0,1),(0,0),(0,0),(0,2),(0,2),(0,0),(1,0),(1,2),(1,2),(1,0)])
   summary=json.loads((a.output/'summary.json').read_text())
   self.assertTrue(summary['component_screen_completed']);self.assertEqual(summary['negative_candidate_modes'],[1]);self.assertEqual(summary['gpu1_eligible_modes'],[2]);self.assertTrue(summary['cleanup']['passed']);self.assertFalse(summary['api_start_stop_actions']);self.assertFalse(summary['promoted'])
if __name__=='__main__':unittest.main(verbosity=2)
