import hashlib,json,tempfile,unittest,struct,subprocess,signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_screen as R
import analyze as A
def samples():
 result=[]
 for p in range(3):
  for r in range(2):
   for d in range(5):
    for q in range(3):
     for pos,arm in enumerate((0,1,1,0)):
      i=len(result);result.append(dict(type='sample',index=i,descriptor=d,pattern=p,regime=r,quartet=q,position=pos,arm=arm,launches=34+2*i,ms=2 if arm==0 else (1 if d%2==0 else 3)))
 return result
class Tests(unittest.TestCase):
 def test_compiled_core_and_real_parser(self):
  with tempfile.TemporaryDirectory() as t:
   binary=Path(t)/'test';subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-O2',str(R.HERE/'test_core.cpp'),'-o',str(binary)],check=True)
   self.assertEqual(subprocess.run([str(binary),'--watchdog'],timeout=3).returncode,-signal.SIGALRM)
   row=json.loads(subprocess.check_output([str(binary),t],text=True));self.assertEqual(row,dict(type='hip_rc',operation='quote" slash\\ tab\t newline\n \x01',rc=7))
   for p in range(3):
    paths=[Path(t)/f'preflight-d4-p{p}-a{a}.bin' for a in range(2)];self.assertEqual(len(A.check_pair(paths,4,p)),2)
   paths=[Path(t)/f'preflight-d4-p0-a{a}.bin' for a in range(2)]
   with paths[0].open('r+b') as f:f.seek(28);f.write(b'\0\0')
   with self.assertRaisesRegex(ValueError,'input'):A.check_pair(paths,4,0)
 def test_budget_and_gate(self):
  rows=samples();r=A.analyze_samples(rows);self.assertEqual(len(r['cells']),30);self.assertEqual(len(r['scenarios']),6);self.assertEqual(32+360*2,752)
  self.assertEqual(r['scenarios'][0]['signed_weighted_delta_ms'],216)
  self.assertEqual(r['scenarios'][0]['optimistic_clipped_weighted_delta_ms'],1836)
  self.assertEqual(r['gate'],'INVESTIGATION_ONLY')
  for row in rows:row['ms']=1
  self.assertEqual(A.analyze_samples(rows)['gate'],'PARK_WITHOUT_MODEL_QUALIFICATION')
  for row in rows:row['ms']=1 if row['arm']==0 else 2
  r=A.analyze_samples(rows);self.assertEqual(r['largest_scenario_optimistic_ms'],0);self.assertTrue(all(s['signed_weighted_delta_ms']<0 for s in r['scenarios']))
 def test_bad_samples_and_json(self):
  for key,value in [('ms',float('nan')),('ms',0),('ms',True),('arm',1),('launches',35)]:
   rows=samples();rows[0][key]=value
   with self.assertRaises(ValueError):A.analyze_samples(rows)
  with self.assertRaises(ValueError):json.loads('{"rc":7,"rc":0}',object_pairs_hook=A.unique)
 def test_raw_event_agreement(self):
  rows=samples();events=[dict(type='event_result',index=x['index'],ms_bits=struct.unpack('<I',struct.pack('<f',x['ms']))[0],rc=0) for x in rows];A.check_event_results(rows,events)
  events[0]['ms_bits']^=1
  with self.assertRaises(ValueError):A.check_event_results(rows,events)
 def test_no_cross_scenario_mixing(self):
  rows=samples()
  for row in rows:
   gain=(row['pattern']==0 and row['descriptor']==0) or (row['pattern']==1 and row['descriptor']==1)
   row['ms']=2 if row['arm']==0 else (1 if gain else 2)
  self.assertEqual(A.analyze_samples(rows)['largest_scenario_optimistic_ms'],1728)
 def test_pins(self):
  a=SimpleNamespace(driver=R.HERE/'screen.cpp',driver_sha=R.sha(R.HERE/'screen.cpp'),output=Path('/tmp/not-used-screen'),leave_api_stopped=True);R.validate(a)
 def test_failure_preservation(self):
  for rc,raw in ((7,'{"type":"failure"}\n'),(0,'malformed\n')):
   with tempfile.TemporaryDirectory() as t:
    a=SimpleNamespace(driver=R.HERE/'screen.cpp',driver_sha=R.sha(R.HERE/'screen.cpp'),output=Path(t)/'out',leave_api_stopped=True);gpu=[dict(pci='mock0',mode='high',power1_cap=190000000)]
    class P:
     returncode=rc
     def __init__(self,cmd,stdout,stderr):stdout.write(raw);stdout.flush();stderr.write('preserved');stderr.flush();(a.output/'preflight-d0-p0-a0.bin').write_bytes(b'raw failure evidence')
     def poll(self):return self.returncode
     def wait(self,timeout):return self.returncode
    with patch.object(R.N,'call',return_value=''),patch.object(R.N,'devices',return_value=[]),patch.object(R.N,'gpu',return_value=gpu),patch.object(R.N,'cooldown',return_value=gpu),patch.object(R.subprocess,'Popen',P),patch.object(R.S,'cleanup') as cleanup:
     with self.assertRaises((RuntimeError,ValueError)):R.execute(a)
     receipt=json.loads((a.output/'summary.json').read_text());self.assertFalse(receipt['passed']);self.assertEqual(receipt['exit_code'],rc);self.assertEqual(cleanup.call_count,1);self.assertEqual(receipt['raw_hashes']['preflight-d0-p0-a0.bin'],hashlib.sha256(b'raw failure evidence').hexdigest())
if __name__=='__main__':unittest.main()
