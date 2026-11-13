import hashlib,json,tempfile,unittest,struct,subprocess,signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_exhaustive as R
import recheck
class Tests(unittest.TestCase):
 def test_compiled_cpu_abi_serializer_binary(self):
  with tempfile.TemporaryDirectory() as t:
   binary=Path(t)/'core-test'
   subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-O2',str(R.HERE/'test_core.cpp'),'-o',str(binary)],check=True)
   self.assertEqual(subprocess.run([str(binary),'--watchdog'],timeout=3).returncode,-signal.SIGALRM)
   rows=subprocess.check_output([str(binary),t],text=True).splitlines()
   self.assertEqual(json.loads(rows[0]),dict(type='hip_rc',operation='quote" slash\\ tab\t newline\n carriage\r \x01',rc=7))
   data=(Path(t)/'fixture.bin').read_bytes();self.assertEqual(recheck.parse(data,(0,4096,32768,2,1,0),initial=True),2)
   self.assertEqual(recheck.parse((Path(t)/'tail.bin').read_bytes(),(1,512,0,1,448,5)),1)
   for pos in (0,28,28+12*2,28+12*4,28+12*4+4,28+12*4+16*4):
    bad=bytearray(data);bad[pos]^=1
    with self.assertRaises(ValueError):recheck.parse(bad,(0,4096,32768,2,1,0))
   with self.assertRaises(ValueError):recheck.parse(data[:-1],(0,4096,32768,2,1,0))
   for index,value in ((0,1),(1,2),(4,1),(15,1)):
    bad=bytearray(data)
    for r in range(4):struct.pack_into('<I',bad,28+12*4+(r*23+1+index)*4,value)
    with self.assertRaisesRegex(ValueError,'stats valid'):recheck.parse(bad,(0,4096,32768,2,1,0))
 def test_duplicate_json_rejected(self):
  with self.assertRaisesRegex(ValueError,'duplicate JSON'):json.loads('{"rc":7,"rc":0}',object_pairs_hook=recheck.unique_object)
 def test_exact_scope(self):
  specs=list(recheck.specs());self.assertEqual(len(specs),548);self.assertEqual(sum(x[3]*2 for x in specs),262216)
  self.assertEqual(sum(x[3]*2 for x in specs if x[0]==1),72)
  for scale in (4096,512):self.assertEqual([p for x in specs if x[0]==0 and x[1]==scale for p in range(x[2],x[2]+x[3])],list(range(65536)))
 def test_validate(self):
  with tempfile.TemporaryDirectory() as t:
   a=SimpleNamespace(driver=R.HERE/'exhaustive.cpp',driver_sha=R.sha(R.HERE/'exhaustive.cpp'),output=Path(t)/'out',leave_api_stopped=True);R.validate(a);a.driver_sha='0'*64
   with self.assertRaisesRegex(RuntimeError,'driver changed'):R.validate(a)
 def mock_failure(self,rc,malformed):
  with tempfile.TemporaryDirectory() as t:
   a=SimpleNamespace(driver=R.HERE/'exhaustive.cpp',driver_sha=R.sha(R.HERE/'exhaustive.cpp'),output=Path(t)/'out',leave_api_stopped=True)
   gpu=[dict(pci='mock0',mode='high',power1_cap=190000000)]
   class P:
    returncode=rc
    def __init__(self,cmd,stdout,stderr):
     stdout.write('malformed\n' if malformed else '{"type":"failure","rc":7}\n');stdout.flush();stderr.write('failure preserved\n');stderr.flush();(a.output/'batch-0000.initial.bin').write_bytes(b'initial evidence')
    def poll(self):return self.returncode
    def wait(self,timeout):return self.returncode
   with patch.object(R.N,'call',return_value=''),patch.object(R.N,'devices',return_value=[]),patch.object(R.N,'gpu',return_value=gpu),patch.object(R.N,'cooldown',return_value=gpu),patch.object(R.subprocess,'Popen',P),patch.object(R.S,'cleanup') as cleanup:
    with self.assertRaises((RuntimeError,ValueError)):R.execute(a)
    self.assertEqual(cleanup.call_count,1);result=json.loads((a.output/'summary.json').read_text());self.assertFalse(result['passed']);self.assertEqual(result['exit_code'],rc)
    self.assertEqual(result['raw_hashes']['batch-0000.initial.bin'],hashlib.sha256(b'initial evidence').hexdigest());self.assertIn('raw.jsonl',result['raw_hashes']);self.assertTrue(result['cleanup']['passed'])
 def test_nonzero_failure_preservation(self):self.mock_failure(7,False)
 def test_parser_failure_preservation(self):self.mock_failure(0,True)
if __name__=='__main__':unittest.main()
