import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_sanity as R
class Tests(unittest.TestCase):
 def test_fixed_scope(self):
  s=(R.HERE/'sanity.cpp').read_text()
  self.assertIn('0x0000,0x8000,0x3f80,0x0001,0x7f80,0x7fc0,0x7f81',s)
  self.assertIn('constexpr float scales[]={4096.f,512.f}',s)
  self.assertEqual(s.count('hipModuleLaunchKernel('),1)
  self.assertIn('int64_t n=1',s);self.assertIn('hipSetDevice(0)',s)
  self.assertNotIn('__global__',s);self.assertNotIn('hipEvent',s)
  self.assertIn('std::strcmp(pci,"0000:05:00.0")',s)
  self.assertIn("timeout=2",(R.HERE/'run_sanity.py').read_text())
  self.assertIn('a.stats==z.stats',s);self.assertIn('guards(a,bits)&&guards(z,bits)',s)
 def test_validate(self):
  with tempfile.TemporaryDirectory() as t:
   a=SimpleNamespace(driver=R.HERE/'sanity.cpp',driver_sha=R.sha(R.HERE/'sanity.cpp'),output=Path(t)/'out',leave_api_stopped=True)
   R.validate(a)
   a.driver_sha='0'*64
   with self.assertRaisesRegex(RuntimeError,'driver changed'):R.validate(a)
 def mock_run(self,rc):
  with tempfile.TemporaryDirectory() as t:
   a=SimpleNamespace(driver=R.HERE/'sanity.cpp',driver_sha=R.sha(R.HERE/'sanity.cpp'),output=Path(t)/'out',leave_api_stopped=True)
   gpu=[{'pci':'mock0','mode':'high','power1_cap':190000000},{'pci':'mock1','mode':'high','power1_cap':190000000}]
   class P:
    returncode=rc
    def __init__(self,cmd,stdout,stderr):
     rows=[{'type':'raw'}]*28+[{'type':'pair','equal':True}]*14+[{'type':'complete','pairs':14,'launches':28,'passed':True}]
     stdout.write('\n'.join(map(json.dumps,rows))+'\n');stdout.flush()
    def poll(self):return self.returncode
    def wait(self,timeout):return self.returncode
   with patch.object(R.N,'call',return_value=''),patch.object(R.N,'devices',return_value=[]),patch.object(R.N,'gpu',return_value=gpu),patch.object(R.N,'cooldown',return_value=gpu),patch.object(R.subprocess,'Popen',P),patch.object(R.S,'cleanup') as cleanup:
    if rc==0:R.execute(a)
    else:
     with self.assertRaisesRegex(RuntimeError,'driver exit'):R.execute(a)
    self.assertEqual(cleanup.call_count,1)
    result=json.loads((a.output/'summary.json').read_text());self.assertEqual(result['passed'],rc==0);self.assertEqual(result['exit_code'],rc)
    self.assertIn('raw.jsonl',result['raw_hashes']);self.assertTrue(result['cleanup']['passed'])
 def test_mock_success(self):self.mock_run(0)
 def test_mock_failure_preserves_raw(self):self.mock_run(7)
if __name__=='__main__':unittest.main()

