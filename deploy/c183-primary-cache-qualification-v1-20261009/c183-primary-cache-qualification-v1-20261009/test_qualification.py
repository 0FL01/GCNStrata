import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import run_qualification as R

def fixture():
 p=json.loads((R.HERE/'baseline-proof.json').read_text())
 return '\n'.join(['strata hc-wneg: device=0 candidates=324 hits=22 learned=11 mismatches=0']+['strata hc-wneg-key: '+' '.join(k+'='+str(v) for k,v in x.items()) for x in p['keys']]+['strata hc-f16 route: attempts=324 taken=99','strata hc-f16 route: attempts=252 taken=104'])
def native():return dict(passed=True,output_ids=[248046],prompt_tokens=12288,prompt_read=12288,prompt_reused=0,actual_outputs=1)
STDOUT='READY 204800\nRESUME 0\nPP 4096 12288 1 1\nPP 8192 12288 1 1\nPP 12287 12288 1 1\nREUSED 0\nT 248046\nDONE 1 12288\n'
class Tests(unittest.TestCase):
 def test_parser(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'stdout';p.write_text(STDOUT)
   self.assertEqual(R.P.parse(fixture(),native(),p)['primary_physical_guards_inferred'],302)
   for a,b in [('device=0','device=1'),('candidates=324','candidates=323'),('hits=22','hits=21'),('learned=11','learned=10'),('mismatches=0','mismatches=1'),('mask=4','mask=0'),('taken=99','taken=98'),('ordinal=0','ordinal=1'),('N=','N=-')]:
    with self.subTest(a=a),self.assertRaises(ValueError):R.P.parse(fixture().replace(a,b,1),native(),p)
   for text in [fixture()+'\n'+fixture().splitlines()[0],fixture()+'\n'+fixture().splitlines()[1],fixture()+'\nstrata hc-wneg: malformed',fixture()+'\n{"type":"hc_observer_summary"}']:
    with self.assertRaises(ValueError):R.P.parse(text,native(),p)
   bad=native();bad['output_ids']=[7]
   with self.assertRaises(ValueError):R.P.parse(fixture(),bad,p)
   p.write_text(STDOUT.replace('PP 12287','PP 12288'))
   with self.assertRaises(ValueError):R.P.parse(fixture(),native(),p)
 def test_failure_and_single_use(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);campaign=root/'logs/perf-campaign-20261009';campaign.mkdir(parents=True);binary=root/'binary';binary.write_bytes(b'mock');baseline=root/'base';baseline.write_bytes(b'base')
   a=SimpleNamespace(leave_api_stopped=True,binary=binary,baseline=baseline,binary_sha=R.N.sha(binary),output=root/'out')
   def arm(tag,r,s,ids,ds,out):
    d=out/'private';d.mkdir();(d/(tag+'.stderr')).write_text(fixture().replace('hits=22','hits=21'));(d/(tag+'.stdout')).write_text(STDOUT);(d/(tag+'.termination.json')).write_text(json.dumps(dict(native_exit=0,native_completed=True,reader_finished=True,reader_error=None,native_cleanup_error=None)));return native()
   with patch.object(R.N,'R',root),patch.object(R,'inputs',return_value=(dict(environment={},native_args=[],prefix_sha256='fixture'),[1]*12288,{})),patch.object(R.N,'call',side_effect=lambda cmd:'[]' if 'inspect' in cmd else ''),patch.object(R.N,'devices',return_value=[]),patch.object(R.N,'gpu',return_value=[]),patch.object(R.A,'run_arm',side_effect=arm) as run,patch.object(R.S,'cleanup'):
    with self.assertRaises(ValueError):R.execute(a)
    self.assertEqual(run.call_count,1);self.assertEqual((a.output/'private'/f'{R.TAG}.stderr').read_text(),fixture().replace('hits=22','hits=21'));self.assertEqual((a.output/'private'/f'{R.TAG}.stdout').read_text(),STDOUT);self.assertTrue((campaign/('hc-primary-qual-'+a.binary_sha+'.once')).exists());self.assertFalse(json.loads((a.output/'summary.json').read_text())['completed'])
    a.output=root/'retry'
    with self.assertRaises(FileExistsError):R.execute(a)
    self.assertEqual(run.call_count,1)
 def test_base_safety_unchanged(self):
  for n in ['native_arm.py','native_support.py','qsa_probe_safety.py','device_compare.py']:
   self.assertEqual((R.HERE/n).read_bytes(),(R.HERE.parent/'c183-hc-cost-runner-v1-20261009'/n).read_bytes())
 def test_pins(self):
  for n,h in json.loads((R.HERE/'input-pins.json').read_text()).items():self.assertEqual(R.N.sha(R.HERE/n),h)

class NativeLifetimeTests(unittest.TestCase):
 def test_destructor_rows_arrive_only_after_quit(self):
  import io
  text=STDOUT.replace('READY 204800\n','INFO expert_slots=19078 expert_slots_primary=9900 kv=int8 context=204800 lookup=3 mtp_max=4 spec=6 kv_resident=32768 pcie_frac=0.00\nREADY 204800\n').replace('DONE 1 12288\n','DONE 1 12288 1.0 1.0 length 0 0 0 0 0 0 0 0.0 12288 0\n')
  class Input(io.StringIO):
   def close(self):self.saved=self.getvalue();super().close()
  class Process:
   def __init__(self):self.stdout=io.StringIO(text);self.stdin=Input();self.returncode=None;self.err=None
   def poll(self):return self.returncode
   def wait(self,timeout=None):
    if self.returncode is None:
     assert self.stdin.saved.count('GEN ')==self.stdin.saved.count('QUIT\n')==1
     self.err.write(fixture());self.err.flush()
    self.returncode=0;return 0
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);out=root/'out';out.mkdir();binary=root/'binary';binary.write_bytes(b'mock');proc=Process();original=Path.read_text
   def read(path,*args,**kwargs):return 'MemAvailable: 999999999 kB\n' if str(path)=='/proc/meminfo' else original(path,*args,**kwargs)
   def popen(*args,**kwargs):proc.err=kwargs['stderr'];proc.err.write('locked in RAM\n');proc.err.flush();return proc
   arm=dict(binary=str(binary),binary_sha256=R.N.sha(binary),release_artifacts={},flags={},native_args=[])
   with patch.object(R.A,'R',root),patch.object(R.A,'call',return_value=''),patch.object(R.A,'cooldown',return_value=[]),patch.object(R.A,'gpu',return_value=[]),patch.object(R.A.subprocess,'Popen',side_effect=popen),patch.object(Path,'read_text',read):
    result=R.A.run_arm(R.TAG,arm,dict(requested_outputs=1,fixture_sha256='mock'),[1]*12288,[],out)
   parsed=R.P.parse((out/'private'/f'{R.TAG}.stderr').read_text(),result,out/'private'/f'{R.TAG}.stdout');self.assertEqual(parsed['primary_hits'],22)
