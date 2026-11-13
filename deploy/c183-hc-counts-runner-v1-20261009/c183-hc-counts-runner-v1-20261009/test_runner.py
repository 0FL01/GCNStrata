import copy,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_counts as R

def fixture():
 rows=[]
 for device,(lo,hi) in enumerate(((0,27),(27,48))):
  context=device+1;key=dict(context=context,generation=context);n=4*(hi-lo)
  rows.append(dict(type='hc_observer_lifecycle',**key,bound=True,thread_changes=0,binding_changes=0,ticket_mismatches=0,inflight=0))
  rows.append(dict(type='hc_observer_summary',**key,device=device,rebind_epoch=0,valid=True,attempts=n*3,eligible=n*3,guards=n*3,original_attempts=n*3,original_taken=n*3,unmapped=0,ambiguous=0,overflows=0,registry_overflows=0,device_query_errors=0,layer_begin=lo,layer_end=hi,chunk_T=[4096,4096,4095],chunk_p0=[0,4096,8192],stages=[0]*12+[n*3]))
  for ordinal in range(4*lo+1,4*hi+1):
   rows.append(dict(type='hc_observer_weight',**key,tensor_ordinal=ordinal,N=320 if ordinal%2 else 10240,K=10240 if ordinal%2 else 320,declarations=3,address_epoch=0,pointer_changes=0,within_generation_W_mask_changes=0,first_observations=1,repeats_same_visible_generation=2,W_bad_repeats_same_visible_generation=0,repeats_changed_visible_generation=0,attempts_by_T=[1,2],eligible_by_T=[1,2],stages_4095=[0]*12+[1],stages_4096=[0]*12+[2],mask_hist_4095=[1]+[0]*63,mask_hist_4096=[2]+[0]*63))
  rows.append(dict(type='hc_observer_end',**key,weight_rows=n))
 return '\n'.join(json.dumps(row,separators=(',',':')) for row in rows)+'\n'

class Tests(unittest.TestCase):
 def test_complete_mock_destructor_after_quit(self):
  raw=fixture();text='INFO expert_slots=19078 expert_slots_primary=9900 kv=int8 context=204800 lookup=3 mtp_max=4 spec=6 kv_resident=32768 pcie_frac=0.00\nREADY 204800\nT 7\nDONE 1 12288 1.0 1.0 length 0 0 0 0 0 0 0 0.0 12288 0\n'
  class Input(io.StringIO):
   def close(self):self.saved=self.getvalue();super().close()
  class Process:
   def __init__(self):self.stdout=io.StringIO(text);self.stdin=Input();self.returncode=None;self.err=None
   def poll(self):return self.returncode
   def wait(self,timeout=None):
    if self.returncode is None:self.err.write(raw);self.err.flush()
    self.returncode=0;return 0
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);out=root/'out';out.mkdir();binary=root/'binary';binary.write_bytes(b'mock');p=Process();original=Path.read_text
   def read(path,*args,**kwargs):return 'MemAvailable: 999999999 kB\n' if str(path)=='/proc/meminfo' else original(path,*args,**kwargs)
   def popen(*args,**kwargs):p.err=kwargs['stderr'];p.err.write('locked in RAM\n');p.err.flush();return p
   arm=dict(binary=str(binary),binary_sha256=R.N.sha(binary),release_artifacts={},flags={},native_args=[])
   with patch.object(R.A,'R',root),patch.object(R.A,'call',return_value=''),patch.object(R.A,'cooldown',return_value=[]),patch.object(R.A,'gpu',return_value=[]),patch.object(R.A.subprocess,'Popen',side_effect=popen),patch.object(Path,'read_text',read):
    result=R.A.run_arm(R.TAG,arm,dict(requested_outputs=1,fixture_sha256='mock'),[1]*12288,[],out)
   self.assertEqual(p.stdin.saved.count('GEN '),1);self.assertEqual(p.stdin.saved.count('QUIT\n'),1)
   self.assertEqual(result['output_ids'],[7]);self.assertNotIn('PP',result)
   err=out/'private'/f'{R.TAG}.stderr';before=err.read_bytes();parsed=R.P.parse(err.read_text())
   self.assertEqual(sum(c['attempts'] for c in parsed['contexts']),576);self.assertTrue(parsed['full_attempt_coverage'])
   with self.assertRaises(ValueError):R.P.parse(err.read_text().replace('"original_taken":324','"original_taken":323'))
   self.assertEqual(before,err.read_bytes())
 def test_strict_parser_rejects_coverage_and_counter_errors(self):
  rows=[json.loads(x) for x in fixture().splitlines()]
  for field,value in [('chunk_T',[4096,4096,4096]),('chunk_p0',[0,4096,8193]),('layer_end',26),('original_attempts',323),('original_taken',323),('guards',323)]:
   changed=copy.deepcopy(rows);changed[1][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):R.P.parse('\n'.join(json.dumps(x,separators=(',',':')) for x in changed))
  with self.assertRaises(ValueError):R.P.parse('\n'.join(fixture().splitlines()[:-1]))
 def test_input_pins(self):
  for name,h in json.loads((R.HERE/'input-pins.json').read_text()).items():self.assertEqual(R.N.sha(R.HERE/name),h)
 def test_changed_receipt_fails(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'receipt';p.write_text('{"passed":true}')
   with self.assertRaises(RuntimeError):R.read_bound(p,'0'*64)
 def test_watchdog_and_lock_static(self):
  arm=(R.HERE/'native_arm.py').read_text();wrapper=(R.HERE/'run_counts.py').read_text()
  self.assertIn('now-started<300',arm);self.assertIn('shutdown_started<60',arm)
  self.assertIn('/tmp/strata-direct-ff34-ce.lock',wrapper);self.assertIn('os.O_CREAT|os.O_EXCL',wrapper)
  self.assertLess(wrapper.index('A.run_arm('),wrapper.index("summary['counts']=P.parse"))
if __name__=='__main__':unittest.main()

class AdmissionTests(unittest.TestCase):
 def test_build_review_and_binary_gate_mutations(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);binary=d/'binary';binary.write_bytes(b'mock observer');baseline=d/'baseline';baseline.write_bytes(b'mock baseline')
   sha=R.N.sha(binary);gate=dict(passed=True,complete_device_payload_bytes_equal=True,complete_module_kernel_metadata_equal=True,baseline=dict(binary_sha256=R.BASE,modules=[{}]*55),observer=dict(binary_sha256=sha,modules=[{}]*55))
   inner=dict(passed=True,status='PASS_HOST_ONLY_OBSERVER_BUILD_AND_DEVICE_EQUALITY',observer_binary_sha256=sha,device_equality_sha256='',v4_inputs=dict(binary_sha256=R.BASE,host_sha256=R.V4_HOST,inner_sha256=R.V4_INNER))
   host=dict(passed=True,status='PASS_COUNTS_OBSERVER_BUILD_DEVICE_EQUALITY',returncode=0,cleanup=dict(passed=True),inner_receipt_sha256='',artifact_hashes={'scratch-build/strata':sha})
   review=dict(passed=True,source_and_cpu_review_passed=True,build_provenance_and_device_gate_passed=True,bindings={})
   a=SimpleNamespace(binary=binary,baseline=baseline,binary_sha=sha)
   def save():
    for name,value in [('device_gate',gate),('inner_receipt',inner),('build_receipt',host),('review_receipt',review)]:
     if name=='inner_receipt':value['device_equality_sha256']=a.device_gate_sha
     if name=='build_receipt':value['inner_receipt_sha256']=a.inner_receipt_sha
     if name=='review_receipt':value['bindings']=dict(binary_sha256=sha,build_receipt_sha256=a.build_receipt_sha,inner_receipt_sha256=a.inner_receipt_sha,device_gate_sha256=a.device_gate_sha,runner_sha256=R.N.sha(R.HERE/'run_counts.py'),input_pins_sha256=R.N.sha(R.HERE/'input-pins.json'))
     path=d/name;path.write_text(json.dumps(value));setattr(a,name,path);setattr(a,name+'_sha',R.N.sha(path))
   save();original_sha=R.N.sha
   def sh(path):return R.BASE if Path(path)==baseline else original_sha(path)
   with patch.object(R.N,'sha',side_effect=sh),patch.object(R.D,'compare',return_value=gate) as compare:
    self.assertEqual(len(R.inputs(a)[1]),12288);self.assertEqual(compare.call_count,1)
    for target,key,bad in [(host,'passed',False),(host,'returncode',1),(inner,'passed',False),(gate,'complete_device_payload_bytes_equal',False),(gate,'complete_module_kernel_metadata_equal',False),(review,'source_and_cpu_review_passed',False),(review,'build_provenance_and_device_gate_passed',False)]:
     old=target[key];target[key]=bad;save()
     with self.subTest(key=key),self.assertRaises(RuntimeError):R.inputs(a)
     target[key]=old;save()
    a.binary_sha='0'*64
    with self.assertRaises(RuntimeError):R.inputs(a)
