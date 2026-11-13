import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_qualification as R
class AdmissionTests(unittest.TestCase):
 def test_build_review_and_binary_gate_mutations(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);binary=d/'binary';binary.write_bytes(b'mock observer');baseline=d/'baseline';baseline.write_bytes(b'mock baseline')
   sha=R.N.sha(binary);gate=dict(passed=True,complete_device_payload_bytes_equal=True,complete_module_kernel_metadata_equal=True,baseline=dict(binary_sha256=R.BASE,modules=[{}]*55),observer=dict(binary_sha256=sha,modules=[{}]*55))
   inner=dict(observer_sources={k:v for k,v in json.loads((R.HERE/'source-pins.json').read_text())['candidate_sha256'].items() if k!='include/strata/prefill/gemm.hpp'},original_inputs=json.loads((R.HERE/'source-pins.json').read_text())['baseline_sha256'],source_review=dict(reviewed_generate_flow_only=True,sha256='f46c48f0fb41a819d1fbe9d2c02783a185f8d16878c57363a5e928ca018ff4e6',source_archive_sha256='f9841342c4f84c33b8e4a3052269b4b413e1152c4d499b3f78ba92676152b54b'),tls_sharing=dict(passed=True,objects=[{},{}],symbol='_ZN6strata7prefill11hc_negative7currentE'),passed=True,status='PASS_PRIMARY_CACHE_BUILD_DEVICE_AND_TLS_EQUALITY',observer_binary_sha256=sha,device_equality_sha256='',v4_inputs=dict(binary_sha256=R.BASE,host_sha256=R.V4_HOST,inner_sha256=R.V4_INNER))
   host=dict(passed=True,status='PASS_PRIMARY_CACHE_BUILD_DEVICE_AND_TLS_EQUALITY',returncode=0,cleanup=dict(passed=True),inner_receipt_sha256='',artifact_hashes={'scratch-build/strata':sha})
   review=dict(passed=True,source_and_cpu_review_passed=True,build_provenance_and_device_gate_passed=True,bindings={})
   a=SimpleNamespace(binary=binary,baseline=baseline,binary_sha=sha)
   def save():
    for name,value in [('device_gate',gate),('inner_receipt',inner),('build_receipt',host),('review_receipt',review)]:
     if name=='inner_receipt':value['device_equality_sha256']=a.device_gate_sha
     if name=='build_receipt':value['inner_receipt_sha256']=a.inner_receipt_sha
     if name=='review_receipt':value['bindings']=dict(binary_sha256=sha,build_receipt_sha256=a.build_receipt_sha,inner_receipt_sha256=a.inner_receipt_sha,device_gate_sha256=a.device_gate_sha,runner_sha256=R.N.sha(R.HERE/'run_qualification.py'),input_pins_sha256=R.N.sha(R.HERE/'input-pins.json'))
     path=d/name;path.write_text(json.dumps(value));setattr(a,name,path);setattr(a,name+'_sha',R.N.sha(path))
   save();original_sha=R.N.sha
   def sh(path):return R.BASE if Path(path)==baseline else original_sha(path)
   with patch.object(R.N,'sha',side_effect=sh),patch.object(R.D,'compare',return_value=gate) as compare:
    self.assertEqual(len(R.inputs(a)[1]),12288);self.assertEqual(compare.call_count,1)
    for target,key,bad in [(inner,'observer_sources',{}),(inner,'tls_sharing',dict(passed=False,objects=[{},{}])),(host,'passed',False),(host,'returncode',1),(inner,'passed',False),(gate,'complete_device_payload_bytes_equal',False),(gate,'complete_module_kernel_metadata_equal',False),(review,'source_and_cpu_review_passed',False),(review,'build_provenance_and_device_gate_passed',False)]:
     old=target[key];target[key]=bad;save()
     with self.subTest(key=key),self.assertRaises(RuntimeError):R.inputs(a)
     target[key]=old;save()
    a.binary_sha='0'*64
    with self.assertRaises(RuntimeError):R.inputs(a)
