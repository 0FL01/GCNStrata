#!/usr/bin/env python3
import copy,json,pathlib,unittest
from unittest.mock import patch
import recovery_support as R
import build_capture as B
P=pathlib.Path(__file__).resolve().parent
class RecoveryTests(unittest.TestCase):
 def setUp(self):
  self.d=json.loads((P/'failed-build.fixture.json').read_text());self.package=json.loads((P/'original_package.v2.json').read_text())
 def test_original_evidence_bytes_and_narrow_acceptance(self):
  self.assertEqual(B.sha(P/'failed-build.fixture.json'),R.ORIGINAL_BUILD)
  self.assertEqual(B.sha(P/'original_package.v2.json'),R.ORIGINAL_PACKAGE)
  before=copy.deepcopy(self.d)
  self.assertTrue(R.validate_original(self.d,self.package,R.CANDIDATE))
  self.assertEqual(self.d,before)
  self.assertFalse(self.d['build_artifacts_restored']);self.assertEqual(self.d['restore_failed_files'],['build-906/strata'])
 def test_other_original_failures_never_waived(self):
  for key,value in [('compiler_confirmed_stopped',False),('sources_restored',False),('sources_invalidated',False),('exit_code',1),('select_object_unchanged',False),('kernel_archive_unchanged',False),('kernel_rebuild_detected',True),('build_artifacts_restored',True),('restore_failed_files',[]),('restore_failed_files',['src/core/verify.cpp']),('restore_failed_files',['build-906/strata','src/core/verify.cpp']),('error_type','CompileError'),('package_sha256','0'*64),('binary_sha256','a'*64)]:
   d=copy.deepcopy(self.d);d[key]=value
   with self.subTest(key=key),self.assertRaises(RuntimeError):R.validate_original(d,self.package,R.CANDIDATE)
 def test_compiled_source_and_baseline_cannot_change(self):
  d=copy.deepcopy(self.d);d['compiled_sources']['src/core/verify.cpp']='0'*64
  with self.assertRaises(RuntimeError):R.validate_original(d,self.package,R.CANDIDATE)
  d=copy.deepcopy(self.d);d['before']['build-906/strata']['mtime_ns']+=1
  with self.assertRaises(RuntimeError):R.validate_original(d,self.package,R.CANDIDATE)
  with self.assertRaises(RuntimeError):R.validate_original(self.d,self.package,'a'*64)
 def test_recovery_schema_exact_types_and_hashes(self):
  recovery=R.expected_recovery();self.assertTrue(R.validate_recovery_document(recovery))
  for key,value in [('original_build_sha256','0'*64),('evidence_sha256','0'*64),('restoration_verified',1),('gpu_executed',True),('native_executed',True),('restored_mtime_ns',0),('restore_target','/tmp/strata'),('restored_mode',0o777),('original_package_sha256','a'*64)]:
   d=copy.deepcopy(recovery);d[key]=value
   with self.subTest(key=key),self.assertRaises(RuntimeError):R.validate_recovery_document(d)
 def test_root_receipt_hash_cannot_be_arbitrary(self):
  with self.assertRaisesRegex(RuntimeError,'exact root recovery receipt SHA'):
   R.validate(self.d,P/'original_package.v2.json',B.O/'recovery.json','a'*64,R.CANDIDATE)
 def test_no_mutating_or_compilation_recovery_code(self):
  s=(P/'recovery_support.py').read_text()
  for forbidden in ['write_text(', 'write_bytes(', 'chmod(', 'chown(', 'copy2(', 'subprocess.', 'os.system(']:self.assertNotIn(forbidden,s)
  for f in ['build_capture.py','capture_arm.py','validate_200k.py','analyze_qsa_dump.py','qsa-200k-capture.patch','verify.capture.cpp']:
   self.assertEqual(B.sha(P/f),self.package['runner_file_sha256'][f] if f in self.package['runner_file_sha256'] else self.package[{'qsa-200k-capture.patch':'patch_sha256','verify.capture.cpp':'patched_verify_sha256'}[f]])
if __name__=='__main__':unittest.main()
