import io,os,json,shlex,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import build_inside as B
import run_reproduce as R
class Tests(unittest.TestCase):
 def test_actual_export_flags_and_link(self):
  raw=(R.HERE/'exported-build-metadata.txt').read_text();flags=raw.split('FILE CMakeFiles/strata_prefill.dir/flags.make',1)[1].split('FILE ',1)[0].strip();self.assertEqual(B.read_flags(flags),B.FLAGS)
  link=raw.split('FILE CMakeFiles/strata.dir/link.txt',1)[1].split('FILE ',1)[0].strip();self.assertEqual(shlex.split(link),B.LINK)
  self.assertIn('-DSTRATA_VERSION="0.1.40"',B.FLAGS['HIP_DEFINES'])
  with self.assertRaisesRegex(RuntimeError,'flag mismatch'):B.read_flags(flags.replace('-O3','-O2'))
 def test_exact_commands_output_only_changes(self):
  gemm,prefill,link=B.commands();self.assertEqual(gemm[0],'/opt/rocm/lib/llvm/bin/clang++');self.assertEqual(prefill[0],'/opt/rocm/llvm/bin/clang++');self.assertEqual(gemm[-4:],['-x','hip','-c','/src/src/prefill/gemm.cu']);self.assertEqual(prefill[-2:],['-c','/src/src/prefill/prefill.cpp'])
  for name,actual in zip(B.OBJECTS,(gemm,prefill)):
   expanded=B.RECIPES[name]
   for key,value in B.FLAGS.items():expanded=expanded.replace('$('+key+')',shlex.join(value))
   expected=shlex.split(expanded)
   self.assertEqual(actual,expected)
  self.assertEqual(link,B.LINK);self.assertEqual(link[link.index('-o')+1],'strata')
 def test_container_isolation_and_no_observer(self):
  cmd=R.command(R.SOURCE,Path('/tmp/fresh'),'unit-test');self.assertIn('--read-only',cmd);self.assertEqual(cmd[cmd.index('--network')+1],'none');self.assertIn('--pull=never',cmd);self.assertFalse(any(x=='--device' or x.startswith('--gpus') for x in cmd));self.assertIn(str(R.SOURCE)+':/src:ro',cmd);self.assertIn('/tmp/fresh:/out:rw',cmd);self.assertIn('/tmp/fresh/scratch-build:/src/build-906:rw',cmd);self.assertIn(str(R.SOURCE)+':/original:ro',cmd);self.assertEqual(cmd[-1],'/probe/build_inside.py');self.assertNotIn('hc_observer',str(cmd));self.assertIn(R.IMAGE,cmd)
 def test_nonzero_step_preserves_evidence(self):
  with tempfile.TemporaryDirectory() as t,patch.object(B,'OUT',Path(t)),patch.object(B,'BUILD',Path(t)):
   receipt={'steps':[]};argv=[sys.executable,'-c','import sys; print("kept stdout"); print("kept stderr", file=sys.stderr);sys.exit(7)']
   with self.assertRaisesRegex(RuntimeError,'step failed'):B.run_step(receipt,'nonzero',argv,5)
   saved=json.loads((Path(t)/'receipt.json').read_text());self.assertEqual(saved['steps'][0]['returncode'],7);self.assertEqual(saved['steps'][0]['argv'],argv);self.assertEqual((Path(t)/'nonzero.stdout').read_text(),'kept stdout\n');self.assertEqual((Path(t)/'nonzero.stderr').read_text(),'kept stderr\n')
 def test_timeout_kills_owned_process_group(self):
  with tempfile.TemporaryDirectory() as t,patch.object(B,'OUT',Path(t)),patch.object(B,'BUILD',Path(t)):
   receipt={'steps':[]}
   with self.assertRaises(subprocess.TimeoutExpired):B.run_step(receipt,'timeout',[sys.executable,'-c','import time; print("started",flush=True);time.sleep(10)'],.1)
   saved=json.loads((Path(t)/'receipt.json').read_text());self.assertTrue(saved['steps'][0]['timed_out']);self.assertEqual(saved['steps'][0]['returncode'],-9);self.assertIn('started',(Path(t)/'timeout.stdout').read_text())
 def test_hash_gates_and_order_are_fixed(self):
  self.assertEqual(B.OBJECTS,{'gemm.cu.o':'9b0d9fb8cf20c97bb81e43b363e6a6b68d32af0d36234b90ed058ad4433ad048','prefill.cpp.o':'efff91107cf3339e5929ac510097168da5c51d64012a93ece3894aa64b5d215d'})
  self.assertEqual(B.BASE,'c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049')
  src=(R.HERE/'build_inside.py').read_text();self.assertLess(src.index('check_engine_replacement(before,after,new_hash)',src.index('def reproduce():')),src.index("run_step(receipt,'relink-unchanged-original-inputs'"));self.assertIn('original archive member differs:',src);self.assertNotIn('STRATA_HC_COUNTS_ONLY_OBSERVER',src)
 def test_source_mismatch_keeps_host_receipt(self):
  with tempfile.TemporaryDirectory() as t,patch.object(R,'output_identity',return_value={}),patch.object(R,'validate_package'),patch.object(B,'validate_metadata',side_effect=RuntimeError('source mismatch expected=abc actual=def')),patch.object(R.subprocess,'run') as run:
   out=Path(t)/'fresh'
   with self.assertRaisesRegex(RuntimeError,'source mismatch'):R.execute(out)
   run.assert_not_called();receipt=json.loads((out/'host-receipt.json').read_text());self.assertFalse(receipt['passed']);self.assertEqual(receipt['status'],'FAILED_STOP_FOR_ANALYSIS');self.assertIn('actual=def',receipt['error'])
 def test_output_uid_contract(self):
  from types import SimpleNamespace
  out=Path('/fake/output');st=SimpleNamespace(st_uid=1000,st_gid=1000,st_mode=0o40700)
  with patch.object(os,'getuid',return_value=1000),patch.object(os,'geteuid',return_value=1000),patch.object(os,'getgid',return_value=1000),patch.object(os,'getegid',return_value=1000),patch.object(Path,'stat',return_value=st):
   identity=R.output_identity(out);cmd=R.command(R.SOURCE,out,'test');self.assertEqual(cmd[cmd.index('--user')+1],'1000:1000');self.assertEqual(identity['output_mode'],'0o700')
   st.st_mode=0o40750
   with self.assertRaisesRegex(RuntimeError,'output owner/mode'):R.output_identity(out)
  with patch.object(os,'getuid',return_value=0):
   with self.assertRaisesRegex(RuntimeError,'expected host uid1000'):R.output_identity(out)
 def test_earliest_inner_mkdir_failure(self):
  with tempfile.TemporaryDirectory() as t,patch.object(B,'OUT',Path(t)/'reproduction'),patch.object(Path,'mkdir',side_effect=PermissionError(13,'test mkdir denied')),patch.object(B,'validate_metadata') as validate,patch('sys.stderr',new_callable=io.StringIO) as err:
   with self.assertRaises(PermissionError):B.main()
   validate.assert_not_called();receipt=json.loads((Path(t)/'inner-failure.json').read_text());self.assertFalse(receipt['passed']);self.assertEqual(receipt['error_type'],'PermissionError');self.assertEqual(json.loads(err.getvalue())['inner_failure'],receipt)
 def test_unwritable_fallback_still_emits_json(self):
  with patch.object(B,'reproduce',side_effect=PermissionError(13,'mkdir denied')),patch.object(B,'write',side_effect=PermissionError(13,'fallback denied')),patch('sys.stderr',new_callable=io.StringIO) as err:
   with self.assertRaises(PermissionError):B.main()
   receipt=json.loads(err.getvalue())['inner_failure'];self.assertFalse(receipt['passed']);self.assertIn('fallback denied',receipt['fallback_receipt_error'])
 def test_minimal_fresh_copy_closure(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'original';build=source/'build-906';build.mkdir(parents=True)
   for name in B.closure():
    p=build/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
   (build/'excluded.bin').write_bytes(b'not needed');dest=root/'scratch';receipt=B.copy_build(source,dest);self.assertEqual(set(receipt['files']),set(B.closure()));self.assertFalse((dest/'excluded.bin').exists());B.verify_closure(source,receipt)
   for name in B.closure():self.assertNotEqual((build/name).stat().st_ino,(dest/name).stat().st_ino);self.assertEqual((dest/name).stat().st_nlink,1)
   (dest/B.PREFIX/'gemm.cu.o').write_bytes(b'new output');B.verify_closure(source,receipt)
   (build/B.PREFIX/'gemm.cu.o').write_bytes(b'changed original')
   with self.assertRaisesRegex(RuntimeError,'original closure changed'):B.verify_closure(source,receipt)
 def test_exact_original_output_spellings(self):
  gemm,prefill,link=B.commands();self.assertEqual(gemm[gemm.index('-o')+1],B.PREFIX+'gemm.cu.o')
  for flag,suffix in [('-o','prefill.cpp.o'),('-MF','prefill.cpp.o.d'),('-MT','prefill.cpp.o')]:self.assertEqual(prefill[prefill.index(flag)+1],B.PREFIX+suffix)
  self.assertEqual(str(B.BUILD),'/src/build-906');self.assertEqual(link,B.LINK)
 def test_verifier_recipe_pins_and_overlay(self):
  self.assertEqual(B.validate_baseline(),{k:v[1] for k,v in B.BASELINE.items()});argv=B.verifier_command()
  for flag,suffix in [('-o','verify.cpp.o'),('-MF','verify.cpp.o.d'),('-MT','verify.cpp.o')]:self.assertEqual(argv[argv.index(flag)+1],B.VERIFY_PREFIX+suffix)
  cmd=R.command(R.SOURCE,Path('/tmp/fresh'),'test')
  for relative,(filename,digest) in B.BASELINE.items():self.assertIn(str(R.HERE/'baseline'/filename)+':/src/'+relative+':ro',cmd)
 def test_replacement_rejects_any_other_member_change(self):
  before=[dict(name='a.o',sha256='same'),dict(name='verify.cpp.o',sha256=B.CAPTURE_VERIFY)]
  after=[dict(name='a.o',sha256='same'),dict(name='verify.cpp.o',sha256='new')]
  B.check_engine_replacement(before,after,'new')
  after[0]['sha256']='changed'
  with self.assertRaisesRegex(RuntimeError,'non-verifier'):B.check_engine_replacement(before,after,'new')
  with self.assertRaisesRegex(RuntimeError,'order/list'):B.check_engine_replacement(before,after[::-1],'new')
if __name__=='__main__':unittest.main()
