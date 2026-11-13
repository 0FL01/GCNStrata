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
   root=Path(t);source=root/'original';build=source/'scratch-build';build.mkdir(parents=True)
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
 def test_observer_pins_and_mounts(self):
  self.assertEqual(B.validate_baseline(),B.PATCHED);cmd=R.command(R.SOURCE,Path('/tmp/fresh'),'test')
  self.assertIn(str(B.V4)+':/v4:ro',cmd);self.assertIn('/tmp/fresh/header-overlay:/src/include/strata/prefill:ro',cmd)
  for relative in ('src/prefill/gemm.cu','src/prefill/prefill.cpp'):self.assertIn(str(R.HERE/'patched'/relative)+':/src/'+relative+':ro',cmd)
 def test_archive_replacement_rejects_other_member(self):
  before=[dict(name=n,sha256=h) for n,h in B.OBJECTS.items()]+[dict(name='other.o',sha256='same')];changed={n:'new'+n for n in B.OBJECTS};after=[dict(name=r['name'],sha256=changed.get(r['name'],r['sha256'])) for r in before];B.check_replacement(before,after,changed);after[-1]['sha256']='changed'
  with self.assertRaisesRegex(RuntimeError,'unrelated'):B.check_replacement(before,after,changed)
 def test_v4_receipts_and_every_closure_hash_required(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'reproduction').mkdir();(p/'scratch-build').mkdir();files={}
   for name in B.closure():
    f=p/'scratch-build'/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(name.encode());files['scratch-build/'+name]=B.sha(f)
   (p/'scratch-build/strata').write_bytes(b'baseline');base=B.sha(p/'scratch-build/strata');inner=dict(passed=True,status='PASS_ORIGINAL_OBJECT_AND_FROZEN_BINARY_REPRODUCTION',relinked_binary=dict(equal=True,actual_sha256=base));B.write(p/'reproduction/receipt.json',inner);ih=B.sha(p/'reproduction/receipt.json');host=dict(passed=True,status='PASS_ORIGINAL_SOURCE_PROVENANCE',returncode=0,cleanup=dict(passed=True),inner_receipt_sha256=ih,artifact_hashes=files);B.write(p/'host-receipt.json',host)
   with patch.object(B,'BASE',base),patch.object(B,'V4_HOST',B.sha(p/'host-receipt.json')),patch.object(B,'V4_INNER',ih),patch.object(B,'REPAIRED_ENGINE',files['scratch-build/libstrata_engine.a']):
    result=B.validate_v4(p);self.assertEqual(len(result['files']),len(B.closure()));(p/'scratch-build/libstrata_engine.a').write_bytes(b'changed')
    with self.assertRaisesRegex(RuntimeError,'v4 closure mismatch'):B.validate_v4(p)
 def test_header_overlay_copy_and_change_rejection(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'source';src=source/'include/strata/prefill';src.mkdir(parents=True);(src/'gemm.hpp').write_bytes(b'original');out=root/'out';out.mkdir();receipt=B.prepare_headers(source,out);B.check_headers(source,out,receipt);self.assertEqual((out/'header-overlay/gemm.hpp').read_bytes(),b'original');(src/'gemm.hpp').write_bytes(b'changed')
   with self.assertRaisesRegex(RuntimeError,'header changed'):B.check_headers(source,out,receipt)
 def test_real_device_payload_self_check_and_mutation_rejection(self):
  import device_compare as D
  baseline=Path('/home/opencode/ai/recovered-c183-input-20261009/strata')
  if not baseline.exists():self.skipTest('controller-only immutable baseline fixture unavailable')
  result=D.compare(baseline,baseline);self.assertEqual(len(result['baseline']['modules']),55)
  raw=baseline.read_bytes();_,secs=D.sections(raw,62);fat=next(x for x in secs if x['name']=='.hip_fatbin');first=result['baseline']['modules'][0];at=fat['offset']+first['payloads'][0]['offset']
  with tempfile.TemporaryDirectory() as t:
   changed=bytearray(raw);changed[at]=0;f=Path(t)/'bad';f.write_bytes(changed)
   with self.assertRaisesRegex(RuntimeError,'ELF format'):D.compare(baseline,f)
  with self.assertRaisesRegex(RuntimeError,'bounds'):D.sections(raw[:64],62)
if __name__=='__main__':unittest.main()
