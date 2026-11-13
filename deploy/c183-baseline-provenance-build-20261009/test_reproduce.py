import json,shlex,subprocess,sys,tempfile,unittest
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
   for flag in ['-o','-MF']:
    if flag in expected:
     i=expected.index(flag)+1;expected[i]=str(B.OUT/'objects'/Path(expected[i]).name)
   self.assertEqual(actual,expected)
  expected=B.LINK.copy();expected[expected.index('-o')+1]=str(B.OUT/'strata.relinked');self.assertEqual(link,expected);self.assertNotIn('/out/reproduction/objects/gemm.cu.o',link)
 def test_container_isolation_and_no_observer(self):
  cmd=R.command(R.SOURCE,Path('/tmp/fresh'),'unit-test');self.assertIn('--read-only',cmd);self.assertEqual(cmd[cmd.index('--network')+1],'none');self.assertIn('--pull=never',cmd);self.assertFalse(any(x=='--device' or x.startswith('--gpus') for x in cmd));self.assertIn(str(R.SOURCE)+':/src:ro',cmd);self.assertIn('/tmp/fresh:/out:rw',cmd);self.assertEqual(cmd[-1],'/probe/build_inside.py');self.assertNotIn('hc_observer',str(cmd));self.assertIn(R.IMAGE,cmd)
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
  src=(R.HERE/'build_inside.py').read_text();self.assertLess(src.index('recompiled object mismatch:'),src.index("run_step(receipt,'relink-unchanged-original-inputs'"));self.assertIn('original archive member differs:',src);self.assertNotIn('STRATA_HC_COUNTS_ONLY_OBSERVER',src)
 def test_source_mismatch_keeps_host_receipt(self):
  with tempfile.TemporaryDirectory() as t,patch.object(R,'validate_package'),patch.object(B,'validate_metadata',side_effect=RuntimeError('source mismatch expected=abc actual=def')),patch.object(R.subprocess,'run') as run:
   out=Path(t)/'fresh'
   with self.assertRaisesRegex(RuntimeError,'source mismatch'):R.execute(out)
   run.assert_not_called();receipt=json.loads((out/'host-receipt.json').read_text());self.assertFalse(receipt['passed']);self.assertEqual(receipt['status'],'FAILED_STOP_FOR_ANALYSIS');self.assertIn('actual=def',receipt['error'])
if __name__=='__main__':unittest.main()
