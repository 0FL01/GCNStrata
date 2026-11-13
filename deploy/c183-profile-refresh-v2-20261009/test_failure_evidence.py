import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_profile as P

class Tests(unittest.TestCase):
    def test_completed_native_protocol_keeps_full_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'out';out.mkdir();binary=root/'strata';binary.write_bytes(b'test binary')
            text='INFO expert_slots=19078 expert_slots_primary=9900 kv=int8 context=204800 lookup=3 mtp_max=4 spec=6 kv_resident=32768 pcie_frac=0.00\nREADY 204800\nT 7\nT 8\nDONE 2 2 1.0 1.0 length 1 1 0 0 0 0 0 0.0 2 0\n'
            class Process:
                def __init__(self):self.stdout=io.StringIO(text);self.stdin=io.StringIO();self.returncode=None
                def poll(self):return self.returncode
                def wait(self,timeout=None):self.returncode=0;return 0
            p=Process();original_read=Path.read_text
            def read(path,*args,**kwargs):
                return 'MemAvailable: 999999999 kB\n' if str(path)=='/proc/meminfo' else original_read(path,*args,**kwargs)
            def popen(*args,**kwargs):kwargs['stderr'].write('locked in RAM\n');kwargs['stderr'].flush();return p
            r={'binary':str(binary),'binary_sha256':P.N.sha(binary),'release_artifacts':{},'flags':{},'native_args':[]}
            with patch.object(P.A,'R',root),patch.object(P.A,'call',return_value=''),patch.object(P.A,'cooldown',return_value=[]),patch.object(P.A,'gpu',return_value=[]),patch.object(P.A.subprocess,'Popen',side_effect=popen),patch.object(Path,'read_text',read):
                result=P.A.run_arm('success',r,{'requested_outputs':2,'fixture_sha256':P.FIXTURE_SHA},[1,2],[],out)
            self.assertEqual(result['output_ids'],[7,8]);self.assertEqual(result['native_done_field_count'],16)
            private=out/'private';self.assertEqual((private/'success.stdout').read_text(),text)
            status=json.loads((private/'success.termination.json').read_text())
            self.assertTrue(status['native_completed']);self.assertTrue(status['done_seen_by_parser'])
            self.assertIsNone(status['error']);self.assertEqual(status['native_exit'],0)
            self.assertEqual(json.loads((private/'success.partial-output-ids.json').read_text()),[7,8])

    def test_pre_done_failure_evidence(self):
        class Process:
            def __init__(self,text):self.stdout=io.StringIO(text);self.stdin=io.StringIO();self.returncode=None
            def poll(self):return self.returncode
            def wait(self,timeout=None):
                if getattr(self,'cleanup_fail',False):raise RuntimeError('deliberate process wait failure')
                self.returncode=1;return self.returncode
            def kill(self):self.returncode=-9
        cases=[('err','T 7\nERR diagnostic failure\n','RuntimeError'),
               ('eof','T 7\n','RuntimeError'),('malformed','T 7\nT invalid\nT 9\n','ValueError'),
               ('timeout','T 7\nT 8\n','RuntimeError'),('cancel','T 7\nT 8\n','RuntimeError'),
               ('cleanup','T 7\nERR diagnostic failure\n','RuntimeError')]
        for mode,text,error_type in cases:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);out=root/'out';out.mkdir();binary=root/'strata';binary.write_bytes(b'test binary')
                p=Process(text);p.cleanup_fail=mode=='cleanup';ticks=0;original_read=Path.read_text
                def read(path,*args,**kwargs):
                    return 'MemAvailable: 999999999 kB\n' if str(path)=='/proc/meminfo' else original_read(path,*args,**kwargs)
                def now():
                    nonlocal ticks
                    ticks+=1
                    if ticks==3 and mode=='cancel':(out/'cancel').touch()
                    return 1801.0 if mode=='timeout' and ticks>=3 else 0.0
                r={'binary':str(binary),'binary_sha256':P.N.sha(binary),'release_artifacts':{},'flags':{},'native_args':[]}
                with patch.object(P.A,'R',root),patch.object(P.A,'call',return_value=''),patch.object(P.A,'cooldown',return_value=[]),patch.object(P.A,'gpu',return_value=[]),patch.object(P.A.subprocess,'Popen',return_value=p),patch.object(Path,'read_text',read),patch.object(P.A.time,'monotonic',side_effect=now):
                    with self.assertRaises((RuntimeError,ValueError)):P.A.run_arm(mode,r,{'requested_outputs':1024,'fixture_sha256':P.FIXTURE_SHA},[1,2],[],out)
                private=out/'private';self.assertEqual((private/(mode+'.stdout')).read_text(),text)
                self.assertEqual(json.loads((private/(mode+'.partial-output-ids.json')).read_text()),[7])
                status=json.loads((private/(mode+'.termination.json')).read_text())
                self.assertFalse(status['native_completed']);self.assertFalse(status['done_seen_by_parser'])
                self.assertEqual(status['error_type'],error_type);self.assertEqual(status['parsed_token_count'],1)
                self.assertTrue(status['reader_finished']);self.assertIsNone(status['reader_error'])
                self.assertEqual(status['native_cleanup_error'] is not None,mode=='cleanup')
                manifest=json.loads((out/(mode+'.manifest.json')).read_text())
                self.assertTrue(manifest['private_termination_forensics']['raw_stdout']['complete'])
                for path in private.iterdir():self.assertEqual(path.stat().st_mode&0o777,0o600)

if __name__=='__main__':unittest.main()
