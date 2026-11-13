import copy
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import run_profile as P

DIAG='''strata prefill timing: 65535 tokens, GPU timeline 1000 ms, wall 1001 ms, host staging 0 ms: hc read 300 (30.0%)
strata decode timing: 333 windows, avg T 3.8, 3.08 tokens/window, 58.00 ms/window = verify 50.00
strata decode GPU stages (ms/window): GDN layers: rec 1.00 | QSA layers: attention 2.00 | total 20.00 ms/window over 333 windows
strata decode GPU stages, stage 1 (ms/window): GDN layers: rec 1.00 | QSA layers: attention 2.00 | total 19.00 ms/window over 333 windows
strata serve: stage 0: 333 windows; per window: wait for the GPU 20.00 ms
strata serve: stage 1: 333 windows; per window: wait for the GPU 19.00 ms
'''

class Tests(unittest.TestCase):
    def test_native_clone_exact_minimal_edit(self):
        original=inspect.getsource(P.N.run_arm)
        old="topology=fit17_topology(stderr);need(bool(topology),'FIT17 host topology trace required');"
        self.assertEqual(original.count(old),1)
        self.assertEqual(inspect.getsource(P.A.run_arm),original.replace(old,'topology=fit17_topology(stderr);'))
        for name,sha in P.PINNED.items(): self.assertEqual(P.N.sha(P.HERE/name),sha)

    def test_done16_correct_optional_null(self):
        done='DONE 1024 65536 101361.7 19026.0 length 648 872 0 582049 600000 0 0 0.0 65536 0'.split()
        d=P.N.native_counter_fields(done)
        self.assertEqual(d['native_done_field_count'],16)
        self.assertIsNone(d['decode_windows']);self.assertIsNone(d['native_suffix_offered'])
        self.assertFalse(d['optional_chain_counters_available'])
        with self.assertRaises(RuntimeError):P.N.native_counter_fields(done[:-1])

    def test_diagnostic_required_stage_coverage(self):
        self.assertEqual(P.diagnostic(DIAG)['diagnostic_decode_windows'],333)
        with self.assertRaises(RuntimeError):P.diagnostic(DIAG.replace('over 333 windows','over 332 windows',1))
        with self.assertRaises(RuntimeError):P.diagnostic(DIAG.replace('strata decode GPU stages, stage 1','missing'))

    def test_compare_allows_adaptive_counter_differences(self):
        a=json.loads((P.HERE/'historical.result.json').read_text());b=copy.deepcopy(a)
        b['accepted']+=1;b['decode_ms']*=1.1
        c=P.compare(a,b)
        self.assertAlmostEqual(c['native_phase_time_change_pct']['decode_ms'],10)
        self.assertFalse(c['performance_gain_claim']);self.assertFalse(c['pure_instrumentation_overhead_established'])
        self.assertEqual(c['counter_values']['decode_windows'],{'control':None,'profile':None})
        b['output_ids'][0]+=1
        with self.assertRaises(RuntimeError):P.compare(a,b)

    def test_exact_input_and_timestamp_gates(self):
        self.assertEqual(P.N.sha(P.HERE/'prompt-65536.ids.json'),P.FIXTURE_SHA)
        with tempfile.TemporaryDirectory() as td:
            a=types.SimpleNamespace(binary=Path(td)/'synthetic-binary',fixture=P.HERE/'prompt-65536.ids.json',timestamp_receipt=Path(td)/'synthetic-unit-test-only.json')
            receipt={'binary_sha256':P.BINARY_SHA,'kernel_archive_sha256':P.ARCHIVE_SHA,
                     'timestamp_source_sha256':P.STAMP_SOURCE_SHA,'timestamp_object_sha256':P.STAMP_OBJECT_SHA,
                     'archive_member_matches_object':True,'producer_isa_verified':True,'producer_ns_per_tick':40,
                     'query_source_sha256':P.N.sha(P.HERE/'query_wallclock.cpp'),
                     'clock_query':{'init_rc':0,'count_rc':0,'device_count':2,'devices':[
                         {'index':i,'attribute_rc':0,'wall_clock_rate_khz':25000,'pci_rc':0,'pci':pci,'props_rc':0,'gcn_arch':'gfx906'}
                         for i,pci in enumerate(['0000:05:00.0','0000:08:00.0'])]}}
            real_sha=P.N.sha
            def sha(path):return P.BINARY_SHA if path==a.binary else real_sha(path)
            def save(d):
                a.timestamp_receipt.write_text(json.dumps(d));a.timestamp_receipt_sha=real_sha(a.timestamp_receipt)
            save(receipt)
            with patch.object(P.N,'sha',side_effect=sha):
                _,ids,_=P.inputs(a);self.assertEqual(len(ids),65536)
                for key,value in [('producer_isa_verified',False),('producer_ns_per_tick',10),('timestamp_object_sha256','0'*64)]:
                    bad=copy.deepcopy(receipt);bad[key]=value;save(bad)
                    with self.assertRaises(RuntimeError):P.inputs(a)
                bad=copy.deepcopy(receipt);bad['clock_query']['devices'][1]['wall_clock_rate_khz']=100000;save(bad)
                with self.assertRaises(RuntimeError):P.inputs(a)

    def mocked_run(self,fail=False):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=types.SimpleNamespace(binary=root/'strata',fixture=root/'fixture',output=root/'output',leave_api_stopped=True,timestamp_receipt_sha='a'*64)
            base=json.loads((P.HERE/'baseline.manifest.json').read_text())
            historical=json.loads((P.HERE/'historical.result.json').read_text());calls=[]
            def run(tag,r,s,ids,ds,out):
                calls.append((tag,r,s,len(ids)))
                if fail:raise RuntimeError('deliberate native failure')
                (out/'private').mkdir(mode=0o700,exist_ok=True)
                (out/'private'/(tag+'.stderr')).write_text(DIAG if tag=='profile' else '')
                ans=copy.deepcopy(historical);ans['tag']=tag;return ans
            rows=[{'pci':'0000:05:00.0','mode':'high','power1_cap':190000000},{'pci':'0000:08:00.0','mode':'high','power1_cap':190000000}]
            def call(args,*unused):return '[]' if args[1:3]==['image','inspect'] else ''
            def sha(path):return P.BINARY_SHA if path==a.binary else P.FIXTURE_SHA
            with patch.object(P.N,'R',root),patch.object(P,'inputs',return_value=(base,[1]*65536,{})),patch.object(P.N,'sha',side_effect=sha),patch.object(P.N,'call',side_effect=call),patch.object(P.N,'devices',return_value=[]),patch.object(P.N,'gpu',return_value=rows),patch.object(P.A,'run_arm',side_effect=run),patch.object(P.S,'cleanup') as cleanup:
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'deliberate native failure'):P.execute(a)
                else:P.execute(a)
                self.assertEqual(cleanup.call_count,2)
            summary=json.loads((a.output/'summary.json').read_text())
            self.assertTrue(summary['cleanup']['passed'])
            return calls,summary,base

    def test_bounded_two_process_recipe(self):
        calls,summary,base=self.mocked_run()
        self.assertEqual([x[0] for x in calls],['control','profile'])
        self.assertEqual(calls[0][1]['flags'],base['flags'])
        self.assertEqual(calls[1][1]['flags'],dict(base['flags'],**{k:'1' for k in P.TIMING}))
        self.assertTrue(all(x[2]['requested_outputs']==1024 and x[3]==65536 for x in calls))
        self.assertTrue(summary['completed']);self.assertFalse(summary['api_actions'])

    def test_failure_stops_without_retry(self):
        calls,summary,_=self.mocked_run(fail=True)
        self.assertEqual(len(calls),1);self.assertFalse(summary['completed'])
        self.assertEqual(summary['error'],'deliberate native failure')

if __name__=='__main__':unittest.main()
