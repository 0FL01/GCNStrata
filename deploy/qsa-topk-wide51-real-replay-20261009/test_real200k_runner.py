#!/usr/bin/env python3
"""CPU-only generated logs and mocked launches; no actual capture or GPU claims."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, mock_open
import run_real200k_wide51 as R
from test_validate_200k import fixture

def header(device=0, pos=200000, nq=4):
    lo,hi,layer = (0,27,23) if device==0 else (27,48,47)
    return dict(device=device,layer_begin=lo,layer_end=hi,layer=layer,pos0=pos,T=nq,max_blocks=51202,cap=2051)

def output(h, device, batch, samples, latency=0.25):
    return (f"PARITY_PASS production=reference=captured; padding/end_canaries=PASS\n"
        f"META captured_device={h['device']} replay_device={device} layer={h['layer']} stage=[{h['layer_begin']},{h['layer_end']}) pos0={h['pos0']} nq={h['T']} capacity_blocks={h['max_blocks']} cap={h['cap']} batch={batch} samples={samples} device=AMD Radeon Instinct MI50\n"
        'SAMPLES_MS_PER_CALL' + ''.join(f' {latency:.9f}' for _ in range(samples)) + '\n' +
        f'RESULT median_ms_per_call={latency:.9f} min={latency:.9f} max={latency:.9f} input_bytes_unchanged=PASS graph_ids=PASS\n' + R.SCOPE+'\n')

def args(root):
    return types.SimpleNamespace(build=root/'build',resource_build=root/'resource-build.json',manual_review=root/'manual.json',
        resource_gate=root/'resource.json',boundary_summary=root/'boundary.json',capture_summary=root/'capture.json',
        real0=root/'captures/d0.bin',real1=root/'captures/d1.bin',output=root/'out',build_sha='a'*64,
        baseline_sha='b'*64,candidate_sha='c'*64,capture_summary_sha='d'*64,real0_sha='e'*64,real1_sha='f'*64,
        leave_api_stopped=True,execute=True)

class Tests(unittest.TestCase):
    def test_exact_header_parser_rows_two_through_six(self):
        for device in (0,1):
            for nq in range(2,7):
                for batch,samples in ((1,1),(64,7)):
                    h=header(device,200258 if nq==6 else 200000,nq)
                    r=R.parse_result(output(h,device,batch,samples),'',h,device,batch,samples)
                    self.assertTrue(r['all_parity_passed'])
                    self.assertFalse(r['route_observed'])

    def test_complete_gate_and_capture_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=args(root);a.build.mkdir();a.real0.parent.mkdir()
            a.real0.write_bytes(fixture(0,T=2));a.real1.write_bytes(fixture(1,T=2))
            resource_build={'compiled':True,'resource_isa_gate_passed':True,'support_archive_used':False,
                'files':{**{'qsa_select.'+v+'.cu':h for v,h in R.SOURCE.items()},**{'qsa_select.'+v+'.o':h for v,h in R.OBJECT.items()}}}
            manual={'decision':'PASS','blockers':[],'bound_artifact_sha256':{'build.receipt.json':R.RESOURCE_BUILD,'resource-gate.json':R.RESOURCE}}
            boundary={'qualification_completed':True,'cleanup':{'passed':True},'manual_sha256':R.MANUAL,'resource_sha256':R.RESOURCE,
                'processes':[{'cases':1212 if i==0 else 0,'status':'passed','parity_passed':True,'exit_code':0} for i in range(42)]}
            build={'passed':True,'exit_code':0,'gpu_executed':False,'support_archive_used':False,'resource_build_sha256':R.RESOURCE_BUILD,
                'files':{'qsa_topk_fixture_replay.cpp':R.CONSUMER,'qsa_select.fit17.o':R.OBJECT['fit17'],'qsa_select.wide51.o':R.OBJECT['wide51'],
                         'replay.fit17':a.baseline_sha,'replay.wide51':a.candidate_sha}}
            capture={'capture_passed':True,'decision':'PASS_CAPTURE_ONLY','owned_containers_cleaned':True,'all_containers_stopped':True,
                'cleanup_failed_arms':[],'counter_comparison':{'all_equal':True},'control_binary_sha256':R.V.BASELINE_SHA,'fixture_sha256':R.V.FIXTURE_SHA,
                'prompt_tokens':200000,'requested_outputs':256,'all_256_reference_ids_equal_per_arm':True,
                'capture_files':[{'filename':p.name,'sha256':R.S.sha(p),'bytes':p.stat().st_size} for p in [a.real0,a.real1]]}
            for path,doc in [(a.resource_build,resource_build),(a.manual_review,manual),(a.resource_gate,{'passed':True,'failures':[]}),
                    (a.boundary_summary,boundary),(a.build/'build.receipt.json',build),(a.capture_summary,capture)]:
                path.write_text(json.dumps(doc))
            # Only the hash gate is bypassed for CPU test inputs; actual strict validator/analyzer execute.
            with patch.object(R,'rehash'):
                self.assertTrue(R.gate(a)['passed'])
                for key,value in [('capture_passed',False),('decision','PAUSE_COUNTER_DIVERGENCE'),('owned_containers_cleaned',False),
                    ('all_containers_stopped',False),('cleanup_failed_arms',['diagnostic']),('control_binary_sha256','0'*64),
                    ('prompt_tokens',65536),('all_256_reference_ids_equal_per_arm',False),('counter_comparison',{'all_equal':False})]:
                    bad=copy.deepcopy(capture);bad[key]=value;a.capture_summary.write_text(json.dumps(bad))
                    with self.assertRaises(RuntimeError):R.gate(a)
                a.capture_summary.write_text(json.dumps(capture))
                swapped=copy.copy(a);swapped.real0,swapped.real1=a.real1,a.real0
                with self.assertRaises(RuntimeError):R.gate(swapped)
                for key,value in [('support_archive_used',True),('passed',False),('exit_code',7),('gpu_executed',True),('resource_build_sha256','0'*64)]:
                    bad=copy.deepcopy(build);bad[key]=value;(a.build/'build.receipt.json').write_text(json.dumps(bad))
                    with self.assertRaises(RuntimeError):R.gate(a)

    def test_cpu_exact_id_gate_rejects_valid_but_wrong_ids(self):
        import struct
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'d0.bin';b=root/'d1.bin'
            wrong=bytearray(fixture(0,T=2));offset=160+2*16+16+50001*4
            struct.pack_into('<2051i',wrong,offset,*range(1,2052))
            a.write_bytes(wrong);b.write_bytes(fixture(1,T=2))
            R.V.read_200k(a)
            with self.assertRaises(AssertionError):R.V.validate_pair([a,b])

    def test_parser_strict_missing_duplicate_corrupt_lines(self):
        h=header(); good=output(h,0,64,7)
        bad=[good.replace('pos0=200000','pos0=65536'),good.replace('nq=4','nq=6'),
             good.replace('stage=[0,27)','stage=[27,48)'),good.replace('replay_device=0','replay_device=1'),
             good.replace('padding/end_canaries=PASS','padding/end_canaries=FAIL'),
             good.replace('graph_ids=PASS','graph_ids=FAIL'),good.replace('input_bytes_unchanged=PASS','input_bytes_unchanged=FAIL'),
             good+good.splitlines()[0]+'\n',good.split('\n',1)[1],good.replace(R.SCOPE,'SCOPE wrong'),
             good.replace('SAMPLES_MS_PER_CALL 0.250000000','SAMPLES_MS_PER_CALL nan',1),
             good.replace('SAMPLES_MS_PER_CALL 0.250000000','SAMPLES_MS_PER_CALL -1',1),
             good.replace('SAMPLES_MS_PER_CALL 0.250000000','SAMPLES_MS_PER_CALL',1),
             good.replace('min=0.250000000','min=0.2'),good.replace('max=0.250000000','max=0.3')]
        for text in bad:
            with self.assertRaises((RuntimeError,ValueError)):
                R.parse_result(text,'',h,0,64,7)
        for err in ['FAIL: no\n','ERROR bad\n','strata gfx906 topk fit17: old topology\n']:
            with self.assertRaises(RuntimeError):
                R.parse_result(good,err,h,0,64,7)

    def test_exact_schedule_and_no_flag_sweep(self):
        plans=R.schedule()
        self.assertEqual(len(plans),12)
        self.assertEqual([(r['device'],r['variant']) for r in plans[:4]],[(0,'fit17'),(0,'wide51'),(1,'fit17'),(1,'wide51')])
        self.assertTrue(all((r['batch'],r['samples'],r['phase'])==(1,1,'correctness') for r in plans[:4]))
        self.assertTrue(all((r['batch'],r['samples'],r['phase'])==(64,7,'abba') for r in plans[4:]))
        self.assertEqual([(r['device'],r['label']) for r in plans[4:]],[(d,l) for d in (0,1) for l in ('A1','B1','B2','A2')])

    def test_commands_exact_flags_no_profile_and_own_device(self):
        a=args(Path('/fake'))
        for plan in R.schedule():
            cmd=R.command(a,plan,'owned')
            tail=cmd[cmd.index(R.S.IMAGE)+1:]
            for key in R.CONTROLLED:
                at=tail.index(key)
                self.assertEqual(tail[at-1],'-u')
            for wanted in ['HIP_VISIBLE_DEVICES=0,1','STRATA_GFX906_TOPK_REG=2','STRATA_GFX906_TOPK_FIT17=1','STRATA_TOPK_TRACE=1']:
                self.assertEqual(tail.count(wanted),1)
            self.assertEqual(tail.count('STRATA_GFX906_TOPK_WIDE51=1'),int(plan['variant']=='wide51'))
            self.assertEqual(cmd[-5:],['/work/replay.'+plan['variant'],f"/fixture/d{plan['device']}.bin",str(plan['device']),str(plan['batch']),str(plan['samples'])])
            self.assertFalse(any(s in cmd for s in ['start','--privileged']))

    def run_mocked(self, fail_at=None, corrupt=False, cleanup_fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=args(root);calls=[]
            analyzed={'passed':True,'files':[{'header':header(d)} for d in (0,1)]}
            power=[{'pci':p,'mode':'high','power1_cap':190000000} for p in ['0000:05:00.0','0000:08:00.0']]
            def launch(cmd,name,ds,out,tag,timeout):
                plan=R.schedule()[len(calls)];calls.append(plan)
                self.assertEqual(timeout,180)
                self.assertEqual(cmd,R.command(a,plan,name))
                self.assertEqual(exclusive.call_count,1+2*len(calls))
                if plan['phase']=='abba':
                    self.assertTrue(json.loads((out/'summary.json').read_text())['all_correctness_passed'])
                text=output(header(plan['device']),plan['device'],plan['batch'],plan['samples'],0.3 if plan['variant']=='wide51' else 0.25)
                if corrupt and len(calls)==fail_at:text=text.replace('graph_ids=PASS','graph_ids=FAIL')
                (out/(tag+'.stdout')).write_text(text);(out/(tag+'.stderr')).write_text('')
                return {'exit_code':7 if len(calls)==fail_at and not corrupt else 0}
            with patch.object(R.S,'R',root),patch.object(R,'gate',return_value=analyzed),patch.object(R,'rehash'),\
                 patch.object(R,'exclusive') as exclusive,patch.object(R.S,'devices',return_value=['fake0','fake1']),\
                 patch.object(R.S,'gpu',return_value=power),patch.object(R.S,'cooldown',return_value=power) as cooldown,\
                 patch.object(R.S,'cleanup',side_effect=RuntimeError('cleanup failed') if cleanup_fail else None) as cleanup,\
                 patch.object(R.P,'launch',side_effect=launch),contextlib.redirect_stdout(io.StringIO()):
                if fail_at or cleanup_fail:
                    with self.assertRaises(RuntimeError):R.execute(a)
                else:R.execute(a)
                self.assertEqual(len(calls),fail_at or 12)
                self.assertEqual(cooldown.call_count,len(calls))
                self.assertEqual(cleanup.call_count,1 if cleanup_fail else len(calls))
            summary=json.loads((a.output/'summary.json').read_text())
            self.assertEqual(summary['qualification_completed'],not(fail_at or cleanup_fail))
            self.assertFalse(summary['api_start_stop_actions']);self.assertFalse(summary['model_pp_tg_gain_claim'])
            for row in summary['processes']:
                if row['phase']=='correctness' and row['status']=='passed':
                    self.assertFalse(row['timing_claim']);self.assertNotIn('samples_ms',row);self.assertNotIn('median_ms_per_call',row)
            if not fail_at and not cleanup_fail:
                self.assertEqual(len(summary['components']),2)
                self.assertTrue(all(c['latency_reduction_pct']<0 for c in summary['components']))
            self.assertTrue((a.output/'RECEIPTS.sha256.json').exists())

    def test_bounded_run_even_when_candidate_is_slower(self):self.run_mocked()
    def test_failure_stops_before_any_abba(self):self.run_mocked(fail_at=4)
    def test_missing_graph_parity_stops_immediately(self):self.run_mocked(fail_at=1,corrupt=True)
    def test_abba_failure_never_retries(self):self.run_mocked(fail_at=6)
    def test_cleanup_failure_invalidates_completion(self):self.run_mocked(cleanup_fail=True)

    def test_preflight_fails_before_docker_gpu_or_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=args(root)
            with patch.object(R.S,'R',root),patch.object(R,'gate',side_effect=RuntimeError('bad capture')),\
                 patch.object(R,'exclusive') as exclusive,patch.object(R.S,'devices') as devices,patch.object(R.P,'launch') as launch:
                with self.assertRaises(RuntimeError):R.execute(a)
                exclusive.assert_not_called();devices.assert_not_called();launch.assert_not_called()
                self.assertFalse(a.output.exists())

    def test_binding_rejects_symlink_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Path(directory)/'data';f.write_bytes(b'CPU only')
            sha=R.S.sha(f);R.bound(f,sha)
            with self.assertRaises(RuntimeError):R.bound(f,'0'*64)
            s=Path(directory)/'link';s.symlink_to(f)
            with self.assertRaises(RuntimeError):R.bound(s,sha)

    def test_default_is_noop_and_single_global_lock(self):
        a=args(Path('/fake'));argv=[]
        for key in ['build','resource_build','manual_review','resource_gate','boundary_summary','capture_summary','real0','real1','output',
                    'build_sha','baseline_sha','candidate_sha','capture_summary_sha','real0_sha','real1_sha']:
            argv+=['--'+key.replace('_','-'),str(getattr(a,key))]
        argv+=['--leave-api-stopped']
        with patch.object(R,'execute') as execute,patch.object(R.fcntl,'flock') as flock,patch.object(R.os,'umask'),\
             patch('builtins.open',mock_open()) as opened,contextlib.redirect_stdout(io.StringIO()):
            R.main(argv);execute.assert_not_called();opened.assert_not_called();flock.assert_not_called()
            R.main(argv+['--execute']);execute.assert_called_once();opened.assert_called_once_with(R.LOCK,'w')
            self.assertEqual(flock.call_args.args[1],R.fcntl.LOCK_EX|R.fcntl.LOCK_NB)

if __name__=='__main__':unittest.main(verbosity=2)
