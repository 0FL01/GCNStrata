import unittest,json,re
from pathlib import Path
import run_public_fit17 as W
HERE=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_actual_capture_and_trace_modes(self):
  text=(HERE/'test_logs/real-g0-A1.stdout').read_text()
  trace='strata gfx906 topk fit17: enabled=1 topology=guarded-ref256/reg17/reg66 max_blocks=17408 nq=4 capacity_blocks=51202 cap=2051 (device steps select the per-window arm)\n'
  self.assertTrue(W.parse_capture(text,trace,0,True,64,7)['all_parity_passed'])
  self.assertTrue(W.parse_capture(text,'',0,True,64,7,False)['all_parity_passed'])
  self.assertTrue(W.parse_capture(text,'',0,False,64,7)['all_parity_passed'])
  for err in ['',trace+trace,trace.replace('17408','17409')]:
   with self.assertRaises(RuntimeError):W.parse_capture(text,err,0,True,64,7)
  with self.assertRaises(RuntimeError):W.parse_capture(text,trace,0,True,64,7,False)
 def test_transition_examples(self):
  # Explicit synthetic stdout from fixture metadata; not GPU receipts.
  mf=json.loads((HERE/'test_logs/reg17-transition-manifest.json').read_text())
  for f in mf['fixtures']:
   for mode in [0,1]:
    text=''.join(f"CASE_PASS nq={f['nq']} case={c['name']} last_nkv={c['nkv'][-1]} max_active_blocks={c['max_active_blocks']} candidate_intended_route={c['expected_route']} IDs=graph=reference=CPU padding=end_canaries=inputs_unchanged=PASS\n" for c in f['cases'])
    text+=f"TRANSITION_PASS nq={f['nq']} cases={len(f['cases'])} graph_instantiations=1 constant_addresses=1 constant_capacities=1 fit17={mode} timing_claim=0\n"
    err=f"strata gfx906 topk fit17: enabled=1 topology=guarded-ref256/reg17/reg66 max_blocks=17408 nq={f['nq']} capacity_blocks=65537 cap=2080 (device steps select the per-window arm)\n" if mode and f['nq']<=8 else ''
    self.assertTrue(W.parse_transition(text,err,0,bool(mode),mode,f)['parity_passed'])
    with self.assertRaises(RuntimeError):W.parse_transition(text.replace('CASE_PASS','BAD',1),err,0,bool(mode),mode,f)
 def test_mocked_full_flow_keeps_gpu1_below30pct(self):
  import tempfile,types,contextlib,io
  from unittest.mock import patch
  mf=json.loads((HERE/'test_logs/reg17-transition-manifest.json').read_text());calls=[]
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);fixtures=root/'fixtures';fixtures.mkdir()
   for f in mf['fixtures']:(fixtures/f['file']).touch()
   for n in ['real0','real1']:(root/n).touch()
   a=types.SimpleNamespace(leave_api_stopped=True,build=root/'build',output=root/'out',fixtures=fixtures,real0=root/'real0',real1=root/'real1',gate_sha='e'*64)
   g={'binaries':{n:'b'*64 for n in W.BINARIES}}
   def fake_sha(path):
    path=Path(path)
    if path.name in ['real0','real1']:return W.P.REAL[int(path.name[-1])]
    for f in mf['fixtures']:
     if path.name==f['file']:return f['sha256']
    return 'b'*64
   def launch(cmd,name,ds,out,tag,timeout):
    calls.append((tag,cmd));device=int(re.search(r'-g(\d)',tag)[1]);binary=next(x for x in cmd if x.startswith('/work/'));enabled=binary.endswith('.fit17') and 'STRATA_GFX906_TOPK_FIT17=1' in cmd
    trace_on='STRATA_GFX906_TOPK_FIT17_TRACE=1' in cmd;err='';text=''
    if 'public-parity' in binary:
     text=('PASS context synthetic\nadversarial=8/8 changed-step-graph=SKIP\n'*18)+'qsa_topk_parity: PASS\n';nq,cap,width=8,33796,2051
    elif 'transition' in binary:
     nq=int(tag.rsplit('-t',1)[1]);f=next(x for x in mf['fixtures'] if x['nq']==nq);mode=int(tag.split('-m')[1].split('-')[0]);cap,width=65537,2080
     text=''.join(f"CASE_PASS nq={nq} case={c['name']} last_nkv={c['nkv'][-1]} max_active_blocks={c['max_active_blocks']} candidate_intended_route={c['expected_route']} IDs=graph=reference=CPU padding=end_canaries=inputs_unchanged=PASS\n" for c in f['cases'])
     text+=f"TRANSITION_PASS nq={nq} cases={len(f['cases'])} graph_instantiations=1 constant_addresses=1 constant_capacities=1 fit17={mode} timing_claim=0\n"
    else:
     nq,cap,width=4,51202,2051;batch,samples=map(int,cmd[-2:]);v=9 if enabled else 10
     text=f"PARITY_PASS production=reference=captured; padding/end_canaries=PASS\nMETA captured_device={device} replay_device={device} layer={23 if device==0 else 47} pos0=65536 nq=4 capacity_blocks=51202 cap=2051 batch={batch} samples={samples}\nSAMPLES_MS_PER_CALL "+' '.join([str(v)]*samples)+f"\nRESULT median_ms_per_call={v} min={v} max={v} input_bytes_unchanged=PASS graph_ids=PASS\n"
    if enabled and trace_on and nq<=8:err=f"strata gfx906 topk fit17: enabled=1 topology=guarded-ref256/reg17/reg66 max_blocks=17408 nq={nq} capacity_blocks={cap} cap={width} (device steps select the per-window arm)\n"
    (out/(tag+'.stdout')).write_text(text);(out/(tag+'.stderr')).write_text(err);return {'mocked':True}
   power=[dict(pci='test',mode='high',power1_cap=190000000)]
   with patch.object(W.S,'R',root),patch.object(W,'gate',return_value=(g,mf)),patch.object(W.S,'sha',side_effect=fake_sha),patch.object(W.S,'call',return_value=''),patch.object(W.S,'devices',return_value=[]),patch.object(W.S,'gpu',return_value=power),patch.object(W.S,'cooldown',return_value=power),patch.object(W.S,'cleanup'),patch.object(W.P,'launch',side_effect=launch),contextlib.redirect_stdout(io.StringIO()):W.execute(a)
   self.assertEqual(len(calls),54);self.assertTrue(all(not tag.startswith('abba') for tag,_ in calls[:46]));self.assertEqual([tag for tag,_ in calls[46:]],['abba-g0-A1','abba-g0-B1','abba-g0-B2','abba-g0-A2','abba-g1-A1','abba-g1-B1','abba-g1-B2','abba-g1-A2'])
   self.assertTrue(all('STRATA_GFX906_TOPK_FIT17_TRACE=0' in cmd for _,cmd in calls[46:]));summary=json.loads((a.output/'summary.json').read_text());self.assertTrue(summary['qualification_completed']);self.assertEqual(len(summary['component']),2);self.assertFalse(summary['component'][0]['original_30pct_discovery_gate_passed']);self.assertTrue(summary['cleanup']['passed'])
 def test_both_gpu_loop_has_no_discovery_gate_break(self):
  import ast,inspect
  tree=ast.parse(inspect.getsource(W.execute))
  # Narrow structural guard: qualification must never break after a sub30% GPU0 measurement.
  self.assertFalse(any(isinstance(n,ast.Break) for n in ast.walk(tree)))
  self.assertIn('trace_on=False',inspect.getsource(W.execute))
if __name__=='__main__':unittest.main(verbosity=2)
