#!/usr/bin/env python3
"""Permission probe tests only; Docker and device operations are mocked."""
import ast, json, pathlib, subprocess, tempfile, unittest
from unittest.mock import Mock, patch
import access_preflight as a
import run_paired_200k as runner

class AccessTests(unittest.TestCase):
 def setUp(self):
  self.spec=json.loads((pathlib.Path(__file__).parent/'correctness.spec.json').read_text())
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.campaign=pathlib.Path(self.tmp.name)
  self.flags=runner.make_arms(self.spec,'a'*64)[0]['flags']
  self.binaries=['/src/logs/c183/strata','/src/logs/capture/strata']
 def test_exact_identity_paths_and_readonly_binds(self):
  with patch.object(a.os,'geteuid',return_value=1001),patch.object(a.os,'getegid',return_value=1002),patch.object(a.os,'getgroups',return_value=[109,44,109]):
   s=a.permission_spec(self.spec['native_args'],self.binaries)
  self.assertEqual((s['uid'],s['gid'],s['groups']),(1001,1002,[44,109]))
  self.assertEqual(len(s['files']),3);self.assertEqual(len(s['directories']),2)
  cmd=a.command(runner.IMAGE,runner.R,'/private-probe',self.flags,s)
  self.assertEqual(cmd[cmd.index('--user')+1],'1001:1002')
  self.assertEqual([cmd[i+1] for i,x in enumerate(cmd) if x=='--group-add'],['video','44','109'])
  volumes=[cmd[i+1] for i,x in enumerate(cmd) if x=='-v']
  self.assertEqual([x for x in volumes if x.endswith(':rw')],['/private-probe:/captures:rw'])
  self.assertIn(str(runner.R)+':/src:ro',volumes)
  self.assertEqual(cmd[cmd.index('--entrypoint')+1],'python3')
  self.assertNotIn('--serve',cmd)
  self.assertEqual(cmd[-1],json.dumps(s,separators=(',',':')))
 def test_probe_no_hip_no_model_reads(self):
  tree=ast.parse(a.PROBE)
  imports=[n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)]
  self.assertFalse(any(x.startswith(('hip','torch','ctypes')) for x in imports))
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
  self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('read','read_bytes','read_text','CDLL') for n in calls))
  self.assertIn("os.access(p,os.R_OK|os.W_OK)",a.PROBE)
  self.assertIn("gpu_nodes_opened':False",a.PROBE)
 def invoke(self,run_result=None,run_error=None,active=False):
  fake=run_result or subprocess.CompletedProcess([],0,json.dumps({'passed':True,'hip_initialized':False}), '')
  calls=[]
  def run(cmd,**kw):
   calls.append(cmd)
   if '--entrypoint' in cmd:
    if run_error:raise run_error
    return fake
   return subprocess.CompletedProcess(cmd,0,'','')
  outputs=[b'owned\n',b''] if active else [b'',b'']
  with patch.object(a.subprocess,'run',side_effect=run),patch.object(a.subprocess,'check_output',side_effect=outputs):
   result=a.run(runner.IMAGE,runner.R,self.campaign,self.spec['native_args'],self.binaries,self.flags)
  return result,calls
 def test_success_is_one_permission_process_no_native(self):
  result,calls=self.invoke();self.assertTrue(result['passed']);self.assertEqual(len(calls),1)
  self.assertEqual(list(self.campaign.iterdir()),[])
 def test_access_failure_stops_before_native(self):
  with self.assertRaisesRegex(RuntimeError,'permission-only container preflight failed'):
   self.invoke(subprocess.CompletedProcess([],1,'','PermissionError: model file'))
  self.assertEqual(list(self.campaign.iterdir()),[])
 def test_timeout_cleans_owned_container(self):
  calls=[]
  def run(cmd,**kw):
   calls.append(cmd)
   if '--entrypoint' in cmd:raise subprocess.TimeoutExpired(cmd,120)
   return subprocess.CompletedProcess(cmd,0,'','')
  with patch.object(a.subprocess,'run',side_effect=run),patch.object(a.subprocess,'check_output',side_effect=[b'owned\n',b'']):
   with self.assertRaises(subprocess.TimeoutExpired):a.run(runner.IMAGE,runner.R,self.campaign,self.spec['native_args'],self.binaries,self.flags)
  self.assertEqual(calls[1],['docker','stop','--time','10','strata-qsa-200k-access-preflight'])
 def test_order_before_output_and_native_generation(self):
  source=(pathlib.Path(__file__).parent/'run_paired_200k.py').read_text()
  self.assertLess(source.index('access = preflight.run('),source.index('OUT.mkdir(mode=0o700)'))
  self.assertLess(source.index('access = preflight.run('),source.index('control, diagnostic = paired_requests('))
  self.assertIn("'permission_only_preflight_processes': 1",source)
if __name__=='__main__':unittest.main()
