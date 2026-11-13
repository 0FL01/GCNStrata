import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_cost as R
from test_runner import fixture

class Tests(unittest.TestCase):
 def test_parser_failure_preserves_raw_and_one_use_markers(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);campaign=root/'logs/perf-campaign-20261009';campaign.mkdir(parents=True)
   binary=root/'observer';binary.write_bytes(b'cost observer');baseline=root/'baseline';baseline.write_bytes(b'base');h=R.N.sha(binary)
   old=campaign/('hc-counts-'+h+'.once');old.write_text('prior counts remains\n')
   a=SimpleNamespace(leave_api_stopped=True,binary=binary,baseline=baseline,binary_sha=h,output=root/'out')
   raw=fixture().replace('"ns":1000000','"ns":-1',1);stdout='complete raw stdout\n'
   def arm(tag,r,s,ids,ds,out):
    private=out/'private';private.mkdir();(private/(tag+'.stderr')).write_text(raw);(private/(tag+'.stdout')).write_text(stdout)
    (private/(tag+'.termination.json')).write_text(json.dumps(dict(native_exit=0,native_completed=True,reader_finished=True,reader_error=None,native_cleanup_error=None)))
    return dict(passed=True)
   with patch.object(R.N,'R',root),patch.object(R,'inputs',return_value=(dict(environment={},native_args=[],prefix_sha256='fixture'),[1]*12288,{})),patch.object(R.N,'call',side_effect=lambda cmd: '[]' if 'inspect' in cmd else ''),patch.object(R.N,'devices',return_value=[]),patch.object(R.N,'gpu',return_value=[]),patch.object(R.A,'run_arm',side_effect=arm) as run,patch.object(R.S,'cleanup'):
    with self.assertRaisesRegex(ValueError,'invalid/overflow raw duration'):R.execute(a)
    self.assertEqual(run.call_count,1)
    self.assertEqual((a.output/'private/hc-cost.stderr').read_text(),raw);self.assertEqual((a.output/'private/hc-cost.stdout').read_text(),stdout)
    summary=json.loads((a.output/'summary.json').read_text());self.assertFalse(summary['completed']);self.assertTrue(summary['cleanup']['passed']);self.assertEqual(summary['error_type'],'ValueError')
    self.assertEqual(old.read_text(),'prior counts remains\n');self.assertTrue((campaign/('hc-cost-'+h+'.once')).exists())
    a.output=root/'retry'
    with self.assertRaises(FileExistsError):R.execute(a)
    self.assertEqual(run.call_count,1)
 def test_screen_boundary_and_first_hits(self):
  rows=[json.loads(x) for x in fixture().splitlines()]
  for row in rows:
   if row['type']=='hc_observer_weight':
    for o in row['guard_observations']:o['ns']=0
  target=next(r for r in rows if r['type']=='hc_observer_weight' and r['tensor_ordinal']==1)
  def parse():return R.P.parse('\n'.join(json.dumps(r,separators=(',',':')) for r in rows))
  target['guard_observations'][0]['ns']=2**62;target['guard_observations'][2]['ns']=1003739603
  x=parse();self.assertEqual(x['gate'],'PARK_WITHOUT_CACHE_PATCH_OR_MODEL_QUALIFICATION');self.assertEqual(x['conditional_64K_screen_sum_ns'],1003739603)
  target['guard_observations'][2]['ns']+=1;x=parse();self.assertEqual(x['gate'],'ATTRIBUTION_AND_LIFETIME_INVESTIGATION_ONLY');self.assertEqual([s['conditional_64K_repeat_count'] for s in x['guard_wall_stages']],[165,135]);self.assertFalse(x['performance_claim'])
 def test_base_wrapper_and_gate_unchanged(self):
  base=R.HERE.parent/'c183-hc-counts-runner-v1-20261009'
  for name in ['native_arm.py','native_support.py','qsa_probe_safety.py','device_compare.py','baseline.manifest.json','prompt-12288.ids.json','parse_counts.py']:
   self.assertEqual((base/name).read_bytes(),(R.HERE/name).read_bytes(),name)
if __name__=='__main__':unittest.main()
