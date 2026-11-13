from pathlib import Path
import subprocess,json,time,urllib.request,re,hashlib,datetime
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261009/qsa-selection-replay'
def gpu():
 out=[]
 for d in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
  if not (d/'mem_info_vram_total').exists():continue
  row={'pci':d.resolve().name,'mode':(d/'power_dpm_force_performance_level').read_text().strip()};assert row['mode']=='high'
  for h in (d/'hwmon').glob('hwmon*'):
   if (h/'power1_cap').exists():assert int((h/'power1_cap').read_text())==190000000
   for i in [1,2,3]:
    p=h/f'temp{i}_input';c=h/f'temp{i}_crit'
    if p.exists():
     row[f'temp{i}']=int(p.read_text())
     if c.exists():assert row[f'temp{i}']<int(c.read_text())-10000
  out.append(row)
 assert len(out)==2;return out
results=[];restore={};name='strata-topk-fixture-replay'
try:
 assert json.loads((O/'build.json').read_text())['exit_code']==0
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 for dev,fixture in [(0,'window.dev0.layers0-27.bin'),(1,'window.dev1.layers27-48.bin')]:
  cmd=['docker','run','--rm','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','-e','HIP_VISIBLE_DEVICES=0,1','-e','STRATA_GFX906_TOPK_REG=2','-e','STRATA_TOPK_TRACE=1','-v',str(O)+':/work:ro','--entrypoint','/work/qsa_topk_fixture_replay','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','/work/'+fixture,str(dev),'64','7']
  start=time.monotonic();sensors=[]
  with (O/f'device{dev}.stdout').open('x') as out,(O/f'device{dev}.stderr').open('x') as err:
   p=subprocess.Popen(cmd,stdout=out,stderr=err)
   while p.poll() is None:
    sensors.append({'elapsed_s':time.monotonic()-start,'gpu':gpu()})
    if time.monotonic()-start>180:raise TimeoutError('replay')
    time.sleep(1)
  stdout=(O/f'device{dev}.stdout').read_text();stderr=(O/f'device{dev}.stderr').read_text()
  assert p.returncode==0,(dev,p.returncode,stderr)
  assert 'PARITY_PASS production=reference=captured' in stdout and 'input_bytes_unchanged=PASS graph_ids=PASS' in stdout
  m=re.search(r'RESULT median_ms_per_call=([0-9.]+) min=([0-9.]+) max=([0-9.]+)',stdout);assert m
  raw=re.search(r'SAMPLES_MS_PER_CALL([^\n]+)',stdout);assert raw
  row={'device':dev,'fixture':fixture,'fixture_sha256':hashlib.sha256((O/fixture).read_bytes()).hexdigest(),'median_ms_per_call':float(m[1]),'min_ms':float(m[2]),'max_ms':float(m[3]),'samples_ms':[float(x) for x in raw[1].split()],'parity_passed':True,'input_bytes_unchanged':True,'graph_ids_passed':True,'route_trace':stderr,'sensors':sensors}
  results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='sensors'}),flush=True)
finally:
 if subprocess.check_output(['docker','ps','-q','--filter','name=^/'+name+'$']).strip():subprocess.run(['docker','stop','--time','30',name],check=False)
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True,'health':h};break
  except (OSError,ValueError):pass
  time.sleep(5)
 (O/'summary.json').write_text(json.dumps({'results':results,'production_restore':restore,'scope':'unchanged production top-k replay; two captured layers only; no candidate or whole-model gain','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));assert restore.get('passed')
print('QSA TOPK FIXTURE REPLAY COMPLETE',flush=True)
