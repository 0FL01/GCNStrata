from pathlib import Path
import subprocess,json,time,datetime,urllib.request
r=Path('/home/radneon/strata');events=[]
def stamp(kind,**kw):events.append({'kind':kind,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
def run(argv,log):
 with (r/'logs'/log).open('w') as f:subprocess.run(argv,cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
with (r/'logs/initial-live-server.log').open('w') as f:subprocess.run(['docker','logs','--timestamps','strata-api'],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 run(['./stop.sh'],'first-stop.log')
 assert subprocess.run(['docker','inspect','strata-api'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
 assert not subprocess.check_output(['ss','-ltnH','sport = :8080']).strip()
 stamp('first_stop',container_absent=True,port_closed=True,autostart_removed=True)
 run(['./start.sh'],'restart-start.log')
 deadline=time.monotonic()+600
 while True:
  try:
   h=json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5))
   if h.get('loaded') and h.get('api_key') and h.get('max_context')==131072:break
  except OSError:pass
  if time.monotonic()>deadline:raise RuntimeError('restart readiness timed out')
  time.sleep(3)
 policy=subprocess.check_output(['docker','inspect','strata-api','--format','{{.HostConfig.RestartPolicy.Name}}'],text=True).strip()
 assert policy=='unless-stopped'
 stamp('restart_ready',health=h,restart_policy=policy)
 run(['docker','exec','-e','STRATA_TEST_LABEL=restart','strata-api','python3','/src/deploy/test-live-api.py'],'restart-api-test.log')
 stamp('restart_arithmetic_and_auth_smoke',passed=True)
finally:
 run(['./stop.sh'],'final-stop.log')
 gone=subprocess.run(['docker','inspect','strata-api'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
 closed=not subprocess.check_output(['ss','-ltnH','sport = :8080']).strip()
 stamp('final_stop',container_absent=gone,port_closed=closed,autostart_removed=gone)
 (r/'logs/lifecycle-verification.json').write_text(json.dumps({'events':events,'final_off':gone and closed},indent=2))
assert gone and closed
print(json.dumps({'passed':True,'events':events}))

