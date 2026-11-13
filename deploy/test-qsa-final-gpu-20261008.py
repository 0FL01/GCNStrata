from pathlib import Path
import json,subprocess,urllib.request,time,hashlib,re
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/qsa-query-swizzle';D=O/'final-gpu';D.mkdir(exist_ok=False)
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
summary={'parity':[],'timing':[]};restore={}
def run(build,gpu,nq,ctx,masked,mode,tag,reps):
 bdir=O/build
 assert json.loads((bdir/'build-final.json').read_text())['exit_code']==0
 out=D/(tag+'.bin')
 cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-qsa-final-gpu','--entrypoint','/src/'+str((bdir/'gfx906_qsa_query_swizzle_probe').relative_to(R)),'-e','HIP_VISIBLE_DEVICES='+str(gpu),'-e','STRATA_ATTN_LANECELL=0','-e','STRATA_GFX906_ATTN_QUERY_SWIZZLE='+str(mode),'-e','STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE=1','-v',str(R)+':/src',image,str(nq),str(ctx),str(reps),'/src/'+str(out.relative_to(R))]+(['--masked-pages'] if masked else [])
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
 (D/(tag+'.log')).write_text(p.stdout)
 assert p.returncode==0,p.stdout
 assert 'active=%d '%mode in p.stdout,p.stdout
 raw=out.read_bytes();assert len(raw)==nq*24*256*4
 old=O/'parity'/('gpu%d-nq%d-ctx%d-mask%d-mode0.bin'%(gpu,nq,ctx,masked))
 assert raw==old.read_bytes(),'final-build output differs from original baseline'
 row={'tag':tag,'build':build,'gpu':gpu,'mode':mode,'nq':nq,'ctx':ctx,'masked':masked,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'median_ms':float(re.search(r'median_ms=([0-9.]+)',p.stdout).group(1))}
 print(json.dumps(row),flush=True);return row
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 for build in ['hardened-build','clean-pr-build']:
  for gpu in [0,1]:
   for nq,ctx,masked in [(1,4,False),(33,4096,True),(32,65536,False)]:
    for mode in [0,1]:
     tag='%s-gpu%d-nq%d-ctx%d-mask%d-mode%d'%(build,gpu,nq,ctx,masked,mode)
     summary['parity'].append(run(build,gpu,nq,ctx,masked,mode,tag,5))
 for tag,mode in [('A1',0),('B1',1),('B2',1),('A2',0)]:
  summary['timing'].append(run('clean-pr-build',0,32,65536,False,mode,'timing-'+tag,50))
 a=sum(x['median_ms'] for x in summary['timing'] if x['mode']==0)/2
 b=sum(x['median_ms'] for x in summary['timing'] if x['mode']==1)/2
 summary['component_throughput_gain_pct']=100*(a/b-1)
 summary['passed']=True
finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):restore={'passed':True};break
  except (OSError,ValueError):pass
  time.sleep(5)
 summary['production_restore']=restore
 (D/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);assert restore.get('passed')
