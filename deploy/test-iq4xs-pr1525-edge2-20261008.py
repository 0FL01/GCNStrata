from pathlib import Path
import subprocess as s,json,time,urllib.request,hashlib,re
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/iq4xs-flat-dequant/pr1525-edge';D=O/'tests2';D.mkdir(exist_ok=False)
assert all(x['exit_code']==0 for x in json.loads((O/'build2.json').read_text()))
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
summary={'runs':[]}
def run(tag,variant,probe,args,env):
 binary=O/(variant+'-'+probe)
 cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-iq4xs-edge-test','--entrypoint','/src/'+str(binary.relative_to(R))]
 for k,v in env.items():cmd+=['-e',k+'='+str(v)]
 cmd+=['-v',str(R)+':/src',image]+args
 p=s.run(cmd,stdout=s.PIPE,stderr=s.STDOUT,text=True,timeout=300)
 (D/(tag+'.log')).write_text(p.stdout)
 assert p.returncode==0,p.stdout[-4000:]
 b=(D/(tag+'.bin')).read_bytes()
 row={'tag':tag,'variant':variant,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 m=re.search(r'median_ms=([0-9.]+)',p.stdout)
 if m:row['median_ms']=float(m.group(1))
 summary['runs'].append(row);print(json.dumps(row),flush=True)
 return b
try:
 s.run(['docker','stop','strata-api'],check=True,timeout=120)
 assert not s.check_output(['docker','ps','-q'],text=True).strip()
 ref=None
 for variant in ['baseline','candidate']:
  tag='edge-'+variant
  b=run(tag,variant,'gfx906_iq4xs_edge_probe',['--device','0','--device','1','--device','0','--dump','/src/'+str((D/(tag+'.bin')).relative_to(R))],{'HIP_VISIBLE_DEVICES':'0,1'})
  if ref is None:ref=b
  assert b==ref,'edge full dump mismatch'
 for gpu in [0,1]:
  for shape in ['qkv','output']:
   for pad in [False,True]:
    ref=None
    for variant in ['baseline','candidate']:
     tag='gpu%d-%s-pad%d-%s'%(gpu,shape,pad,variant)
     args=['--device','0','--shape',shape,'--warmups','5','--reps','30']+(['--pad'] if pad else [])
     b=run(tag,variant,'gfx906_iq4xs_dequant_probe',args,{'HIP_VISIBLE_DEVICES':gpu,'STRATA_DEQUANT_DUMP':'/src/'+str((D/(tag+'.bin')).relative_to(R))})
     if ref is None:ref=b
     assert b==ref,'dense full dump mismatch'
 summary['passed']=True
except Exception as e:
 summary['error']=repr(e);raise
finally:
 s.run(['docker','start','strata-api'],check=True,timeout=120)
 summary['api_restored']=False
 for _ in range(120):
  try:
   h=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5).read())
   if h.get('loaded') and h.get('images') and h.get('api_key'):
    summary['api_restored']=True;summary['health']=h;break
  except Exception:pass
  time.sleep(5)
 (D/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 assert summary['api_restored'],'API restore failed'
