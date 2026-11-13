from pathlib import Path
import subprocess,json,hashlib,time,urllib.request,re
R=Path('/home/radneon/strata');O=R/'logs/perf-campaign-20261008/qsa-query-swizzle';D=O/'parity';D.mkdir(exist_ok=False)
summary={'runs':[],'original_micro_gate_passed':False,'qualification_exception':'14.79% stable component throughput with no arithmetic/resource cliff; bounded paired64K test justified by projected1.74% GPU timeline savings, not gate pass','model_gate':'same1024 output IDs, >=1% prompt and request latency reduction, no >0.5% TG regression'}
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
try:
 subprocess.run(['docker','stop','--timeout','120','strata-api'],check=True)
 assert not subprocess.check_output(['docker','ps','-q']).strip()
 for gpu in [0,1]:
  for nq,ctx,masked in [(1,4,False),(33,4096,True),(32,65536,False)]:
   ref=None;input_ref=None
   for mode in [0,1]:
    tag='gpu%d-nq%d-ctx%d-mask%d-mode%d'%(gpu,nq,ctx,masked,mode)
    output=D/(tag+'.bin')
    cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-qsa-query-parity','--entrypoint','/src/logs/perf-campaign-20261008/qsa-query-swizzle/gfx906_qsa_query_swizzle_probe','-e','HIP_VISIBLE_DEVICES='+str(gpu),'-e','STRATA_ATTN_LANECELL=0','-e','STRATA_GFX906_ATTN_QUERY_SWIZZLE='+str(mode),'-e','STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE=1','-v',str(R)+':/src',image,str(nq),str(ctx),'5','/src/'+str(output.relative_to(R))]+(['--masked-pages'] if masked else [])
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
    (D/(tag+'.log')).write_text(p.stdout)
    assert p.returncode==0,p.stdout
    assert 'active=%d '%mode in p.stdout
    b=output.read_bytes();assert len(b)==nq*24*256*4
    ih=re.search(r'INPUT_FNV1A64=([0-9a-f]+)',p.stdout).group(1)
    if ref is None:ref=b;input_ref=ih
    assert ref==b and input_ref==ih,'output/input changed'
    if masked:assert int(re.search(r'masked_selected=(\d+)',p.stdout).group(1))>0
    row={'tag':tag,'gpu':gpu,'mode':mode,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'input_hash':ih}
    summary['runs'].append(row);print(json.dumps(row),flush=True)
 summary['passed']=True
finally:
 subprocess.run(['docker','start','strata-api'],check=True)
 summary['api_restored']=False
 for _ in range(120):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as f:h=json.load(f)
   if h.get('loaded') and h.get('images') and h.get('api_key'):summary['api_restored']=True;break
  except (OSError,ValueError):pass
  time.sleep(5)
 (D/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 assert summary['api_restored']
