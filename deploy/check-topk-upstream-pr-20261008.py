from pathlib import Path
import subprocess,json,time,datetime
R=Path('/home/radneon/strata');W=Path('/home/radneon/strata-pr-topk-20261008');O=R/'logs/topk-upstream-pr-20261008/components-corrected';O.mkdir(exist_ok=False)
assert not subprocess.check_output(['docker','ps','-q']).strip(),'API or another container still active'
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
rows=[]
for gpu in [0,1]:
 tag='g%d-ctest'%gpu
 cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-topk-component','-e','HIP_VISIBLE_DEVICES='+str(gpu),'--entrypoint','ctest','-v',str(W)+':/src',image,'--test-dir','/src/build-906','--output-on-failure','-R','^qsa_topk_(parity|gfx906_guarded)$']
 t=time.monotonic()
 with (O/(tag+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 rows.append({'tag':tag,'argv':cmd,'exit_code':p.returncode,'elapsed_s':time.monotonic()-t})
 (O/'results.json').write_text(json.dumps(rows,indent=2))
 assert p.returncode==0,tag
 cases=[]
 for mode in [0,2]:
  cases.append(('selftest-mode'+str(mode),mode,False,['--selftest']))
 for nq in [1,6,8,9]:
  for context in [4096,24575,24576,24577,65536,200000]:
   cases.append(('nq%d-ctx%d'%(nq,context),2,False,[str(context),str(nq),'100','204800','0']))
 cases.append(('old-override',2,True,['65536','6','100','204800','0']))
 cases.append(('capacity-fallback',2,False,['65536','6','100','524288','0']))
 cases.append(('counted-regression',2,False,['65536','6','100','204800','1']))
 for tag,mode,old,args in cases:
  name='g%d-%s'%(gpu,tag)
  cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--ulimit','memlock=-1:-1','--name','strata-topk-component','-e','HIP_VISIBLE_DEVICES='+str(gpu),'-e','STRATA_GFX906_TOPK_REG='+str(mode),'--entrypoint','/src/build-906/qsa_topk_parity','-v',str(W)+':/src',image]+args
  if old:cmd[cmd.index('--entrypoint'):cmd.index('--entrypoint')]=['-e','STRATA_TOPK_OLD=1']
  t=time.monotonic()
  with (O/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  r={'tag':name,'argv':cmd,'exit_code':p.returncode,'elapsed_s':time.monotonic()-t}
  rows.append(r);(O/'results.json').write_text(json.dumps(rows,indent=2))
  print(json.dumps(r),flush=True)
  assert p.returncode==0,name
(O/'complete.json').write_text(json.dumps({'passed':True,'case_count':len(rows),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
