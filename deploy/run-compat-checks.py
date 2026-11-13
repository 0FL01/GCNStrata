from pathlib import Path
import subprocess,json,time
R=Path('/home/radneon/strata');O=R/'logs/quant-comparison-64k-20261006';O.mkdir(exist_ok=True)
base=['docker','run','--rm','--name','strata-quant-compat','--entrypoint','python3','-v',str(R)+':/src','-v','/home/radneon/models-nvme:/models:ro','-w','/src','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646']
tests=[('keep1-stock-pack',['tools/iq_pack.py','--gguf','/models/qwen38-keep1-Q4_0.gguf','--out','/src/logs/quant-comparison-64k-20261006/compat-keep1']),('shared-q8-stock-mtp',['tools/mtp_rt.py','--gguf','/models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf','--out','/src/logs/quant-comparison-64k-20261006/compat-mtp'])]
results=[]
for tag,args in tests:
 start=time.monotonic()
 with (O/(tag+'.log')).open('x') as log:p=subprocess.run(base+args,stdout=log,stderr=subprocess.STDOUT)
 result={'tag':tag,'native_exit':p.returncode,'elapsed_s':time.monotonic()-start,'argv':base+args,'tail':(O/(tag+'.log')).read_text()[-2000:]};results.append(result);print(json.dumps(result),flush=True)
(O/'compatibility-results.json').write_text(json.dumps(results,indent=2))
