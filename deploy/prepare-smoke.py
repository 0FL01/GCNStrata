import json
from pathlib import Path
r=Path('/home/radneon/strata')
s=json.loads((r/'configs/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M.json').read_text())
a=s['args'];b=[];i=0
while i<len(a):
 if a[i] in ['--mtp','--spec','--spec-min-p']:i+=2
 else:b.append(a[i]);i+=1
s['args']=b+['--spec','0']
(r/'configs/smoke-no-mtp.json').write_text(json.dumps(s,indent=2))
print('Prepared no-MTP loader/API smoke; final configuration remains MTP4')

