import json
from pathlib import Path
p=Path('/home/radneon/strata/configs/smoke-no-mtp.json');s=json.loads(p.read_text());a=s['args'];a[a.index('--spec')+1]='4';p.write_text(json.dumps(s,indent=2))

