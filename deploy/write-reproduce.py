import json
from pathlib import Path
r=Path('/home/radneon/strata')
for n,s in json.loads((r/'deploy/reproduce-files.json').read_text()).items():
 p=r/n;assert not p.exists();p.write_text(s);p.chmod(0o755)

