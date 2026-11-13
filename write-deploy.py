from pathlib import Path
import json
r=Path('/home/radneon/strata')
for name,data in json.loads((r/'deploy-files.json').read_text()).items():
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),name;p.write_text(data)
 if name.endswith('.sh'):p.chmod(0o755)
print('Deployment files prepared')

