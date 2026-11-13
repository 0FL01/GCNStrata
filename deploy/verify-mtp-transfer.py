from pathlib import Path
import json,datetime
r=Path('/home/radneon/strata');p=r/'runtime/mtp'
m=json.loads((r/'deploy/mtp-transfer-manifest.json').read_text())
for name,n in m['files'].items():assert (p/name).is_file() and (p/name).stat().st_size==n,(name,n)
assert m['files']['experts.bin']==512*1382400
end=0
for line in (p/'dense.txt').read_text().splitlines():
 name,kind,rows,cols,offset,size=line.split();rows,cols,offset,size=map(int,[rows,cols,offset,size])
 assert rows>0 and cols>0 and offset==end
 expected=rows*cols*(2 if kind=='bf16' else 4) if kind in ['bf16','f32'] else rows*(cols//32)*34
 assert kind in ['bf16','f32','q8_0'] and (kind!='q8_0' or cols%32==0) and size==expected
 end=offset+size
assert end==m['files']['dense.bin']
m['validated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(p/'.ready').write_text(json.dumps(m,indent=2))
print(json.dumps({'validated':True,'files':m['files'],'dense_extent':end}))

