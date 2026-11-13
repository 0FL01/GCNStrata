"""Strict complete HIP payload equality, plus exhaustive module metadata inventory."""
import hashlib,struct
from pathlib import Path
MAGIC=b'__CLANG_OFFLOAD_BUNDLE__'
def need(ok,label):
 if not ok:raise RuntimeError(label)
def sha(b):return hashlib.sha256(b).hexdigest()
def part(b,off,size):
 need(0<=off<=len(b) and 0<=size<=len(b)-off,'ELF bounds');return b[off:off+size]
def unpack(fmt,b,off):return struct.unpack(fmt,part(b,off,struct.calcsize(fmt)))
def cstr(b,off):
 need(0<=off<len(b),'string offset');end=b.find(b'\0',off);need(end>=0,'unterminated string');return b[off:end].decode('utf-8')
def sections(b,machine):
 h=unpack('<16sHHIQQQIHHHHHH',b,0);need(h[0][:7]==b'\x7fELF\x02\x01\x01' and h[2]==machine,'ELF format/machine');need(h[11]==64 and 0<h[12]<65535 and h[13]<h[12],'ELF section table')
 rows=[unpack('<IIQQQQIIQQ',b,h[6]+i*64) for i in range(h[12])];ns=rows[h[13]];names=part(b,ns[4],ns[5]);result=[]
 for s in rows:result.append(dict(name=cstr(names,s[0]),type=s[1],flags=s[2],address=s[3],offset=s[4],size=s[5],link=s[6],info=s[7],align=s[8],entsize=s[9],data=b'' if s[1]==8 else part(b,s[4],s[5])))
 need(len({s['name'] for s in result})==len(result),'duplicate ELF section names');return h,result
def inventory(b):
 h,secs=sections(b,224);symbols=[]
 for s in secs:
  if s['type'] not in (2,11):continue
  need(s['entsize']==24 and s['size']%24==0 and s['link']<len(secs),'symbol table layout');strings=secs[s['link']]['data']
  for off in range(0,s['size'],24):
   n,info,other,ndx,value,size=unpack('<IBBHQQ',s['data'],off);name=cstr(strings,n)
   if name:symbols.append(dict(name=name,info=info,other=other,section=ndx,value=value,size=size))
 return dict(sha256=sha(b),bytes=len(b),elf_flags=h[7],sections=[{**{k:v for k,v in s.items() if k!='data'},'sha256':sha(s['data'])} for s in secs],symbols=symbols)
def bundles(fat):
 rows=[];pos=0
 while pos<len(fat):
  need(part(fat,pos,len(MAGIC))==MAGIC,'unexpected bundle structure');n=unpack('<Q',fat,pos+len(MAGIC))[0];need(0<n<32,'bundle count');cur=pos+len(MAGIC)+8;entries=[];end=0
  for _ in range(n):
   off,size,length=unpack('<QQQ',fat,cur);cur+=24;need(0<length<4096,'bundle target length');target=part(fat,cur,length).decode('ascii');cur+=length;entries.append((off,size,target));end=max(end,off+size)
  need(len({x[2] for x in entries})==n,'duplicate bundle targets');payloads=[];ranges=[]
  for off,size,target in entries:
   need(off>=cur-pos,'payload overlaps bundle header');blob=part(fat,pos+off,size)
   if size:
    need(target.startswith(('hip-','hipv4-')) and target.endswith('gfx906'),'unexpected device target');ranges.append((off,off+size));payloads.append(dict(target=target,offset=off,metadata=inventory(blob)))
   else:need(target.startswith('host-'),'unexpected empty device payload')
  need(len(payloads)==1,'one gfx906 image per bundle required');need(all(a[1]<=b[0] for a,b in zip(sorted(ranges),sorted(ranges)[1:])),'payload overlap')
  rows.append(dict(offset=pos,bytes=end,entries=[dict(offset=o,bytes=z,target=t) for o,z,t in entries],payloads=payloads))
  finish=pos+end;need(finish<=len(fat),'bundle extent');nextpos=(finish+4095)//4096*4096
  if finish==len(fat):pos=finish
  else:need(nextpos<len(fat) and not any(part(fat,finish,nextpos-finish)),'nonzero or terminal bundle padding');pos=nextpos
 need(len(rows)==55,'expected complete55-module c183 inventory');return rows
def device_view(path):
 raw=Path(path).read_bytes();h,secs=sections(raw,62);names=[s['name'] for s in secs if 'hip' in s['name'].lower() or 'offload' in s['name'].lower()];need(names==['.hip_fatbin','.hipFatBinSegment'],'unexpected device-bearing section inventory');fat=next(s for s in secs if s['name']=='.hip_fatbin');segment=next(s for s in secs if s['name']=='.hipFatBinSegment');mods=bundles(fat['data']);need(segment['size']==24*len(mods),'wrapper count');wrappers=[]
 rela=next(s for s in secs if s['name']=='.rela.dyn');need(rela['entsize']==24 and rela['size']%24==0,'relocation layout');relocs=[unpack('<QQq',rela['data'],i) for i in range(0,rela['size'],24)]
 for i,mod in enumerate(mods):
  magic,version,pointer,reserved=unpack('<IIQQ',segment['data'],24*i);need(magic==0x48495046 and version==1 and reserved==0,'wrapper format');address=segment['address']+24*i+8;matches=[r for r in relocs if r[0]==address];need(len(matches)==1 and matches[0][1]==8,'wrapper relative relocation');need(pointer==0 and matches[0][2]==fat['address']+mod['offset'],'wrapper bundle binding');wrappers.append(dict(magic=magic,version=version,pointer=pointer,reserved=reserved,bundle_offset=mod['offset']))
 return dict(binary_sha256=sha(raw),device_sections=names,fatbin_sha256=sha(fat['data']),fatbin_bytes=fat['size'],wrapper_sha256=sha(segment['data']),wrappers=wrappers,modules=mods),fat['data'],segment['data']
def compare(before,after):
 a,af,aw=device_view(before);b,bf,bw=device_view(after);need(af==bf,'complete .hip_fatbin bytes changed');need(aw==bw,'raw wrapper bytes changed');need(a['modules']==b['modules'] and a['wrappers']==b['wrappers'],'module/kernel metadata or wrapper mapping changed');return dict(passed=True,complete_device_payload_bytes_equal=True,complete_module_kernel_metadata_equal=True,baseline=a,observer=b)
