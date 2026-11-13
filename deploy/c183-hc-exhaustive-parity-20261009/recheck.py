#!/usr/bin/env python3
"""Independent CPU bitwise parity and raw guard recheck. No numerical NaN oracle."""
import hashlib,json,struct
from pathlib import Path
GUARD=0xa55aa55a
def unique_object(items):
 result={}
 for key,value in items:
  need(key not in result,'duplicate JSON key');result[key]=value
 return result
def need(ok,label):
 if not ok:raise ValueError(label)
def specs():
 for scale in (4096,512):
  for first in range(0,65536,256):yield (0,scale,first,256,1,0)
  for offset in (0,5):
   for n in (0,1,63,64,65,255,256,257,448):yield (1,scale,0,1,n,offset)
def parse(data,expected,initial=False):
 need(len(data)>=28,'truncated header');h=struct.unpack_from('<7I',data);need(h==(0x48435031,*expected),'header identity/order')
 kind,scale,start,pairs,n,offset=expected;words=2*pairs*(n+2);count=2*pairs*23
 need(len(data)==28+words*4+count*4,'raw length')
 inputs=struct.unpack_from('<'+str(words)+'H',data,28);outputs=struct.unpack_from('<'+str(words)+'H',data,28+words*2);stats=struct.unpack_from('<'+str(count)+'I',data,28+words*4)
 edge=(0,0x8000,1,0x7f,0x80,0x3f80,0xbf80,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc0,0x7f81,0xff81)
 for r in range(pairs*2):
  at=r*(n+2);expected_input=(0xa55a,)+tuple((start+r//2) if kind==0 else (edge[j] if j<14 else (j*40503+start)&65535) for j in range(n))+(0x5aa5,)
  need(inputs[at:at+n+2]==expected_input,'input changed')
  need(outputs[at]==0xa55a and outputs[at+n+1]==0x5aa5,'output guard')
  need(all(stats[r*23+j]==GUARD for j in range(23) if j<1+offset or j>=17+offset),'stats guard')
  for j,value in enumerate(stats[r*23+1+offset:r*23+17+offset]):need(value<=1 if n and j in (1,2,3) else value==0,'stats valid flags/untouched words')
  if initial:
   need(all(x==0xdead for x in outputs[at+1:at+n+1]),'initial output sentinel')
   need(all(x==0 for x in stats[r*23+1+offset:r*23+17+offset]),'initial zero stats')
  if r%2:
   need(outputs[at:at+n+2]==outputs[at-n-2:at],'output parity')
   need(stats[r*23:(r+1)*23]==stats[(r-1)*23:r*23],'all stats parity')
  if n==0:need(all(x==0 for x in stats[r*23+1+offset:r*23+17+offset]),'n=0 stats')
 return pairs
def check(out):
 out=Path(out);raw=(out/'raw.jsonl').read_bytes();rows=[json.loads(line,object_pairs_hook=unique_object) for line in raw.splitlines()]
 need(rows and rows[-1]=={'type':'complete','pairs':131108,'launches':262216,'batches':548,'passed':True},'completion')
 need(all(type(r.get('rc')) is int and r['rc']==0 and type(r.get('operation')) is str for r in rows if r.get('type')=='hip_rc'),'HIP return code')
 need(not any(r['type']=='failure' for r in rows),'native failure')
 need(sum(r['type']=='hip_rc' and r['operation'].startswith('hipModuleLaunchKernel(') for r in rows)==262216,'launch RC coverage')
 begins=[r for r in rows if r['type']=='batch_begin'];ends=[r for r in rows if r['type']=='batch_complete'];need(len(begins)==len(ends)==548,'batch coverage')
 hashes={};pairs=0;launches=0
 for i,spec in enumerate(specs()):
  kind,scale,start,num,n,offset=spec
  need(begins[i]==dict(type='batch_begin',batch=i,kind=kind,scale=scale,start=start,pairs=num,n=n,offset=offset,launches_before=launches),'batch begin identity')
  launches+=2*num
  need(ends[i]==dict(type='batch_complete',batch=i,launches=launches,equal=True),'batch end identity')
  f=out/f'batch-{i:04d}.complete.bin';data=f.read_bytes();pairs+=parse(data,spec);hashes[f.name]=hashlib.sha256(data).hexdigest()
  initial=out/f'batch-{i:04d}.initial.bin';data=initial.read_bytes();parse(data,spec,initial=True);hashes[initial.name]=hashlib.sha256(data).hexdigest()
 need(len(list(out.glob('batch-*.complete.bin')))==548,'extra raw batch')
 return dict(passed=True,pairs=pairs,launches=launches,batches=548,raw_binary_sha256=hashes,raw_jsonl_sha256=hashlib.sha256(raw).hexdigest(),performance_claim=False)
if __name__=='__main__':
 import sys
 print(json.dumps(check(sys.argv[1]),indent=2))
