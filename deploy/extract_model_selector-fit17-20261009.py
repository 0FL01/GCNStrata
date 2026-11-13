#!/usr/bin/env python3
"""CPU-only inspect existing model/diagnostic HIP objects; never compile or access GPUs."""
import argparse,hashlib,json,re,struct,subprocess
from pathlib import Path
TOOLS=Path('/opt/rocm/lib/llvm/bin')
def need(v,s):
 if not v:raise RuntimeError(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args,out):
 with out.open('wb') as f:subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,check=True)
def extract(label,obj,wanted,out):
 need(sha(obj)==wanted,label+' input object SHA mismatch')
 dest=out/label;dest.mkdir();raw=obj.read_bytes();target='direct-amdgpu-elf'
 if raw[:4]==b'\x7fELF' and len(raw)>20 and struct.unpack_from('<H',raw,18)[0]==224:
  image=raw
 else:
  fat=dest/'hip-fatbin.bin'
  # This reads the existing host ELF; objcopy does not rewrite the input.
  run([TOOLS/'llvm-objcopy','--dump-section','.hip_fatbin='+str(fat),obj,dest/'host-copy.o'],dest/'objcopy.log')
  data=fat.read_bytes();magic=b'__CLANG_OFFLOAD_BUNDLE__';need(data.startswith(magic),'unsupported/compressed fatbin: preserve bytes; inspect pinned clang-offload-bundler before continuing')
  pos=len(magic);need(len(data)>=pos+8,'truncated bundle count');count=struct.unpack_from('<Q',data,pos)[0];pos+=8;need(0<count<=64,'invalid bundle count');entries=[]
  for _ in range(count):
   need(pos+24<=len(data),'truncated bundle header');offset,size,n=struct.unpack_from('<QQQ',data,pos);pos+=24
   need(0<n<4096 and pos+n<=len(data),'invalid bundle target');name=data[pos:pos+n].decode('ascii');pos+=n
   need(offset<=len(data) and size<=len(data)-offset,'bundle image bounds');entries.append(dict(target=name,offset=offset,size=size))
  need(all(e['offset']>=pos for e in entries),'image overlaps bundle directory')
  (dest/'bundle-targets.json').write_text(json.dumps(entries,indent=2)+'\n')
  candidates=[e for e in entries if re.search(r'(?:^|[-:])gfx906(?:$|[-:+])',e['target'])]
  need(len(candidates)==1,'need exactly one existing gfx906 bundle, including target feature suffixes')
  e=candidates[0];target=e['target'];image=data[e['offset']:e['offset']+e['size']]
 need(image[:4]==b'\x7fELF' and len(image)>20 and image[5]==1 and struct.unpack_from('<H',image,18)[0]==224,'selected image is not little-endian AMDGPU ELF')
 elf=dest/'gfx906.codeobject';elf.write_bytes(image)
 run([TOOLS/'llvm-readelf','--notes',elf],dest/'notes.txt')
 run([TOOLS/'llvm-readelf','--symbols',elf],dest/'symbols.txt')
 run([TOOLS/'llvm-objdump','--disassemble','--mcpu=gfx906',elf],dest/'disassembly.txt')
 need(sha(obj)==wanted,'input object changed during extraction')
 return dict(input_object=str(obj),input_object_sha256=wanted,target=target,device_image_sha256=sha(elf),disassembly_sha256=sha(dest/'disassembly.txt'),notes_sha256=sha(dest/'notes.txt'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['model-object','diagnostic-object','output']:p.add_argument('--'+n,type=Path,required=True)
 for n in ['model-object-sha','diagnostic-object-sha']:p.add_argument('--'+n,required=True)
 a=p.parse_args();need(all(re.fullmatch('[0-9a-f]{64}',x) for x in [a.model_object_sha,a.diagnostic_object_sha]),'explicit SHA256 values required')
 need(all(x.is_absolute() for x in [a.model_object,a.diagnostic_object,a.output]),'absolute paths required');need(not a.output.exists(),'fresh output directory required');a.output.mkdir(parents=True)
 run([TOOLS/'llvm-objdump','--version'],a.output/'toolchain.txt')
 result={'cpu_only':True,'recompiled':False,'model':extract('model',a.model_object,a.model_object_sha,a.output),'diagnostic':extract('diagnostic',a.diagnostic_object,a.diagnostic_object_sha,a.output)}
 (a.output/'extraction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
