#!/usr/bin/env python3
"""Original-source provenance reproduction only. No GPU/model/observer execution."""
import hashlib,json,os,shlex,subprocess,time,signal
from pathlib import Path
SOURCE=Path('/src');BUILD=SOURCE/'build-906';OUT=Path('/out/reproduction')
BASE='c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049'
PINS={'src/prefill/gemm.cu':'6f73546e4b654a9cc81f4b333a09530c7425e13f6ce33ac16ecb88a05b07fabd','src/prefill/prefill.cpp':'1a8855dd4e1d6116f10060f2be2110014ecc2afb9e80c9164fc0689c8f175b42','include/strata/prefill/gemm.hpp':'a46c76cad934f4c4114a0f984aee2a7145c8549fc7b1efee16b09d62aab2cfaf'}
OBJECTS={'gemm.cu.o':'9b0d9fb8cf20c97bb81e43b363e6a6b68d32af0d36234b90ed058ad4433ad048','prefill.cpp.o':'efff91107cf3339e5929ac510097168da5c51d64012a93ece3894aa64b5d215d'}
PREFIX='CMakeFiles/strata_prefill.dir/src/prefill/'
DEFINES=['-DSTRATA_HIP_GFX906=1','-DSTRATA_NATIVE_EXPERTS=1','-DSTRATA_PREFILL_FUSED=1','-DSTRATA_PREFILL_MMQ=1','-DSTRATA_VERSION="0.1.40"','-DUSE_PROF_API=1','-D__HIP_PLATFORM_AMD__=1']
COMMON_INCLUDES=['-I/src/include','-I/src/third_party/llama.cpp/ggml/src/../include','-isystem','/src/include/strata/platform/hip_compat']
FLAGS={'CXX_DEFINES':DEFINES,'CXX_INCLUDES':COMMON_INCLUDES+['-isystem','/opt/rocm/include','-isystem','/opt/rocm/core-7.14/include'],'CXX_FLAGS':['-O3','-DNDEBUG','-std=c++20','-Wall','-Wextra'],'HIP_DEFINES':DEFINES+['-D__HIP_ROCclr__=1'],'HIP_INCLUDES':COMMON_INCLUDES,'HIP_FLAGS':['-O3','-DNDEBUG','-std=c++20','--offload-arch=gfx906','-Wall','-Wextra']}
LINK=shlex.split('/opt/rocm/lib/llvm/bin/clang++ -O3 -DNDEBUG --offload-arch=gfx906 CMakeFiles/strata.dir/src/program/generate.cpp.o -o strata libstrata_engine.a libstrata_prefill.a libstrata_spec.a libstrata_engine.a libstrata_kernels.a libstrata_core.a libstrata_kernels_cpu.a ggml/src/libggml-cpu.a /opt/rocm/core-7.14/lib/libhipblas.so.3.5 libstrata_mmq.a /opt/rocm/lib/libamdhip64.so.7.14.60850-0000000 /opt/rocm/core-7.14/lib/llvm/lib/clang/23/lib/linux/libclang_rt.builtins-x86_64.a /opt/rocm/lib/libamdhip64.so.7.14.60850-0000000 ggml/src/libggml-base.a -lm')
RECIPES={'gemm.cu.o':'/opt/rocm/lib/llvm/bin/clang++ $(HIP_DEFINES) $(HIP_INCLUDES) $(HIP_FLAGS) -o CMakeFiles/strata_prefill.dir/src/prefill/gemm.cu.o -x hip -c /src/src/prefill/gemm.cu','prefill.cpp.o':'/opt/rocm/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o -MF CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o.d -o CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o -c /src/src/prefill/prefill.cpp'}
def need(ok,label):
 if not ok:raise RuntimeError(label)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def read_flags(text):
 parsed={}
 for line in text.splitlines():
  if ' = ' in line:
   k,v=line.split(' = ',1)
   if k in FLAGS:need(k not in parsed,'duplicate flag group');parsed[k]=shlex.split(v)
 need(parsed==FLAGS,'original compile flag mismatch');return parsed
def validate_metadata(source=SOURCE):
 source=Path(source);build=source/'build-906';hashes={}
 for relative,digest in PINS.items():
  actual=sha(source/relative);need(actual==digest,'source mismatch: '+relative+' expected='+digest+' actual='+actual);hashes[relative]=actual
 for name,digest in OBJECTS.items():
  actual=sha(build/PREFIX/name);need(actual==digest,'original object mismatch: '+name+' expected='+digest+' actual='+actual);hashes['build-906/'+PREFIX+name]=actual
 flags=build/'CMakeFiles/strata_prefill.dir/flags.make';read_flags(flags.read_text());hashes[str(flags.relative_to(source))]=sha(flags)
 make=build/'CMakeFiles/strata_prefill.dir/build.make';text=make.read_text()
 for name,recipe in RECIPES.items():
  suffix='-c /src/src/prefill/'+name[:-2];matches=[' '.join(x.split()) for x in text.splitlines() if x.strip().startswith('/opt/rocm/') and x.rstrip().endswith(suffix)];need(matches==[recipe],'original recipe mismatch: '+name)
 hashes[str(make.relative_to(source))]=sha(make)
 link=build/'CMakeFiles/strata.dir/link.txt';need(shlex.split(link.read_text())==LINK,'original link command mismatch');hashes[str(link.relative_to(source))]=sha(link)
 return hashes
def commands(out=OUT):
 out=Path(out);obj=out/'objects'
 gemm=['/opt/rocm/lib/llvm/bin/clang++']+FLAGS['HIP_DEFINES']+FLAGS['HIP_INCLUDES']+FLAGS['HIP_FLAGS']+['-o',str(obj/'gemm.cu.o'),'-x','hip','-c','/src/src/prefill/gemm.cu']
 prefill=['/opt/rocm/llvm/bin/clang++']+FLAGS['CXX_DEFINES']+FLAGS['CXX_INCLUDES']+FLAGS['CXX_FLAGS']+['-MD','-MT',PREFIX+'prefill.cpp.o','-MF',str(obj/'prefill.cpp.o.d'),'-o',str(obj/'prefill.cpp.o'),'-c','/src/src/prefill/prefill.cpp']
 link=LINK.copy();link[link.index('-o')+1]=str(out/'strata.relinked')
 return gemm,prefill,link
def link_inputs(build=BUILD):
 result={}
 for arg in LINK[1:]:
  if arg.startswith('-') or arg=='strata':continue
  p=Path(arg) if arg.startswith('/') else Path(build)/arg
  need(p.is_file(),'link input missing: '+str(p));result[str(p)]=sha(p)
 return result
def run_step(receipt,label,argv,timeout,stdout=None):
 entry=dict(label=label,argv=argv,cwd=str(BUILD),timeout_seconds=timeout,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),returncode=None)
 receipt['steps'].append(entry);write(OUT/'receipt.json',receipt)
 out=Path(stdout) if stdout else OUT/(label+'.stdout');err=OUT/(label+'.stderr');entry.update(stdout=str(out),stderr=str(err));write(OUT/'receipt.json',receipt)
 try:
  with out.open('xb') as fo,err.open('xb') as fe:
   p=subprocess.Popen(argv,cwd=BUILD,stdout=fo,stderr=fe,start_new_session=True)
   try:entry['returncode']=p.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGKILL);entry['returncode']=p.wait();raise
  need(p.returncode==0,'step failed: '+label)
 except subprocess.TimeoutExpired:
  entry.update(timed_out=True,error='bounded subprocess timeout');raise
 finally:
  entry['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());entry['output_hashes']={str(f):sha(f) for f in (out,err) if f.exists()};write(OUT/'receipt.json',receipt)
def reproduce():
 OUT.mkdir()
 receipt=dict(status='RUNNING_ORIGINAL_SOURCE_REPRODUCTION',passed=False,steps=[],source_read_only=True,gpu_devices=False,network=False,observer_source_used=False)
 write(OUT/'receipt.json',receipt)
 try:
  need(not Path('/dev/kfd').exists() and not Path('/dev/dri').exists(),'GPU devices must not be exposed')
  initial=validate_metadata();receipt['original_inputs']=initial;receipt['link_inputs_before']=link_inputs();write(OUT/'receipt.json',receipt)
  (OUT/'objects').mkdir();(OUT/'archive-members').mkdir()
  for label,tool in [('hip-compiler','/opt/rocm/lib/llvm/bin/clang++'),('cxx-compiler','/opt/rocm/llvm/bin/clang++'),('archiver','/opt/rocm/llvm/bin/llvm-ar')]:
   receipt.setdefault('tool_hashes',{})[tool]=sha(tool);run_step(receipt,label+'-version',[tool,'--version'],15)
  run_step(receipt,'archive-list',['/opt/rocm/llvm/bin/llvm-ar','t',str(BUILD/'libstrata_prefill.a')],30)
  names=(OUT/'archive-list.stdout').read_text().splitlines()
  for name,digest in OBJECTS.items():
   need(names.count(name)==1,'archive member uniqueness: '+name);path=OUT/'archive-members'/name
   run_step(receipt,'archive-'+name,['/opt/rocm/llvm/bin/llvm-ar','p',str(BUILD/'libstrata_prefill.a'),name],30,path)
   need(sha(path)==digest,'original archive member differs: '+name)
  for (name,digest),argv in zip(OBJECTS.items(),commands()[:2]):
   run_step(receipt,'compile-'+name,argv,300);actual=sha(OUT/'objects'/name);receipt.setdefault('recompiled_objects',{})[name]=dict(expected_sha256=digest,actual_sha256=actual,equal=actual==digest);write(OUT/'receipt.json',receipt);need(actual==digest,'recompiled object mismatch: '+name+'; stop, do not relax gate')
  run_step(receipt,'relink-unchanged-original-inputs',commands()[2],180);digest=sha(OUT/'strata.relinked');receipt['relinked_binary']=dict(expected_sha256=BASE,actual_sha256=digest,equal=digest==BASE);write(OUT/'receipt.json',receipt);need(digest==BASE,'relinked binary mismatch; stop, do not relax gate')
  need(validate_metadata()==initial,'source/build metadata changed');after=link_inputs();receipt['link_inputs_after']=after;need(after==receipt['link_inputs_before'],'original link input changed')
  receipt.update(status='PASS_ORIGINAL_OBJECT_AND_FROZEN_BINARY_REPRODUCTION',passed=True)
 except BaseException as e:receipt.update(error_type=type(e).__name__,error=str(e),status='FAILED_STOP_FOR_ANALYSIS');raise
 finally:
  receipt['artifact_hashes']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('receipt.json','receipt.json.tmp')};write(OUT/'receipt.json',receipt)
def main():
 try:reproduce()
 except BaseException as e:
  import sys
  failure=dict(passed=False,status='FAILED_STOP_FOR_ANALYSIS',error_type=type(e).__name__,error=str(e),phase='inner_entry_or_reproduction',uid=os.getuid(),gid=os.getgid(),output=str(OUT))
  try:write(OUT.parent/'inner-failure.json',failure)
  except BaseException as fallback:failure['fallback_receipt_error']=str(fallback)
  print(json.dumps(dict(inner_failure=failure),sort_keys=True),file=sys.stderr,flush=True)
  raise
if __name__=='__main__':main()
