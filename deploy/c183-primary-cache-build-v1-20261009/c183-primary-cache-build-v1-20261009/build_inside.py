#!/usr/bin/env python3
"""Original-source provenance reproduction only. No GPU/model/observer execution."""
import hashlib,json,os,shlex,subprocess,time,signal
from pathlib import Path
ORIGINAL=Path('/original')
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
HERE=Path(__file__).resolve().parent
V4=Path('/home/radneon/strata/logs/perf-campaign-20261009/hc-baseline-provenance-v4-results')
V4_HOST='5696899be076d084e8b1896efffed5ad1981c484b79913a5486fc729f864ad87'
V4_INNER='1b483e54874fdc44b8497209068daff93eb46b798e4a1d96038f7c728b4fda27'
REPAIRED_ENGINE='6999f03f57473776a99b72bf4ebe8766a9c6c751f74affd010a2fccf36412da9'
PATCHED={'include/strata/prefill/hc_negative_cache.hpp': '973aa1dd196c578d52bb2107e2826f2aae271400a036abefda3c8ca3135e59a2', 'src/prefill/gemm.cu': '34d9eebdb4a43691bec0af73512dd1104c62be2a3586a435f007bba2dc18d774', 'src/prefill/prefill.cpp': '6993ea906198364e8bca514bcb8df42bfe8fe3f7531fff96522903e00197b6cb'}
SOURCE_REVIEW='f46c48f0fb41a819d1fbe9d2c02783a185f8d16878c57363a5e928ca018ff4e6'
def validate_source_review():
 need(sha(HERE/'source-review.json')==SOURCE_REVIEW,'source review seal changed')
 review=json.loads((HERE/'source-review.json').read_text());pins=json.loads((HERE/'source-pins.json').read_text())
 need(review.get('passed') is True and review.get('status')=='PASS_CONDITIONAL_PRIMARY_WNEG_V2_SOURCE_AND_CPU_REVIEW','source review gate')
 need(review['source_pins']==pins and pins['baseline_sha256']==PINS,'review/source baseline binding')
 need(pins['candidate_sha256']=={**PATCHED,'include/strata/prefill/gemm.hpp':PINS['include/strata/prefill/gemm.hpp']},'review/source candidate binding')
 need(sha(HERE/'primary-hc-wnegative.patch')==pins['patch_sha256']==review['patch_sha256'],'review patch binding')
 need(sha(HERE/'patched/include/strata/prefill/gemm.hpp')==PINS['include/strata/prefill/gemm.hpp'],'public Gemm header changed')
 return dict(sha256=SOURCE_REVIEW,status=review['status'],source_archive_sha256=review['package_archive_sha256'],reviewed_generate_flow_only=True)
def validate_baseline():
 validate_source_review();result={}
 for relative,digest in PATCHED.items():
  actual=sha(HERE/'patched'/relative);need(actual==digest,'observer source mismatch: '+relative);result[relative]=actual
 return result
def validate_v4(root=V4):
 root=Path(root);need(sha(root/'host-receipt.json')==V4_HOST and sha(root/'reproduction/receipt.json')==V4_INNER,'v4 receipt pin mismatch');host=json.loads((root/'host-receipt.json').read_text());inner=json.loads((root/'reproduction/receipt.json').read_text())
 need(host['passed'] is True and host['status']=='PASS_ORIGINAL_SOURCE_PROVENANCE' and host['returncode']==0 and host['cleanup']['passed'] is True and host['inner_receipt_sha256']==V4_INNER,'v4 host gate')
 need(inner['passed'] is True and inner['status']=='PASS_ORIGINAL_OBJECT_AND_FROZEN_BINARY_REPRODUCTION' and inner['relinked_binary']['equal'] is True and inner['relinked_binary']['actual_sha256']==BASE,'v4 inner gate');need(sha(root/'scratch-build/strata')==BASE,'v4 baseline binary mismatch')
 files={}
 for name in closure():
  actual=sha(root/'scratch-build'/name);need(actual==host['artifact_hashes']['scratch-build/'+name],'v4 closure mismatch: '+name);files[name]=actual
 need(files['libstrata_engine.a']==REPAIRED_ENGINE,'repaired engine pin mismatch');return dict(host_sha256=V4_HOST,inner_sha256=V4_INNER,binary_sha256=BASE,files=files)
def prepare_headers(source,out):
 src=Path(source)/'include/strata/prefill';dest=Path(out)/'header-overlay';dest.mkdir(mode=0o700);result={};files=sorted(src.iterdir());need(0<len(files)<=16,'bounded header directory')
 for f in files:
  need(f.is_file() and not f.is_symlink() and f.stat().st_size<=1024**2,'regular bounded header required');data=f.read_bytes();(dest/f.name).write_bytes(data);result[f.name]=sha(f);need(sha(dest/f.name)==result[f.name],'header copy mismatch')
 need('hc_negative_cache.hpp' not in result,'observer header already present in original');(dest/'hc_negative_cache.hpp').write_bytes((HERE/'patched/include/strata/prefill/hc_negative_cache.hpp').read_bytes());return result
def check_headers(source,out,receipt):
 src=Path(source)/'include/strata/prefill';dest=Path(out)/'header-overlay';need(sorted(x.name for x in src.iterdir())==sorted(receipt),'original header inventory changed')
 for name,digest in receipt.items():need(sha(src/name)==digest and sha(dest/name)==digest,'header changed: '+name)
 need(sha(dest/'hc_negative_cache.hpp')==PATCHED['include/strata/prefill/hc_negative_cache.hpp'],'observer header changed')
def archive_members(receipt,phase):
 archive=BUILD/'libstrata_prefill.a';ar='/opt/rocm/llvm/bin/llvm-ar';run_step(receipt,phase+'-prefill-list',[ar,'t',str(archive)],30);names=(OUT/(phase+'-prefill-list.stdout')).read_text().splitlines();need(len(names)==len(set(names)) and all(names.count(n)==1 for n in OBJECTS),'unique prefill archive members required');result=[]
 for i,name in enumerate(names):
  need(name and '/' not in name and name not in ('.','..'),'unsafe archive member name');out=OUT/(phase+'-member-'+str(i)+'.o');run_step(receipt,phase+'-member-'+str(i),[ar,'p',str(archive),name],30,out);result.append(dict(name=name,sha256=sha(out)))
 return result
def check_replacement(before,after,changed):
 need([x['name'] for x in before]==[x['name'] for x in after],'prefill member order/list changed')
 for a,b in zip(before,after):
  if a['name'] in OBJECTS:need(a['sha256']==OBJECTS[a['name']] and b['sha256']==changed[a['name']],'replacement member mismatch')
  else:need(a==b,'unrelated prefill member changed')
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
def commands():
 result=[]
 for recipe in RECIPES.values():
  for key,value in FLAGS.items():recipe=recipe.replace('$('+key+')',shlex.join(value))
  result.append(shlex.split(recipe))
 return result+[LINK.copy()]
def closure():
 names={PREFIX+n for n in OBJECTS}
 names.update(['CMakeFiles/strata_prefill.dir/flags.make','CMakeFiles/strata_prefill.dir/build.make','CMakeFiles/strata.dir/link.txt'])
 names.update(a for a in LINK[1:] if not a.startswith(('-', '/')) and a!='strata')
 return sorted(names)
def copy_build(source,dest):
 source=Path(source)/'scratch-build';dest=Path(dest);need(not dest.exists(),'fresh scratch build required');dest.mkdir(mode=0o700);result={};total=0;started=time.monotonic()
 for name in closure():
  src=source/name;dst=dest/name;need(not src.is_symlink() and src.is_file(),'regular original closure file required: '+name);need(source.resolve() in src.resolve().parents,'closure path escaped original build')
  size=src.stat().st_size;total+=size;need(total<=4*1024**3,'closure exceeds 4GiB bound');before=sha(src);dst.parent.mkdir(parents=True,exist_ok=True)
  with src.open('rb') as fi,dst.open('xb') as fo:
   while True:
    need(time.monotonic()-started<120,'scratch copy120s deadline');data=fi.read(4*1024**2)
    if not data:break
    fo.write(data)
  need(sha(dst)==before and sha(src)==before,'copy hash mismatch: '+name);need(dst.stat().st_nlink==1 and (dst.stat().st_dev,dst.stat().st_ino)!=(src.stat().st_dev,src.stat().st_ino),'copy must not be hardlinked')
  result[name]=dict(sha256=before,bytes=size)
 return dict(files=result,total_bytes=total,copy_deadline_seconds=120)
def verify_closure(source,receipt):
 for name,row in receipt['files'].items():need(sha(Path(source)/'scratch-build'/name)==row['sha256'],'original closure changed: '+name)
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
 OUT.mkdir();receipt=dict(status='RUNNING_OBSERVER_BUILD',passed=False,steps=[],source_read_only=True,gpu_devices=False,network=False,model_run=False);write(OUT/'receipt.json',receipt)
 try:
  need(not Path('/dev/kfd').exists() and not Path('/dev/dri').exists(),'GPU devices must not be exposed');initial=validate_metadata(ORIGINAL);v4=validate_v4(Path('/v4'));receipt['v4_inputs']=v4;receipt['original_inputs']=initial;receipt['observer_sources']=validate_baseline();receipt['source_review']=validate_source_review()
  for relative,digest in PATCHED.items():need(sha(SOURCE/relative)==digest,'observer overlay mismatch')
  host=json.loads(Path('/out/host-receipt.json').read_text());check_headers(ORIGINAL,Path('/out'),host['header_overlay']);receipt['link_inputs_before']=link_inputs();receipt['original_link_inputs_before']=link_inputs(ORIGINAL/'build-906')
  for name in closure():need(sha(BUILD/name)==v4['files'][name],'scratch closure differs from v4: '+name)
  for label,tool in [('hip-compiler','/opt/rocm/lib/llvm/bin/clang++'),('cxx-compiler','/opt/rocm/llvm/bin/clang++'),('archiver','/opt/rocm/llvm/bin/llvm-ar'),('readelf','/usr/bin/readelf')]:receipt.setdefault('tool_hashes',{})[tool]=sha(tool);run_step(receipt,label+'-version',[tool,'--version'],15)
  before=archive_members(receipt,'before');receipt['members_before']=before;write(OUT/'receipt.json',receipt)
  for row in before:
   if row['name'] in OBJECTS:need(row['sha256']==OBJECTS[row['name']],'original prefill member mismatch')
  changed={}
  for name,argv in zip(OBJECTS,commands()[:2]):run_step(receipt,'compile-observer-'+name,argv,300);changed[name]=sha(BUILD/PREFIX/name);receipt['observer_objects']=changed;write(OUT/'receipt.json',receipt)
  run_step(receipt,'replace-two-prefill-members',['/opt/rocm/llvm/bin/llvm-ar','r','libstrata_prefill.a']+[PREFIX+n for n in OBJECTS],30)
  after=archive_members(receipt,'after');receipt['members_after']=after;check_replacement(before,after,changed);repaired=link_inputs();receipt['link_inputs_after_replacement']=repaired
  for name,digest in receipt['link_inputs_before'].items():
   if name!=str(BUILD/'libstrata_prefill.a'):need(repaired[name]==digest,'unrelated link input changed')
  need(sha(BUILD/'libstrata_engine.a')==REPAIRED_ENGINE,'repaired verifier changed');write(OUT/'receipt.json',receipt)
  run_step(receipt,'link-observer',LINK.copy(),180);receipt['observer_binary_sha256']=sha(BUILD/'strata');write(OUT/'receipt.json',receipt)
  import tls_verify
  receipt['tls_sharing']=tls_verify.verify(BUILD/'strata',[BUILD/PREFIX/n for n in OBJECTS],OUT,lambda label,argv:run_step(receipt,label,argv,30))
  write(OUT/'receipt.json',receipt)
  import device_compare
  comparison=device_compare.compare(Path('/v4/scratch-build/strata'),BUILD/'strata');write(OUT/'device-equality.json',comparison);receipt['device_equality_sha256']=sha(OUT/'device-equality.json');need(comparison['passed'] is True,'device equality gate')
  need(validate_v4(Path('/v4'))==v4,'v4 changed');need(validate_metadata(ORIGINAL)==initial and link_inputs(ORIGINAL/'build-906')==receipt['original_link_inputs_before'],'original changed');need(link_inputs()==repaired,'link inputs changed during link');check_headers(ORIGINAL,Path('/out'),host['header_overlay'])
  for relative,digest in PATCHED.items():need(sha(SOURCE/relative)==digest,'observer overlay changed')
  receipt.update(status='PASS_PRIMARY_CACHE_BUILD_DEVICE_AND_TLS_EQUALITY',passed=True)
 except BaseException as e:receipt.update(status='FAILED_STOP_FOR_ANALYSIS',error_type=type(e).__name__,error=str(e));raise
 finally:receipt['artifact_hashes']={str(x.relative_to(OUT)):sha(x) for x in OUT.rglob('*') if x.is_file() and x.name not in ('receipt.json','receipt.json.tmp')};write(OUT/'receipt.json',receipt)
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
