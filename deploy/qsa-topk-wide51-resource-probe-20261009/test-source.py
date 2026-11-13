#!/usr/bin/env python3
"""CPU-only source-contract tests. Never calls HIP, links a selector, or opens a GPU."""
import argparse, difflib, hashlib, itertools, json, os, pathlib, re, subprocess

ap=argparse.ArgumentParser()
ap.add_argument("--output", required=True)
ap.add_argument("--cxx", default=os.environ.get("CXX","c++"))
args=ap.parse_args()
root=pathlib.Path(__file__).resolve().parent
out=pathlib.Path(args.output).resolve()
out.mkdir(parents=True,exist_ok=False)
base=(root/"qsa_select.fit17.cu").read_text()
cand=(root/"qsa_select.wide51.cu").read_text()
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
assert sha(base)=="5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a"
old_assert="(WINDOW_GUARD == 2 && PER == TK_PER_MAX)"
new_assert="(WINDOW_GUARD == 2 && (PER == TK_PER_MAX || PER == 51))"
old_launch="            block_topk_reg_kernel<TK_PER_MAX, true, 2><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);"
start=cand.index("            // Allocation capacity is immutable")
end=cand.index("\n        } else {",start)
new_launch=cand[start:end]
assert cand.replace(new_assert,old_assert).replace(new_launch,old_launch)==base
patch="".join(difflib.unified_diff(base.splitlines(True),cand.splitlines(True),fromfile="a/src/kernels/cuda/qsa_select.cu",tofile="b/src/kernels/cuda/qsa_select.cu"))
assert patch==(root/"qsa-topk-wide51.patch").read_text()
assert base.count("constexpr int TK_PER_MAX = 66;")==cand.count("constexpr int TK_PER_MAX = 66;")==1
assert new_launch.count('std::getenv("STRATA_GFX906_TOPK_WIDE51")')==1
assert 'return v && std::strcmp(v, "1") == 0;' in new_launch
assert "if (wide51 && reach <= (int64_t) TK_T * 51)" in new_launch
assert cand.index("if (fit17) {") < start < cand.index("\n        } else {",start)
assert cand.count("STRATA_GFX906_TOPK_WIDE51")==1
# Every other byte, including the device guards, arithmetic and legacy trace, is unchanged.
kernel_start=base.index("template <int PER, bool LONG_ONLY")
kernel_end=base.index("\n//",kernel_start)
while kernel_end<base.index("    const int64_t eq_budget",kernel_start):
    kernel_end=base.index("\n//",kernel_end+1)
guard_start=base.index("    if constexpr (LONG_ONLY) {",kernel_start)
guard_end=base.index("    int32_t* out =",guard_start)
guard=base[guard_start:guard_end]
assert cand[cand.index("    if constexpr (LONG_ONLY) {",kernel_start):cand.index("    int32_t* out =",guard_start)].replace(new_assert,old_assert)==guard
assert "const bool fits17 = (int64_t) max_bid + 1 <= (int64_t) TK_T * 17;" in guard
assert "__syncthreads" not in guard
assert "if constexpr (WINDOW_GUARD == 1) { if (!fits17) return; }" in guard
assert "else { if (fits17) return; }" in guard

prefix=r'''
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
using cudaError_t=int;
constexpr int cudaSuccess=0;
int cudaGetLastError(){return 0;}
const char* cudaGetErrorString(int){return "";}
constexpr int TK_T=1024,TK_PER=33,R=4,kTopkMaxCells=2048,CL_MAXQ=8;
#ifdef __HIPCC__
constexpr int TK_PER_MAX=66;
#else
constexpr int TK_PER_MAX=TK_PER;
#endif
struct QsaShapes { int idx_block=4; };
int qsa_selection_width(int,const QsaShapes&){return 2048;}
std::vector<std::string> launches;
void record(const char* s){launches.emplace_back(s);}
void qsa_block_topk_ref(const float*,const int32_t*,int64_t,int64_t,int64_t,const QsaShapes&,int32_t*,void*){record("ref");}
bool qsa_block_topk_cluster(const float*,const int32_t*,int64_t,int64_t,int64_t,const QsaShapes&,int32_t*,void*){return false;}
bool topk_active_turing_device(){return false;}
'''
main=r'''
int main(int argc,char**argv){
 if(argc!=4)return 2;
 QsaShapes s;
 qsa_block_topk(nullptr,nullptr,std::stoll(argv[2]),std::stoll(argv[1]),2048,s,nullptr,nullptr,std::stoll(argv[3]));
 for(auto&v:launches)std::cout<<v<<"\n";
}
'''
arches={"gfx906":["-D__HIPCC__=1","-DSTRATA_HIP_GFX906=1"],"hip_other":["-D__HIPCC__=1"],"cuda":[]}
binaries={}
for variant,source in [("fit17",base),("wide51",cand)]:
    body=source[source.index("void qsa_block_topk(const float*"):source.rindex("\n}  // namespace strata::kernels")]
    body,n=re.subn(r"(block_topk_(?:reg_)?kernel(?:<[^;\n]*?>)?|block_topk_wide_kernel)<<<.*?>>>\([^;]*\);",lambda m:'record("'+m.group(1)+'");',body)
    assert n==(7 if variant=="fit17" else 8),(variant,n)
    assert "<<<" not in body
    cpp=out/(variant+".dispatch.cpp")
    cpp.write_text(prefix+body+main)
    for arch,defs in arches.items():
        exe=out/(variant+"."+arch)
        subprocess.run([args.cxx,"-std=c++20","-O0",*defs,str(cpp),"-o",str(exe)],check=True,capture_output=True)
        binaries[variant,arch]=exe
env_names=["STRATA_GFX906_TOPK_WIDE51","STRATA_GFX906_TOPK_FIT17","STRATA_GFX906_TOPK_FIT17_TRACE","STRATA_GFX906_TOPK_REG","STRATA_TOPK_OLD","STRATA_QSA_CLUSTER","STRATA_TOPK_CAPACITY_GUARD","STRATA_TOPK_ACTIVE_ANY","STRATA_TOPK_STREAM"]
def dispatch(variant,arch,cap,nq,active,wide="1",fit="1",reg="2",old=None):
    env=dict(os.environ)
    for name in env_names: env.pop(name,None)
    for name,val in [("STRATA_GFX906_TOPK_WIDE51",wide),("STRATA_GFX906_TOPK_FIT17",fit),("STRATA_GFX906_TOPK_REG",reg),("STRATA_TOPK_OLD",old)]:
        if val is not None:env[name]=val
    return subprocess.run([str(binaries[variant,arch]),str(cap),str(nq),str(active)],env=env,check=True,text=True,capture_output=True).stdout.splitlines()
cases=[]
def check(arch,cap,nq,active=0,wide="1",fit="1",reg="2",old=None):
    a=dispatch("fit17",arch,cap,nq,active,wide,fit,reg,old)
    b=dispatch("wide51",arch,cap,nq,active,wide,fit,reg,old)
    eligible=(arch=="gfx906" and wide=="1" and fit=="1" and reg=="2" and old is None and active<=0 and 0<nq<=8 and 33792<cap<=52224)
    want=a[:]
    if eligible:
        assert a==["block_topk_kernel<true>","block_topk_reg_kernel<17, true, 1>","block_topk_reg_kernel<TK_PER_MAX, true, 2>"],a
        want[-1]="block_topk_reg_kernel<51, true, 2>"
    assert b==want,(arch,cap,nq,active,wide,fit,reg,old,a,b,want)
    assert len(a)==len(b)
    cases.append(dict(arch=arch,capacity=cap,nq=nq,active=active,wide=wide,fit=fit,reg=reg,old=old,launches=b))
env_cases=[None,"","0","1","01","10","2","true"," 1","1 ","+1","1\n"]
for arch,cap,nq in itertools.product(arches,[33792,33793,51202,52223,52224,52225,67584,67585],[1,4,6,8,9]):
    check(arch,cap,nq)
for cap,nq,value in itertools.product([52223,52224,52225],[1,4,6,8,9],env_cases):
    check("gfx906",cap,nq,wide=value)
    check("gfx906",cap,nq,fit=value)
for arch,cap,nq,active in itertools.product(arches,[52223,52224,52225],[1,4,6,8,9],[1,7167,7168,17408,33792,52224,70000]):
    check(arch,cap,nq,active=active)
for cap,nq in itertools.product([52223,52224,52225],[1,4,6,8,9]):
    for old in ["","0","1"]:check("gfx906",cap,nq,old=old)
    for reg in [None,"","0","1"]:check("gfx906",cap,nq,reg=reg)
for arch,nq in itertools.product(arches,[-1,0]):check(arch,52224,nq)
# Compile and execute the actual extracted guard, replacing only early void return with false.
guard_cand=cand[cand.index("    if constexpr (LONG_ONLY) {",kernel_start):cand.index("    int32_t* out =",guard_start)]
guard_fn=guard_cand.replace("return;","return false;")
gprefix=r'''
#include <cstdint>
#include <cstdio>
#include <algorithm>
constexpr int TK_T=1024,TK_PER_MAX=66,kStepCount=4,kStepNKv=1,kStepNBid=2,GFX906_TOPK_SHORT_CELLS=24576;
struct {unsigned x;} gridDim;
template<int PER,bool LONG_ONLY,int WINDOW_GUARD>
bool guard(const int32_t*steps){
'''
gmain=r'''
 return true;
}
int main(){
 int32_t steps[8*kStepCount]={};
 int tested=0;
 for(int nq: {1,4,6,8}){
  gridDim.x=nq;
  for(int nb: {1,17407,17408,17409,51202,52223,52224}){
   for(int short_window: {0,1}){
    for(int row=0;row<nq;row++){steps[row*kStepCount+kStepNKv]=short_window?GFX906_TOPK_SHORT_CELLS:GFX906_TOPK_SHORT_CELLS+1;steps[row*kStepCount+kStepNBid]=0;}
    // Put largest row first and smallest last: guards must not assume sorted n_bid.
    steps[kStepNBid]=nb-1;
    const bool g17=guard<17,true,1>(steps);
    const bool g51=guard<51,true,2>(steps);
    const bool g66=guard<66,true,2>(steps);
    if(g51!=g66 || (g17+g51 != (short_window?0:1)) || (g51 != (!short_window && nb>17408)))return 3;
    ++tested;
   }
  }
 }
 std::printf("%d\n",tested);
}
'''
guard_cpp=out/"guard.cpp";guard_cpp.write_text(gprefix+guard_fn+gmain)
assert "constexpr int64_t GFX906_TOPK_SHORT_CELLS = 24576;" in base
header=(root/"include/strata/kernels/qsa.hpp").read_text()
for name,value in [("kStepCount",4),("kStepNKv",1),("kStepNBid",2)]:
    assert re.search(r"\b"+name+r"\s*=\s*"+str(value)+r"\b",header)
guard_exe=out/"guard"
subprocess.run([args.cxx,"-std=c++20",str(guard_cpp),"-o",str(guard_exe)],check=True,capture_output=True)
guard_count=int(subprocess.run([str(guard_exe)],check=True,text=True,capture_output=True).stdout)
# Compile-time validity matrix retains the LONG_ONLY and guard/PER restrictions.
matrix=[]
for per,long_only,window in [(17,True,1),(51,True,2),(66,True,2),(51,False,2),(51,True,1),(17,True,2),(51,True,3),(50,True,2)]:
    valid=long_only and ((window==1 and per==17)or(window==2 and per in [51,66]))
    text=gprefix+guard_fn+"\n return true;\n}\nint main(){int32_t s[32]={};return guard<%d,%s,%d>(s);}\n"%(per,str(long_only).lower(),window)
    trial=out/"assert-trial.cpp";trial.write_text(text)
    r=subprocess.run([args.cxx,"-std=c++20","-fsyntax-only",str(trial)],capture_output=True,text=True)
    assert (r.returncode==0)==valid,(per,long_only,window,r.stderr)
    matrix.append(dict(per=per,long_only=long_only,window_guard=window,accepted=valid))
# Exact ceiling proof for all possible valid row counts, not a few handpicked rows.
for capacity in [51202,52223,52224]:
    for nb in range(1,capacity+1):assert (nb+1023)//1024<=51
assert (52225+1023)//1024==52
receipt=dict(passed=True,gpu_executed=False,source_only=True,dispatch_cases=len(cases),guard_cases=guard_count,static_assert_matrix=matrix,all_other_source_bytes_preserved=True,launch_count_preserved=True,capacity_contract="Every valid n_bid + 1 <= immutable max_blocks; unknown invalid rows are not supported.",base_sha256=sha(base),candidate_sha256=sha(cand),patch_sha256=sha(patch),limitations=["CPU dispatch uses extracted host function with launch recording; no GPU semantic or performance claim.","Existing FIT17 trace text remains byte-identical and does not identify WIDE51.","The 64K fixture exercises reg17 and cannot qualify wide51 performance."])
(out/"dispatch-cases.json").write_text(json.dumps(cases,indent=2)+"\n")
(out/"source-proof.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))

