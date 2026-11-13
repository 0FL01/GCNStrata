#!/usr/bin/env python3
"""CPU-only extracted host dispatch and actual device guard predicates."""
import argparse, hashlib, itertools, json, os, pathlib, re, subprocess
ROOT=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--output',required=True)
ap.add_argument('--cxx',default=os.environ.get('CXX','c++'))
args=ap.parse_args()
OUT=pathlib.Path(args.output).resolve();OUT.mkdir(parents=True,exist_ok=False)
prefix='\n#include <cstdint>\n#include <cstdlib>\n#include <cstring>\n#include <cstdio>\n#include <iostream>\n#include <string>\n#include <vector>\nusing cudaError_t=int;\nconstexpr int cudaSuccess=0;\nint cudaGetLastError(){return 0;}\nconst char* cudaGetErrorString(int){return "";}\nconstexpr int TK_T=1024,TK_PER=33,R=4,kTopkMaxCells=2048,CL_MAXQ=8;\n#ifdef __HIPCC__\nconstexpr int TK_PER_MAX=66;\n#else\nconstexpr int TK_PER_MAX=TK_PER;\n#endif\nstruct QsaShapes { int idx_block=4; };\nint qsa_selection_width(int,const QsaShapes&){return 2048;}\nstd::vector<std::string> launches;\nvoid record(const char* s){launches.emplace_back(s);}\nvoid qsa_block_topk_ref(const float*,const int32_t*,int64_t,int64_t,int64_t,const QsaShapes&,int32_t*,void*){record("ref");}\nbool qsa_block_topk_cluster(const float*,const int32_t*,int64_t,int64_t,int64_t,const QsaShapes&,int32_t*,void*){return false;}\nbool topk_active_turing_device(){return false;}\n'
main='\nint main(int argc,char**argv){\n if(argc!=4)return 2;\n QsaShapes s;\n qsa_block_topk(nullptr,nullptr,std::stoll(argv[2]),std::stoll(argv[1]),2048,s,nullptr,nullptr,std::stoll(argv[3]));\n for(auto&v:launches)std::cout<<v<<"\\n";\n}\n'
arches={'gfx906':['-D__HIPCC__=1','-DSTRATA_HIP_GFX906=1'],'hip_other':['-D__HIPCC__=1'],'cuda':[]}
sources={v:(ROOT/f'qsa_select.{v}.cu').read_text() for v in ('fit17','radix-stop')}
header=(ROOT/'include/strata/kernels/qsa.hpp').read_text()
for name,value in [('kStepCount',4),('kStepNKv',1),('kStepNBid',2)]:
    assert re.search(r'\b'+name+r'\s*=\s*'+str(value)+r'\b',header)
assert 'constexpr int64_t GFX906_TOPK_SHORT_CELLS = 24576;' in sources['fit17']
bins={}
for variant,source in sources.items():
    body=source[source.index('void qsa_block_topk(const float*'):source.rindex('\n}  // namespace strata::kernels')]
    body,n=re.subn(r'(block_topk_(?:reg_)?kernel(?:<[^;\n]*?>)?|block_topk_wide_kernel)<<<.*?>>>\([^;]*\);',lambda m:'record("'+m.group(1)+'");',body)
    assert n==(7 if variant=='fit17' else 9),(variant,n)
    p=OUT/(variant+'.cpp');p.write_text(prefix+body+main)
    for arch,defs in arches.items():
        exe=OUT/(variant+'.'+arch)
        subprocess.run([args.cxx,'-std=c++20','-O0',*defs,str(p),'-o',str(exe)],capture_output=True,check=True)
        bins[variant,arch]=exe
env_names=['STRATA_GFX906_TOPK_RADIX_STOP','STRATA_GFX906_TOPK_FIT17','STRATA_GFX906_TOPK_FIT17_TRACE','STRATA_GFX906_TOPK_REG','STRATA_TOPK_OLD','STRATA_QSA_CLUSTER','STRATA_TOPK_CAPACITY_GUARD','STRATA_TOPK_ACTIVE_ANY','STRATA_TOPK_STREAM']
def run(v,a,capacity,nq,active,stop='1',fit='1',reg='2',old=None):
    env=dict(os.environ)
    for n in env_names:env.pop(n,None)
    for n,val in [('STRATA_GFX906_TOPK_RADIX_STOP',stop),('STRATA_GFX906_TOPK_FIT17',fit),('STRATA_GFX906_TOPK_REG',reg),('STRATA_TOPK_OLD',old)]:
        if val is not None:env[n]=val
    return subprocess.run([str(bins[v,a]),str(capacity),str(nq),str(active)],env=env,check=True,capture_output=True,text=True).stdout.splitlines()
def old_names(xs):return [x.replace('<17, true, 3>','<17, true, 1>').replace('<TK_PER_MAX, true, 4>','<TK_PER_MAX, true, 2>') for x in xs]
cases=[]
def check(a,capacity,nq,active=0,stop='1',fit='1',reg='2',old=None):
    baseline=run('fit17',a,capacity,nq,active,stop,fit,reg,old)
    cand=run('radix-stop',a,capacity,nq,active,stop,fit,reg,old)
    assert old_names(cand)==baseline,(a,capacity,nq,active,stop,fit,reg,old,baseline,cand)
    expected=(a=='gfx906' and stop=='1' and fit=='1' and reg=='2' and old is None and active<=0 and 0<nq<=8 and 33792<capacity<=67584)
    changed=cand!=baseline
    assert changed==expected,(a,capacity,nq,active,stop,fit,reg,old,expected,cand)
    cases.append({'arch':a,'capacity':capacity,'nq':nq,'active':active,'stop':stop,'fit':fit,'reg':reg,'old':old,'changed':changed})
for a,c,n in itertools.product(arches,[17408,33792,33793,51202,67584,67585],[0,1,4,6,8,9]):check(a,c,n)
for stop in [None,'','0','00','1','01','1x','true','2','-1',' 1','1 ','1\n','+1']:
    for a in arches:check(a,51202,4,stop=stop)
for a,fit,reg,old,active in itertools.product(arches,[None,'0','1'],[None,'0','1','2'],[None,'1'],[0,50001]):check(a,51202,4,active,fit=fit,reg=reg,old=old)
# Compile the actual device-only guard arithmetic as a CPU predicate. No HIP body is compiled.
src=sources['radix-stop'];start=src.index('template <int PER, bool LONG_ONLY');end=src.index('    int32_t* out =',start)
part=src[start:end]
constants=part[part.index('    constexpr bool RADIX_STOP'):part.index('    __shared__')]
guard=part[part.index('    if constexpr (LONG_ONLY)'):]
guard=guard.replace('return;','return false;')
ghead='''#include <cstdint>
#include <iostream>
constexpr int TK_T=1024,TK_PER_MAX=66,kStepCount=4,kStepNKv=1,kStepNBid=2;
constexpr int64_t GFX906_TOPK_SHORT_CELLS=24576;
template<int PER,bool LONG_ONLY,int WINDOW_GUARD> bool g(const int32_t* steps,int nq){struct{int x;}gridDim{nq};
'''+constants+guard+'''return true;}
int main(){int32_t st[32]={}; long long tests=0;
 for(int nq: {1,4,6,8})for(int nkv: {24575,24576,24577,69631,69632,69633,200000,270332})for(int hot=0;hot<nq;++hot){
  for(int q=0;q<nq;++q){st[4*q+1]=q==hot?nkv:24577;st[4*q+2]=st[4*q+1]/4;}
  if(g<17,true,1>(st,nq)!=g<17,true,3>(st,nq))return 1;
  if(g<66,true,2>(st,nq)!=g<66,true,4>(st,nq))return 2;
  ++tests;
 }
 std::cout<<tests<<"\\n";
}
'''
p=OUT/'guards.cpp';p.write_text(ghead);exe=OUT/'guards'
subprocess.run([args.cxx,'-std=c++20','-O2',str(p),'-o',str(exe)],check=True,capture_output=True)
guard_cases=int(subprocess.check_output([str(exe)],text=True))
negative=[]
for spec in ['16,true,3','17,false,3','66,false,4','17,true,4','66,true,3','66,true,5']:
    p=OUT/'guard-invalid.cpp';p.write_text(ghead.replace('int main(){','int valid_main(){')+f'\nint main(){{int32_t st[32]={{}};return g<{spec}>(st,1);}}\n')
    r=subprocess.run([args.cxx,'-std=c++20','-fsyntax-only',str(p)],capture_output=True,text=True)
    assert r.returncode!=0 and 'static assertion failed' in r.stderr,spec;negative.append(spec)
result={'passed':True,'gpu_executed':False,'candidate_sha256':hashlib.sha256(sources['radix-stop'].encode()).hexdigest(),'dispatch_cases':len(cases),'guard_cases':guard_cases,'negative_template_cases':negative,'cases':cases}
(OUT/'dispatch-proof.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
