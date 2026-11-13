#!/usr/bin/env python3
"""Portable extracted CPU snippets and adversarial geometry; no capture inputs."""
import argparse, ctypes, difflib, hashlib, json, os, pathlib, random, re, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--output',required=True)
ap.add_argument('--cxx',default=os.environ.get('CXX','c++'))
args=ap.parse_args()
OUT=pathlib.Path(args.output).resolve();OUT.mkdir(parents=True,exist_ok=False)
def order_key(bits):
    if (bits & 0x7fffffff) > 0x7f800000:return 0
    if (bits & 0x7fffffff) == 0:bits=0
    return (~bits & 0xffffffff) if bits & 0x80000000 else bits | 0x80000000
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
base=(ROOT/'qsa_select.fit17.cu').read_text()
src=(ROOT/'qsa_select.radix-stop.cu').read_text()
assert sha(ROOT/'qsa_select.fit17.cu')=='5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a'
assert sha(ROOT/'qsa_select.radix-stop.cu')=='7036ef4eb84a145bbeeab46d0a735548cd719a656c5dcf6ec2a3ab0e3e089964'
patch=''.join(difflib.unified_diff(base.splitlines(True),src.splitlines(True),fromfile='a/src/kernels/cuda/qsa_select.cu',tofile='b/src/kernels/cuda/qsa_select.cu'))
assert patch==(ROOT/'qsa_select.radix-stop.cu.patch').read_text()
assert sha(ROOT/'qsa_select.radix-stop.cu.patch')=='39a29aa4c63c31dd2b8f7bf8714512dd5b8508186d128f8462919c1492710a23' 
start=src.index('template <int PER, bool LONG_ONLY = false, int WINDOW_GUARD = 0>')
end=src.index('\n#if !defined(__HIPCC__)',start)
k=src[start:end]
choose=k[k.index('            int cum = above'):k.index('        }\n        __syncthreads();',k.index('            int cum = above'))]
count=k[k.index('    int gt = 0, eq = 0;'):k.index('    int tot;')]
mask_decl=k[k.index('    const uint32_t key_mask ='):k.index('    const int64_t eq_budget =')]
emit=k[k.index('#pragma unroll',k.index('    int64_t eq_left = my_eq;')):k.rindex('\n}')]
keyfun=base[base.index('__device__ __forceinline__ uint32_t order_key'):base.index('\n\n__global__')]
keyfun=keyfun.replace('__device__ __forceinline__ ','')
cpp='''#include <cstdint>
#include <cstring>
#include <algorithm>
constexpr int R=4;
uint32_t __float_as_uint(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
'''+keyfun+'''
extern "C" uint32_t cpp_key(uint32_t bits){float f;std::memcpy(&f,&bits,4);return order_key(f);}
template<bool RADIX_STOP> void choose_impl(const int* h,int64_t width,int above,int shift,uint32_t* d_out,int* a_out){
 int hist[1][256];std::memcpy(hist[0],h,1024);int s_digit=0;int s_above=0;
'''+choose+'''
 *d_out=s_digit;*a_out=s_above;
}
extern "C" void cpp_choose(const int* h,int64_t width,int above,int shift,int mode,uint32_t* d,int* a){
 if(mode)choose_impl<true>(h,width,above,shift,d,a);else choose_impl<false>(h,width,above,shift,d,a);
}
template<int PER,bool RADIX_STOP> void count_impl(uint32_t* key,int64_t b0,int64_t b1,int64_t n_bid,int64_t n_kv,uint32_t thr,bool stopped,int* result){
 auto weight=[&](int64_t b)->int{return b<n_bid?R:(int)(n_kv-n_bid*R);};
 int s_digit=stopped?256:0;
'''+mask_decl+'''
'''+count+'''
 result[0]=gt;result[1]=eq;
}
extern "C" void cpp_count(uint32_t* key,int64_t b0,int64_t b1,int64_t n_bid,int64_t n_kv,uint32_t thr,int stopped,int per,int mode,int* result){
 if(per==17){if(mode)count_impl<17,true>(key,b0,b1,n_bid,n_kv,thr,stopped,result);else count_impl<17,false>(key,b0,b1,n_bid,n_kv,thr,stopped,result);}
 else {if(mode)count_impl<66,true>(key,b0,b1,n_bid,n_kv,thr,stopped,result);else count_impl<66,false>(key,b0,b1,n_bid,n_kv,thr,stopped,result);}
}
template<int PER> int emit_impl(const uint32_t* key,int64_t b0,int64_t b1,int64_t n_bid,int64_t n_kv,uint32_t thr,int64_t eq_left,int32_t* out){
 auto weight=[&](int64_t b)->int{return b<n_bid?R:(int)(n_kv-n_bid*R);};int64_t wpos=0;
'''+emit+'''
 return (int)wpos;
}
extern "C" int cpp_emit(const uint32_t* key,int64_t b0,int64_t b1,int64_t n_bid,int64_t n_kv,uint32_t thr,int64_t eq_left,int per,int32_t* out){
 return per==17?emit_impl<17>(key,b0,b1,n_bid,n_kv,thr,eq_left,out):emit_impl<66>(key,b0,b1,n_bid,n_kv,thr,eq_left,out);
}
'''
(OUT/'extracted-cpu.cpp').write_text(cpp)
subprocess.run([args.cxx,'-std=c++17','-O2','-fPIC','-shared',str(OUT/'extracted-cpu.cpp'),'-o',str(OUT/'extracted-cpu.so')],check=True)
lib=ctypes.CDLL(str(OUT/'extracted-cpu.so'))
I=ctypes.c_int;U=ctypes.c_uint32;L=ctypes.c_int64;P=ctypes.POINTER
lib.cpp_key.argtypes=[U];lib.cpp_key.restype=U
lib.cpp_choose.argtypes=[P(I),L,I,I,I,P(U),P(I)]
lib.cpp_count.argtypes=[P(U),L,L,L,L,U,I,I,I,P(I)]
lib.cpp_emit.argtypes=[P(U),L,L,L,L,U,L,I,P(I)];lib.cpp_emit.restype=I
rng=random.Random(90620261009)
special=[0,0x80000000,0x7f800000,0xff800000,0x7fc00000,0xffc00000,0x7f800001,0xff800001,1,0x80000001,0x007fffff,0x807fffff,0x7f7fffff,0xff7fffff]
raw=special+[rng.getrandbits(32) for _ in range(100000)]
for bits in raw: assert lib.cpp_key(bits)==order_key(bits),(hex(bits),hex(lib.cpp_key(bits)),hex(order_key(bits)))
def oracle(keys,nkv,width):
    weights=[4]*(nkv//4)+[nkv%4]
    out=[]
    for b in sorted(range(len(keys)),key=lambda b:(-keys[b],b)):
        take=min(weights[b],width-len(out));out.extend(range(4*b,4*b+take))
        if len(out)==width:break
    return sorted(out)
def run(keys,nkv,width,mode,force_per=None):
    nb=len(keys);nbid=nkv//4;assert nb==nbid+1 and 0<=width<=nkv
    if nkv<=width:return list(range(nkv)),{'passes':0,'stopped':False,'positive_boundary_blocks':0}
    per=(nb+1023)//1024;PER=force_per or (17 if per<=17 else 66);assert per<=PER
    weights=[4]*nbid+[nkv%4];prefix=above=0;recover=False;passes=[]
    for shift in (24,16,8,0):
        hist=[0]*256;mask=0 if shift==24 else (0xffffffff<<(shift+8))&0xffffffff
        for key,w in zip(keys,weights):
            if w and key&mask==prefix&mask:hist[(key>>shift)&255]+=w
        d=U();a=I();lib.cpp_choose((I*256)(*hist),width,above,shift,mode,ctypes.byref(d),ctypes.byref(a))
        digit=d.value&255;prefix|=digit<<shift;above=a.value;recover=bool(d.value&256)
        assert above<=width<=above+hist[digit]
        passes.append({'shift':shift,'mass':hist[digit],'above':above,'need':width-above})
        if recover:break
    chunks=[];positive_boundary_blocks=0
    for t in range(min(1024,(nb+per-1)//per)):
        b0=t*per;b1=min(b0+per,nb);arr=(U*PER)(*keys[b0:b1]);counts=(I*2)()
        lib.cpp_count(arr,b0,b1,nbid,nkv,prefix,recover,PER,mode,counts)
        for j,b in enumerate(range(b0,b1)):
            assert arr[j]==(keys[b] & (0xffffff00 if mode and recover and weights[b]>0 else 0xffffffff))
        if recover:positive_boundary_blocks+=sum(w>0 and prefix<=key<=prefix|255 for key,w in zip(keys[b0:b1],weights[b0:b1]))
        chunks.append((arr,b0,b1,counts[0],counts[1]))
    assert not recover or positive_boundary_blocks==1
    thr=prefix;budget=width-above;eq_before=0;output=[]
    for arr,b0,b1,gt,eq in chunks:
        myeq=max(0,min(eq,budget-eq_before));eq_before+=eq;dest=(I*(PER*4))()
        n=lib.cpp_emit(arr,b0,b1,nbid,nkv,thr,myeq,PER,dest)
        assert n==gt+myeq
        output.extend(dest[:n])
    assert len(output)==width
    return output,{'passes':len(passes),'stopped':recover,'positive_boundary_blocks':positive_boundary_blocks,'radix':passes,'threshold':thr}
def check(keys,nkv,width,label,PER=None):
    expected=oracle(keys,nkv,width)
    for mode in (0,1):
        out,stats=run(keys,nkv,width,mode,PER);assert out==expected,(label,mode,nkv,width,stats)
    return stats
cases=0
for nkv in range(1,90):
    nb=nkv//4+1
    for style in range(6):
        keys=([0]*nb if style==0 else [0x80000000]*nb if style==1 else
              [0xbf800000+(i%3) for i in range(nb)] if style==2 else
              [order_key(special[i%len(special)]) for i in range(nb)] if style==3 else
              [0xabcdef00+rng.randrange(256) for _ in range(nb)] if style==4 else
              [rng.getrandbits(32) for _ in range(nb)])
        for width in sorted(set((1,min(3,nkv),min(4,nkv),max(1,nkv//2),nkv-1,nkv))):
            check(keys,nkv,width,'small');cases+=1
for nb in [1023,1024,1025,17408,17409,33792,33793,50001,50002,67584]:
    for tail in range(4):
        nkv=4*(nb-1)+tail;keys=[order_key(rng.getrandbits(32)) for _ in range(nb)]
        check(keys,nkv,min(2051,nkv),'ownership-edge');cases+=1
result={'passed':True,'gpu_executed':False,'timing_claim':False,'source_only':True,
        'cpu_compiler':subprocess.check_output([args.cxx,'--version'],text=True).splitlines()[0],
        'source':{p:sha(ROOT/p) for p in ['qsa_select.fit17.cu','qsa_select.radix-stop.cu','qsa_select.radix-stop.cu.patch']},
        'extracted_cpp_sha256':sha(OUT/'extracted-cpu.cpp'),
        'extracted_sections':['actual order_key','actual weighted digit/stop decision','actual in-place normalization + per-thread gt/eq count','actual output emission'],
        'modeled_sections':['histogram aggregation','barrier happens-before remains source-only','block_excl_scan as integer prefix sums'],
        'raw_float_key_cases':len(raw),'adversarial_source_snippet_cases':cases,
        'capture_replay_in_this_run':False,
        'prior_capture_evidence':'provenance/cpu-source-proof.json; frozen controller source evidence only, not rerun here',
        'limitations':['GPU compiler, execution, barrier behavior, graph parity, and performance are not tested by this CPU suite.']}
(OUT/'source-proof.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
