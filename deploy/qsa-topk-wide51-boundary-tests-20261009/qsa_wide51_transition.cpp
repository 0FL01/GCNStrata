// Isolated WIDE51 correctness-only graph transition driver; no timing and no new selector.
// Route labels are source-derived expectations, not observed device-arm evidence.
// Usage: qsa_wide51_transition QSTRAN01_FILE [DEVICE=0]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#ifndef STRATA_HIP_GFX906
#error "Requires qualified gfx906 selector archive and HIP compatibility headers"
#endif
#if !defined(__HIPCC__) && !defined(__HIP_PLATFORM_AMD__)
#error "HIP gfx906 build required"
#endif
namespace k = strata::kernels;
namespace {
constexpr int32_t GUARD=0x13579bdf;
void require(bool b,const char* why) { if(!b) throw std::runtime_error(why); }
void ck(cudaError_t e,const char* where) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
struct Case { std::string name; std::vector<int32_t> steps,expected; std::vector<uint32_t> scores; };
struct Fixture { uint32_t nq,score_cap,out_cap,guards; std::vector<Case> cases; };
Fixture read_fixture(const char* path) {
    FILE* f=std::fopen(path,"rb"); require(f!=nullptr,"cannot open fixture");
    try {
        auto read=[&](void* p,size_t n){ require(std::fread(p,1,n,f)==n,"truncated fixture"); };
        char magic[8]; uint32_t h[8];read(magic,8);read(h,sizeof h);
        const uint32_t endian=0x01020304;
        require(*reinterpret_cast<const unsigned char*>(&endian)==4,"little-endian host required");
        require(std::memcmp(magic,"QSTRAN01",8)==0 && h[0]==1 && h[1]==endian,"unsupported fixture");
        require((h[2]==1 || h[2]==4 || h[2]==6 || h[2]==8 || h[2]==9) && (h[3]==52223 || h[3]==52224 || h[3]==52225) && h[4]==2080 && h[5]==4 &&
                h[6]>0 && h[6]<=100 && h[7]==16,"unsupported fixture dimensions");
        Fixture a{h[2],h[3],h[4],h[7],{}};
        const auto s=k::qsa_real_shapes();
        require(k::kStepCount==4 && s.idx_block==4 && s.idx_top_k==2048,"unexpected production geometry");
        for(uint32_t ci=0;ci<h[6];++ci) {
            Case c;uint32_t n;read(&n,4);require(n>0 && n<128,"bad case name");
            c.name.resize(n);read(c.name.data(),n);
            c.steps.assign(a.nq*k::kStepCount+a.guards,GUARD);read(c.steps.data(),a.nq*k::kStepCount*4);
            c.scores.assign(a.nq*a.score_cap+a.guards,0x7f800000u);
            std::fill(c.scores.end()-a.guards,c.scores.end(),0xdeadbeefu);
            c.expected.assign(a.nq*a.out_cap+a.guards,-1);
            std::fill(c.expected.end()-a.guards,c.expected.end(),GUARD);
            for(uint32_t t=0;t<a.nq;++t) {
                const int32_t* st=c.steps.data()+t*k::kStepCount;uint32_t lens[2];read(lens,8);
                require(st[k::kStepNKv]>=1 && st[k::kStepNKv]<=262144 && st[k::kStepNBid]+1<=int32_t(a.score_cap) &&
                        st[k::kStepPos]==st[k::kStepNKv]-1 && st[k::kStepNBid]==st[k::kStepNKv]/4 &&
                        st[k::kStepWidth]==k::qsa_selection_width(st[k::kStepNKv],s),"invalid step row");
                require(lens[0]==uint32_t(st[k::kStepNBid]+1) && lens[0]<=a.score_cap &&
                        lens[1]==uint32_t(st[k::kStepWidth]) && lens[1]<=a.out_cap,"invalid packed lengths");
                read(c.scores.data()+t*a.score_cap,lens[0]*4);read(c.expected.data()+t*a.out_cap,lens[1]*4);
                const auto* ids=c.expected.data()+t*a.out_cap;
                for(uint32_t j=0;j<lens[1];++j)
                    require(ids[j]>=0 && ids[j]<st[k::kStepNKv] && (j==0 || ids[j]>ids[j-1]),"invalid CPU IDs");
            }
            a.cases.push_back(std::move(c));
        }
        require(std::fgetc(f)==EOF && !std::ferror(f),"trailing data or read error");
        std::fclose(f);return a;
    } catch(...) {std::fclose(f);throw;}
}
struct Resources {
    float* scores=nullptr;int32_t *steps=nullptr,*out=nullptr,*ref=nullptr;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    ~Resources() {
        if(stream) cudaStreamSynchronize(stream);
        if(exec) cudaGraphExecDestroy(exec);if(graph) cudaGraphDestroy(graph);
        if(scores) cudaFree(scores);if(steps) cudaFree(steps);if(out) cudaFree(out);if(ref) cudaFree(ref);
        if(stream) cudaStreamDestroy(stream);
    }
};
}
int main(int argc,char** argv) {
    try {
        require(argc==2 || argc==3,"usage: qsa_wide51_transition FIXTURE [DEVICE=0]");
        const auto a=read_fixture(argv[1]);const int device=argc==3?std::stoi(argv[2]):0;
        const char* reg=std::getenv("STRATA_GFX906_TOPK_REG");const char* trace=std::getenv("STRATA_TOPK_TRACE");
        const char* mid=std::getenv("STRATA_GFX906_TOPK_FIT17");
        const char* wide=std::getenv("STRATA_GFX906_TOPK_WIDE51");
        const bool fit17_on=mid && std::strcmp(mid,"1")==0;
        const bool wide51_on=wide && std::strcmp(wide,"1")==0;
        require(reg && std::strcmp(reg,"2")==0,"require TOPK_REG=2");
        require(!trace && !std::getenv("STRATA_GFX906_TOPK_FIT17_TRACE"),"all selector traces must be absent; old topology text is not WIDE51 evidence");
        require(!std::getenv("STRATA_TOPK_OLD"),"STRATA_TOPK_OLD must be absent");
        ck(cudaSetDevice(device),"set device");cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,device),"properties");
        require(std::strncmp(prop.gcnArchName,"gfx906",6)==0 &&
                (prop.gcnArchName[6]=='\0' || prop.gcnArchName[6]==':'),"exact gfx906 required");
        require(prop.warpSize==64,"wave64 required");
        Resources r;ck(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking),"stream");
        const size_t sb=a.cases[0].scores.size()*4,tb=a.cases[0].steps.size()*4,ob=a.cases[0].expected.size()*4;
        ck(cudaMalloc((void**)&r.scores,sb),"allocate scores");ck(cudaMalloc((void**)&r.steps,tb),"allocate steps");
        ck(cudaMalloc((void**)&r.out,ob),"allocate output");ck(cudaMalloc((void**)&r.ref,ob),"allocate reference");
        std::vector<int32_t> init(a.cases[0].expected.size(),-1);std::fill(init.end()-a.guards,init.end(),GUARD);
        const auto s=k::qsa_real_shapes();
        auto dispatch=[&]{ k::qsa_block_topk(r.scores,r.steps,a.nq,a.score_cap,a.out_cap,s,r.out,r.stream); };
        auto upload=[&](const Case& c) {
            // Explicitly outside graph/capture. Addresses and capacities never change.
            ck(cudaMemcpy(r.scores,c.scores.data(),sb,cudaMemcpyHostToDevice),"update scores");
            ck(cudaMemcpy(r.steps,c.steps.data(),tb,cudaMemcpyHostToDevice),"update steps");
            ck(cudaMemcpy(r.out,init.data(),ob,cudaMemcpyHostToDevice),"reset output and padding");
            ck(cudaMemcpy(r.ref,init.data(),ob,cudaMemcpyHostToDevice),"reset reference and padding");
        };
        auto check=[&](const Case& c,const int32_t* ptr,const char* label) {
            std::vector<int32_t> got(c.expected.size());ck(cudaMemcpy(got.data(),ptr,ob,cudaMemcpyDeviceToHost),label);
            if(got!=c.expected) {
                size_t at=std::mismatch(got.begin(),got.end(),c.expected.begin()).first-got.begin();
                throw std::runtime_error(c.name+" "+label+" ID/padding/canary mismatch at "+std::to_string(at)+
                    " got="+std::to_string(got[at])+" expected="+std::to_string(c.expected[at]));
            }
        };
        // Exercise lazy host configuration before capture; this is not a timed run.
        upload(a.cases[0]);dispatch();ck(cudaStreamSynchronize(r.stream),"warmup sync");check(a.cases[0],r.out,"warmup");
        ck(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeGlobal),"begin capture");
        dispatch(); // exactly one selector call captured, NO count pointer
        ck(cudaStreamEndCapture(r.stream,&r.graph),"end capture");
        ck(cudaGraphInstantiate(&r.exec,r.graph,nullptr,nullptr,0),"instantiate once");
        for(size_t ci=0;ci<a.cases.size();++ci) {
            const auto& c=a.cases[ci];upload(c);
            ck(cudaGraphLaunch(r.exec,r.stream),"replay same graph");ck(cudaStreamSynchronize(r.stream),"graph sync");
            check(c,r.out,"graph vs CPU");
            // Verify captured selector input bytes BEFORE running the separate reference.
            std::vector<uint32_t> scores(c.scores.size());std::vector<int32_t> steps(c.steps.size());
            ck(cudaMemcpy(scores.data(),r.scores,sb,cudaMemcpyDeviceToHost),"read graph scores");
            ck(cudaMemcpy(steps.data(),r.steps,tb,cudaMemcpyDeviceToHost),"read graph steps");
            require(scores==c.scores && steps==c.steps,"graph selector modified input bytes/guards");
            k::qsa_block_topk_ref(r.scores,r.steps,a.nq,a.score_cap,a.out_cap,s,r.ref,r.stream);
            ck(cudaStreamSynchronize(r.stream),"reference sync");check(c,r.ref,"reference vs CPU");
            ck(cudaMemcpy(scores.data(),r.scores,sb,cudaMemcpyDeviceToHost),"read reference scores");
            ck(cudaMemcpy(steps.data(),r.steps,tb,cudaMemcpyDeviceToHost),"read reference steps");
            require(scores==c.scores && steps==c.steps,"reference modified input bytes/guards");
            int32_t max_blocks=0;for(uint32_t t=0;t<a.nq;++t) max_blocks=std::max(max_blocks,c.steps[t*k::kStepCount+k::kStepNBid]+1);
            const int32_t last_nkv=c.steps[(a.nq-1)*k::kStepCount+k::kStepNKv];
            const char* intended=a.nq>8?"fallback_ref":last_nkv<=24576?"guarded_ref":
                !fit17_on?"guarded_reg66_legacy":max_blocks<=17408?"guarded_reg17":
                (wide51_on && a.score_cap<=52224)?"guarded_reg51":"guarded_reg66";
            std::printf("CASE_PASS nq=%u case=%s last_nkv=%d max_active_blocks=%d source_expected_route=%s route_observed=0 IDs=graph=reference=CPU padding=end_canaries=inputs_unchanged=PASS\n",
                        a.nq,c.name.c_str(),last_nkv,max_blocks,intended);std::fflush(stdout);
        }
        std::printf("TRANSITION_PASS nq=%u capacity=%u cases=%zu graph_instantiations=1 constant_addresses=1 constant_capacities=1 fit17=%s wide51=%s route_observed=0 timing_claim=0\n",
                    a.nq,a.score_cap,a.cases.size(),mid?mid:"<unset>",wide?wide:"<unset>");
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
