// Replay unchanged production top-k on one QSDUMP01 fixture. No new selection kernel.
// Usage: qsa_topk_fixture_replay FILE [REPLAY_DEVICE=header] [BATCH=64] [SAMPLES=7]
// Requires STRATA_GFX906_TOPK_REG=2 and STRATA_TOPK_TRACE=1; OLD must be absent.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef STRATA_HIP_GFX906
#error "Build against the qualified gfx906 production selector archive"
#endif
namespace k = strata::kernels;
namespace {
constexpr int32_t END_GUARD = 0x13579bdf;
void require(bool b, const char* why) { if (!b) throw std::runtime_error(why); }
void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
struct Input {
    int64_t h[19]{};
    std::vector<int32_t> steps, expected;
    std::vector<uint32_t> scores;
};
Input read_input(const char* path) {
    std::FILE* f=std::fopen(path,"rb"); require(f!=nullptr,"cannot open fixture");
    try {
        auto read=[&](void* p,size_t n){require(std::fread(p,1,n,f)==n,"truncated fixture");};
        Input a; char magic[8]; read(magic,8); require(std::memcmp(magic,"QSDUMP01",8)==0,"bad magic");
        read(a.h,sizeof a.h); const auto* h=a.h;
        const uint32_t end=0x01020304u;
        require(*reinterpret_cast<const unsigned char*>(&end)==4,"little-endian host required");
        require(h[0]==1 && h[1]==end,"unsupported format version/endian");
        require(h[2]>=0 && 0<=h[3] && h[3]<=h[5] && h[5]<h[4] && h[4]<=1024 && h[6]>=65536,
                "invalid stage/position");
        require(2<=h[7] && h[7]<=h[8] && h[8]<=8 && 1<=h[9] && h[9]<=1000000 &&
                1<=h[10] && h[10]<=1000000,"invalid dimensions");
        const k::QsaShapes s=k::qsa_real_shapes();
        require(h[11]==k::kStepCount && h[12]==s.idx_block && h[13]==s.idx_top_k &&
                h[14]==s.idx_dim && h[15]==s.idx_n_head,"unsupported QSA geometry");
        require(h[16]==h[9]*4 && h[17]==h[10]*4 && h[18]==h[11]*4,"invalid source strides");
        require(h[10]>=k::qsa_selection_width(k::kTopkMaxCells,s),"invalid output capacity");
        a.steps.resize((size_t)h[7]*h[11]); read(a.steps.data(),a.steps.size()*4);
        // Full production stride. Poison every inactive score with +Inf, never consume it.
        a.scores.assign((size_t)h[7]*h[9]+16,0x7f800000u);
        a.expected.assign((size_t)h[7]*h[10]+16,-1);
        std::fill(a.expected.end()-16,a.expected.end(),END_GUARD);
        for(int64_t t=0;t<h[7];++t) {
            const int32_t* st=a.steps.data()+t*h[11]; int64_t lens[2]; read(lens,sizeof lens);
            require(st[k::kStepPos]==h[6]+t && st[k::kStepNKv]==h[6]+t+1 &&
                    st[k::kStepNBid]==st[k::kStepNKv]/s.idx_block && st[k::kStepNBid]>=0 &&
                    (int64_t)st[k::kStepNBid]+1<=h[9] &&
                    st[k::kStepWidth]==k::qsa_selection_width(st[k::kStepNKv],s) &&
                    st[k::kStepWidth]>0 && st[k::kStepWidth]<=h[10],"inconsistent steps");
            require(lens[0]==(int64_t)st[k::kStepNBid]+1 && lens[1]==st[k::kStepWidth],"invalid packed row lengths");
            read(a.scores.data()+t*h[9],(size_t)lens[0]*4);
            read(a.expected.data()+t*h[10],(size_t)lens[1]*4);
            const int32_t* ids=a.expected.data()+t*h[10];
            for(int64_t j=0;j<lens[1];++j)
                require(ids[j]>=0 && ids[j]<st[k::kStepNKv] && (j==0 || ids[j]>ids[j-1]),"invalid captured IDs");
        }
        require(std::fgetc(f)==EOF && !std::ferror(f),"trailing fixture data/read error");
        std::fclose(f); return a;
    } catch(...) { std::fclose(f); throw; }
}
struct Resources {
    float* scores=nullptr; int32_t *steps=nullptr,*out=nullptr,*ref=nullptr;
    cudaStream_t stream=nullptr; cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
    cudaEvent_t start=nullptr,stop=nullptr;
    ~Resources() {
        if(stream) cudaStreamSynchronize(stream);
        if(exec) cudaGraphExecDestroy(exec); if(graph) cudaGraphDestroy(graph);
        if(start) cudaEventDestroy(start); if(stop) cudaEventDestroy(stop);
        if(scores) cudaFree(scores); if(steps) cudaFree(steps); if(out) cudaFree(out); if(ref) cudaFree(ref);
        if(stream) cudaStreamDestroy(stream);
    }
};
}
int main(int argc,char** argv) {
    try {
        require(argc>=2 && argc<=5,"usage: qsa_topk_fixture_replay FILE [DEVICE] [BATCH=64] [SAMPLES=7]");
        const Input a=read_input(argv[1]); const auto* h=a.h;
        const char* reg=std::getenv("STRATA_GFX906_TOPK_REG");
        const char* trace=std::getenv("STRATA_TOPK_TRACE");
        require(reg && std::strcmp(reg,"2")==0,"require exact STRATA_GFX906_TOPK_REG=2");
        require(trace && std::strcmp(trace,"1")==0,"require STRATA_TOPK_TRACE=1 to retain actual route evidence");
        require(std::getenv("STRATA_TOPK_OLD")==nullptr,"STRATA_TOPK_OLD must be absent (even =0 overrides)");
        const int device=argc>2?std::stoi(argv[2]):(int)h[2];
        const int batch=argc>3?std::stoi(argv[3]):64, samples=argc>4?std::stoi(argv[4]):7;
        require(device>=0 && 1<=batch && batch<=1024 && 1<=samples && samples<=100,"invalid replay arguments");
        ck(cudaSetDevice(device),"set device"); cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop,device),"device props");
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
        require(std::strncmp(prop.gcnArchName,"gfx906",6)==0 &&
                (prop.gcnArchName[6]=='\0' || prop.gcnArchName[6]==':'),"exact gfx906 device required");
#endif
        require(prop.warpSize==64,"wave64 device required");
        Resources r; ck(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking),"stream");
        const size_t score_bytes=a.scores.size()*4, step_bytes=a.steps.size()*4, out_bytes=a.expected.size()*4;
        ck(cudaMalloc((void**)&r.scores,score_bytes),"scores alloc");
        ck(cudaMalloc((void**)&r.steps,step_bytes),"steps alloc");
        ck(cudaMalloc((void**)&r.out,out_bytes),"output alloc");
        ck(cudaMalloc((void**)&r.ref,out_bytes),"reference output alloc");
        ck(cudaMemcpy(r.scores,a.scores.data(),score_bytes,cudaMemcpyHostToDevice),"upload scores");
        ck(cudaMemcpy(r.steps,a.steps.data(),step_bytes,cudaMemcpyHostToDevice),"upload steps");
        std::vector<int32_t> init(a.expected.size(),-1);
        std::fill(init.end()-16,init.end(),END_GUARD);
        ck(cudaMemcpy(r.out,init.data(),out_bytes,cudaMemcpyHostToDevice),"initialize output guards");
        ck(cudaMemcpy(r.ref,init.data(),out_bytes,cudaMemcpyHostToDevice),"initialize reference guards");
        const k::QsaShapes s=k::qsa_real_shapes();
        auto dispatch=[&]{k::qsa_block_topk(r.scores,r.steps,h[7],h[9],h[10],s,r.out,r.stream);}; // NO active count!
        auto check_ids=[&](const int32_t* d,const char* what) {
            std::vector<int32_t> got(a.expected.size()); ck(cudaMemcpy(got.data(),d,out_bytes,cudaMemcpyDeviceToHost),what);
            if(got!=a.expected) {
                const size_t at=(size_t)(std::mismatch(got.begin(),got.end(),a.expected.begin()).first-got.begin());
                throw std::runtime_error(std::string(what)+": ID/padding/canary mismatch at "+std::to_string(at));
            }
        };
        // First parity, including initialized unused output suffixes and end canaries.
        dispatch();
        k::qsa_block_topk_ref(r.scores,r.steps,h[7],h[9],h[10],s,r.ref,r.stream);
        ck(cudaStreamSynchronize(r.stream),"initial parity sync");
        check_ids(r.out,"production vs captured"); check_ids(r.ref,"reference vs captured");
        std::printf("PARITY_PASS production=reference=captured; padding/end_canaries=PASS\n"); std::fflush(stdout);
        ck(cudaEventCreate(&r.start),"start event"); ck(cudaEventCreate(&r.stop),"stop event");
        // A graph of repeated existing calls avoids host launch gaps between measured calls.
        // Nothing allocates, copies to host or synchronizes inside this capture.
        ck(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeGlobal),"begin capture");
        for(int i=0;i<batch;++i) dispatch();
        ck(cudaStreamEndCapture(r.stream,&r.graph),"end capture");
        ck(cudaGraphInstantiate(&r.exec,r.graph,nullptr,nullptr,0),"instantiate");
        ck(cudaMemcpy(r.out,init.data(),out_bytes,cudaMemcpyHostToDevice),"reset before graph parity");
        for(int i=0;i<4;++i) ck(cudaGraphLaunch(r.exec,r.stream),"warmup graph");
        ck(cudaStreamSynchronize(r.stream),"warmup sync"); check_ids(r.out,"graph vs captured");
        std::vector<float> ms;
        for(int i=0;i<samples;++i) {
            ck(cudaEventRecord(r.start,r.stream),"start record");
            ck(cudaGraphLaunch(r.exec,r.stream),"timed graph");
            ck(cudaEventRecord(r.stop,r.stream),"stop record"); ck(cudaEventSynchronize(r.stop),"event wait");
            float dt=0; ck(cudaEventElapsedTime(&dt,r.start,r.stop),"elapsed");
            require(dt>0,"nonpositive timing"); ms.push_back(dt/(float)batch);
        }
        check_ids(r.out,"post-timing vs captured");
        std::vector<uint32_t> scores_after(a.scores.size());
        std::vector<int32_t> steps_after(a.steps.size());
        ck(cudaMemcpy(scores_after.data(),r.scores,score_bytes,cudaMemcpyDeviceToHost),"check input scores");
        ck(cudaMemcpy(steps_after.data(),r.steps,step_bytes,cudaMemcpyDeviceToHost),"check input steps");
        require(scores_after==a.scores && steps_after==a.steps,"selector modified input bytes");
        auto sorted=ms; std::sort(sorted.begin(),sorted.end());
        const float med=sorted.size()%2?sorted[sorted.size()/2]:(sorted[sorted.size()/2-1]+sorted[sorted.size()/2])*0.5f;
        std::printf("META captured_device=%lld replay_device=%d layer=%lld stage=[%lld,%lld) pos0=%lld nq=%lld capacity_blocks=%lld cap=%lld batch=%d samples=%d device=%s\n",
                    (long long)h[2],device,(long long)h[5],(long long)h[3],(long long)h[4],(long long)h[6],
                    (long long)h[7],(long long)h[9],(long long)h[10],batch,samples,prop.name);
        std::printf("SAMPLES_MS_PER_CALL"); for(float x:ms) std::printf(" %.9f",x); std::printf("\n");
        std::printf("RESULT median_ms_per_call=%.9f min=%.9f max=%.9f input_bytes_unchanged=PASS graph_ids=PASS\n",med,sorted.front(),sorted.back());
        std::printf("SCOPE warm-cache isolated existing selector including guarded no-op launch; no scorer, no whole-model timing, one captured layer\n");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
