
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
void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells), else (CUDA) the same threads
    // reading them from memory; the same ids. STRATA_TOPK_OLD=1: the original kernel
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
#if !defined(__HIPCC__)
    // sm_90+: a cluster of CL_N CTAs per query for the calls of a few queries (decode windows); the same ids.
    // STRATA_QSA_CLUSTER=0: the one-CTA kernels below
    static const bool cluster = [] {
        const char* v = std::getenv("STRATA_QSA_CLUSTER");
        return !v || std::atoi(v) != 0;
    }();
    if (cluster && !old && nq <= CL_MAXQ && qsa_block_topk_cluster(scores, steps, nq, max_blocks, cap, s, ids, stream))
        return;
#endif
    // the blocks a query can have: the call's active count when the caller knows it (the prompt path), else the capacity.
    // Decode (no count) keeps the capacity rule; gfx906 can opt into the guarded 66-key path below.
#if defined(__HIPCC__)
    const bool counted = active_blocks > 0;
#else
    // Turing: --max-context 262144 makes the stride 65538, even while a 131K prompt's active blocks fit in
    // TK_T * TK_PER registers. Use the prefill bound on sm_75, keeping max_blocks as the score-row stride.
    // Other CUDA devices keep 0.1.32's capacity rule: #337 was measured on RDNA4, and RTX 5070 64K prompts were
    // 1-3% slower. Decode/captured graphs omit the bound and never query the device here.
    static const bool capacity_guard = std::getenv("STRATA_TOPK_CAPACITY_GUARD") != nullptr;
    // STRATA_TOPK_ACTIVE_ANY=1 (tests): the Turing dispatch on any CUDA card, so qsa_topk_active_parity checks it
    // on whatever card runs the tests (the kernels are the same on every architecture)
    static const bool any_card = [] { const char* v = std::getenv("STRATA_TOPK_ACTIVE_ANY"); return v && v[0] == '1'; }();
    const bool counted = !capacity_guard && active_blocks > 0 && active_blocks <= max_blocks &&
                         (any_card || topk_active_turing_device());
#endif
    const int64_t reach = counted && active_blocks < max_blocks ? active_blocks : max_blocks;
#if defined(STRATA_HIP_GFX906)
    // Opt-in only: retain the existing register kernel at larger decode capacities. The device-side
    // guard uses the current context, not the capacity or a host value frozen during graph capture.
    static const bool decode_reg66 = [] {
        const char* v = std::getenv("STRATA_GFX906_TOPK_REG");
        return v && std::atoi(v) == 2;
    }();
    const bool extended_decode = decode_reg66 && !counted && nq <= 8;
#else
    const bool extended_decode = false;
#endif
    const int64_t fit = (int64_t) TK_T * ((counted || extended_decode) ? TK_PER_MAX : TK_PER);
#if defined(__HIPCC__)
    constexpr int64_t kRegMinBlocks = 7168;   // gfx1201: below ~28K cells the 1,024-thread kernel's fixed cost loses to the ref
    const bool too_small = counted && reach < kRegMinBlocks;
    if (old || too_small || reach > fit) {
#else
    constexpr int64_t kWideMinBlocks = 4608;   // RTX 3060: a prompt batch below ~18K cells is faster on the ref
    const bool too_small = reach > fit && active_blocks > 0 && active_blocks < kWideMinBlocks;
    if (old || too_small) {
#endif
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
#if defined(STRATA_HIP_GFX906)
    // Both launches read current device steps: capture is safe across growth and restores.
    // Use the final query for the whole window, so a threshold crossing does not run both bodies.
    // Each guard precedes every barrier. Exactly one arm writes all rows.
    if (extended_decode && reach > (int64_t) TK_T * TK_PER) {
        record("block_topk_kernel<true>");
        // Opt-in exact-fit path; keep the existing register66 route by default.
        static const bool fit17 = [] {
            const char* v = std::getenv("STRATA_GFX906_TOPK_FIT17");
            return v && std::strcmp(v, "1") == 0;
        }();
        if (fit17) {
            static const bool trace_fit17 = [] {
                const char* v = std::getenv("STRATA_GFX906_TOPK_FIT17_TRACE");
                return v && std::strcmp(v, "1") == 0;
            }();
            static thread_local bool traced_fit17 = false;
            if (trace_fit17 && !traced_fit17) {
                traced_fit17 = true;
                std::fprintf(stderr, "strata gfx906 topk fit17: enabled=1 topology=guarded-ref256/reg17/reg66 "
                                     "max_blocks=17408 nq=%lld capacity_blocks=%lld cap=%lld "
                                     "(device steps select the per-window arm)\n",
                             (long long) nq, (long long) max_blocks, (long long) cap);
            }
            record("block_topk_reg_kernel<17, true, 1>");
            record("block_topk_reg_kernel<TK_PER_MAX, true, 2>");
        } else {
            record("block_topk_reg_kernel<TK_PER_MAX, true>");
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk guarded: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
#endif
#if !defined(__HIPCC__)
    // Turing prefill (the `counted` bound) above ~90K cells (22,528 blocks): the wide kernel, which reads the keys
    // coalesced, instead of the register kernel's uncoalesced per-thread runs (PR #743: 131K, 1.03 -> 0.75 ms); the
    // same ids.  STRATA_TOPK_STREAM=0 restores the register kernel there.
    static const bool turing_wide = [] {
        const char* v = std::getenv("STRATA_TOPK_STREAM");
        return v == nullptr || v[0] != '0';
    }();
    if (reach > fit || (turing_wide && counted && reach > 22528))   // past the register kernel's reach, or Turing's band
        record("block_topk_wide_kernel");
    else
#endif
    if (reach <= (int64_t) TK_T * TK_PER)
        record("block_topk_reg_kernel<TK_PER>");
    else
        record("block_topk_reg_kernel<TK_PER_MAX>");
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

int main(int argc,char**argv){
 if(argc!=4)return 2;
 QsaShapes s;
 qsa_block_topk(nullptr,nullptr,std::stoll(argv[2]),std::stoll(argv[1]),2048,s,nullptr,nullptr,std::stoll(argv[3]));
 for(auto&v:launches)std::cout<<v<<"\n";
}
