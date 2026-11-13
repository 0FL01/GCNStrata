#include <cstdint>
#include <iostream>
constexpr int TK_T=1024,TK_PER_MAX=66,kStepCount=4,kStepNKv=1,kStepNBid=2;
constexpr int64_t GFX906_TOPK_SHORT_CELLS=24576;
template<int PER,bool LONG_ONLY,int WINDOW_GUARD> bool g(const int32_t* steps,int nq){struct{int x;}gridDim{nq};
    constexpr bool RADIX_STOP = WINDOW_GUARD == 3 || WINDOW_GUARD == 4;
    constexpr int GUARD = RADIX_STOP ? WINDOW_GUARD - 2 : WINDOW_GUARD;
    if constexpr (LONG_ONLY) {
        if (steps[((int64_t) gridDim.x - 1) * kStepCount + kStepNKv] <= GFX906_TOPK_SHORT_CELLS) return false;
    }
    if constexpr (GUARD != 0) {
        static_assert(LONG_ONLY && (GUARD == 1 || GUARD == 2), "fit17/wide need LONG_ONLY");
        static_assert((GUARD == 1 && PER == 17) || (GUARD == 2 && PER == TK_PER_MAX), "guard/PER mismatch");
        // Exact actual-capacity specialization: max(n_bid)+1 <= 1024*17.
        // Device rows may be unsorted. This CTA-uniform guard precedes barriers.
        int32_t max_bid = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            if (q < (int) gridDim.x) {
                const int32_t bid = steps[(int64_t) q * kStepCount + kStepNBid];
                max_bid = bid > max_bid ? bid : max_bid;
            }
        }
        const bool fits17 = (int64_t) max_bid + 1 <= (int64_t) TK_T * 17;
        if constexpr (GUARD == 1) { if (!fits17) return false; }
        else { if (fits17) return false; }
    }
return true;}
int main(){int32_t st[32]={}; long long tests=0;
 for(int nq: {1,4,6,8})for(int nkv: {24575,24576,24577,69631,69632,69633,200000,270332})for(int hot=0;hot<nq;++hot){
  for(int q=0;q<nq;++q){st[4*q+1]=q==hot?nkv:24577;st[4*q+2]=st[4*q+1]/4;}
  if(g<17,true,1>(st,nq)!=g<17,true,3>(st,nq))return 1;
  if(g<66,true,2>(st,nq)!=g<66,true,4>(st,nq))return 2;
  ++tests;
 }
 std::cout<<tests<<"\n";
}
