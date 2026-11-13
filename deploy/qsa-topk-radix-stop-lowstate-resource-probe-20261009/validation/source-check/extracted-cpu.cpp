#include <cstdint>
#include <cstring>
#include <algorithm>
constexpr int R=4;
uint32_t __float_as_uint(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
extern "C" uint32_t cpp_key(uint32_t bits){float f;std::memcpy(&f,&bits,4);return order_key(f);}
template<bool RADIX_STOP> void choose_impl(const int* h,int64_t width,int above,int shift,uint32_t* d_out,int* a_out){
 int hist[1][256];std::memcpy(hist[0],h,1024);int s_digit=0;int s_above=0;
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            if constexpr (RADIX_STOP) {
                // Cell mass, not a block count: full blocks weigh R and there
                // is at most one positive partial tail. Mass 1..R is singleton.
                if (shift == 8 && hist[0][d] > 0 && hist[0][d] <= R) s_digit |= 256;
            }
            s_above = cum;

 *d_out=s_digit;*a_out=s_above;
}
extern "C" void cpp_choose(const int* h,int64_t width,int above,int shift,int mode,uint32_t* d,int* a){
 if(mode)choose_impl<true>(h,width,above,shift,d,a);else choose_impl<false>(h,width,above,shift,d,a);
}
template<int PER,bool RADIX_STOP> void count_impl(uint32_t* key,int64_t b0,int64_t b1,int64_t n_bid,int64_t n_kv,uint32_t thr,bool stopped,int* result){
 auto weight=[&](int64_t b)->int{return b<n_bid?R:(int)(n_kv-n_bid*R);};
 int s_digit=stopped?256:0;
    const uint32_t key_mask = RADIX_STOP && (s_digit & 256) ? 0xffffff00u : 0xffffffffu;

    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        // A stopped bucket has one positive-weight block. Normalize keys in
        // their existing storage so the unchanged count and emit use its prefix.
        // The mask dies after this scan; no full-key recovery is necessary.
        if constexpr (RADIX_STOP) key[j] &= key_mask;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }

 result[0]=gt;result[1]=eq;
}
extern "C" void cpp_count(uint32_t* key,int64_t b0,int64_t b1,int64_t n_bid,int64_t n_kv,uint32_t thr,int stopped,int per,int mode,int* result){
 if(per==17){if(mode)count_impl<17,true>(key,b0,b1,n_bid,n_kv,thr,stopped,result);else count_impl<17,false>(key,b0,b1,n_bid,n_kv,thr,stopped,result);}
 else {if(mode)count_impl<66,true>(key,b0,b1,n_bid,n_kv,thr,stopped,result);else count_impl<66,false>(key,b0,b1,n_bid,n_kv,thr,stopped,result);}
}
template<int PER> int emit_impl(const uint32_t* key,int64_t b0,int64_t b1,int64_t n_bid,int64_t n_kv,uint32_t thr,int64_t eq_left,int32_t* out){
 auto weight=[&](int64_t b)->int{return b<n_bid?R:(int)(n_kv-n_bid*R);};int64_t wpos=0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
 return (int)wpos;
}
extern "C" int cpp_emit(const uint32_t* key,int64_t b0,int64_t b1,int64_t n_bid,int64_t n_kv,uint32_t thr,int64_t eq_left,int per,int32_t* out){
 return per==17?emit_impl<17>(key,b0,b1,n_bid,n_kv,thr,eq_left,out):emit_impl<66>(key,b0,b1,n_bid,n_kv,thr,eq_left,out);
}
