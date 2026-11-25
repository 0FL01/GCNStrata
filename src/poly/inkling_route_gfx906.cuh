// Inkling Mini's private, shared decode/prefill route selector for wave64 gfx906.
#pragma once

#include "runtime.hpp"
#include "inkling_expf_gfx906.cuh"

namespace strata::poly::inkling_route {
inline constexpr int kExperts = 128, kPicks = 8;

// The full-wave shuffle, LDS bit swizzles and DPP quad permutations implement XOR exchange.
template<int M>
__device__ __forceinline__ float exchange(float value) {
    if constexpr (M == 32) return __shfl_xor(value, 32, 64);
    const int i = __float_as_int(value);
    if constexpr (M == 1 || M == 2)
        return __int_as_float(__builtin_amdgcn_mov_dpp(i, M == 1 ? 0xb1 : 0x4e, 15, 15, false));
    return __int_as_float(__builtin_amdgcn_ds_swizzle(i, 31 | (M << 10)));
}

__device__ __forceinline__ float probability(float x) {
    // x is -logit. Below the normal-result expf range, 1 + expf(x) rounds to 1;
    // above it, expf(x) overflows to +inf. This also covers both signed infinities.
    if (x < kExpMin) return 1.f;
    if (x > kExpMax) return 0.f;
    return 1.f / (1.f + expf_normal(x));
}

static __global__ void select_kernel(const float* scores, const float* bias, float scale, int32_t* ids, float* w,
                                     int32_t* invalid) {
    constexpr int wave = 64;
    const int lane = (int) threadIdx.x;
    const int row = (int) blockIdx.x;
    const float a = -scores[row * kExperts + lane], b = -scores[row * kExperts + lane + wave];
    const float b0 = bias[lane], b1 = bias[lane + wave];
    if (__ballot(isnan(a) || isnan(b) || !isfinite(b0) || !isfinite(b1))) {
        // Leave valid indices for the provisional kernels on both ranks. The caller must reject this call.
        if (lane < kPicks) ids[row * kPicks + lane] = lane, w[row * kPicks + lane] = 0.f;
        if (lane == 0) atomicOr(invalid, 1);
        return;
    }
    const float p0 = probability(a), p1 = probability(b);
    float s0 = p0 + b0, s1 = p1 + b1, mine = 0.f, sum = 0.f;
    int my_id = 0;
#pragma unroll 1
    for (int k = 0; k < kPicks; ++k) {
        float v = fmaxf(s0, s1);
        v = fmaxf(v, exchange<32>(v));
        v = fmaxf(v, exchange<16>(v));
        v = fmaxf(v, exchange<8>(v));
        v = fmaxf(v, exchange<4>(v));
        v = fmaxf(v, exchange<2>(v));
        v = fmaxf(v, exchange<1>(v));
        // Search the low half first: equal biased scores select the smallest remaining expert ID.
        const auto low = __ballot(v == s0), high = __ballot(v == s1);
        const int id = low ? __ffsll(low) - 1 : 63 + __ffsll(high);
        const float p = __shfl(id < wave ? p0 : p1, id & 63, wave);
        if (id == lane) s0 = -INFINITY;
        if (id == lane + wave) s1 = -INFINITY;
        sum += p;
        if (lane == k) my_id = id, mine = p;
    }
    if (lane < kPicks) {
        ids[row * kPicks + lane] = my_id;
        w[row * kPicks + lane] = mine * (scale / (sum + 1.e-20f));
    }
}

// All buffers are on the same card: scores[rows][128], bias[128], ids/weights[rows][8]. The caller clears
// the sticky `invalid` word before the first launch and checks it before accepting any output or state.
static inline void select(const float* scores, const float* bias, float scale, int32_t* ids, float* weights,
                          int rows, int32_t* invalid, void* stream) {
    select_kernel<<<rows, 64, 0, cs(stream)>>>(scores, bias, scale, ids, weights, invalid);
    ck(cudaGetLastError(), "Inkling GPU route");
}
}  // namespace strata::poly::inkling_route
