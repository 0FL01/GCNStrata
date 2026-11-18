// src/kernels/cuda/route_v2.cuh - the gfx906 512 x 10 router top-10 body (STRATA_GFX906_ROUTE_V2; see native_router.cu),
// shared by its own kernel and the fused router chain (native_bf16.cu).  One 64-lane wave per token; `tl` = the lane
// in the wave (0..63).
#pragma once
#if defined(STRATA_HIP_GFX906)
#include <cfloat>
#include <cstdint>
namespace strata::kernels::rv2 {
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
__device__ __forceinline__ float row_fmax(float v) {
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xf, 0xf, false)));   // quad [1,0,3,2]
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xf, 0xf, false)));   // quad [2,3,0,1]
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x141, 0xf, 0xf, false)));  // row_half_mirror
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x140, 0xf, 0xf, false)));  // row_mirror
    return v;
}
__device__ __forceinline__ uint32_t row_umin(uint32_t v) {
    v = min(v, (uint32_t) __builtin_amdgcn_update_dpp(0, (int) v, 0xB1, 0xf, 0xf, false));
    v = min(v, (uint32_t) __builtin_amdgcn_update_dpp(0, (int) v, 0x4E, 0xf, 0xf, false));
    v = min(v, (uint32_t) __builtin_amdgcn_update_dpp(0, (int) v, 0x141, 0xf, 0xf, false));
    v = min(v, (uint32_t) __builtin_amdgcn_update_dpp(0, (int) v, 0x140, 0xf, 0xf, false));
    return v;
}
__device__ __forceinline__ void route_v2_body(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                              float* __restrict__ weights, const int tl) {
    const int lane = tl & 31, up = tl >> 5;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
    float v8[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float x = (up ? values[8 + j] : values[j]) * reciprocal;
        if (__isnanf(x)) x = -FLT_MAX;
        v8[j] = x;
    }
    const int base = lane + (up ? 256 : 0);   // v8[j] is expert base + 32 j
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v8[0];   // the lane's best, the lowest j on a tie (the old scan's strict >)
        int bj = 0;
#pragma unroll
        for (int j = 1; j < 8; ++j) {
            const bool g = v8[j] > best;
            best = g ? v8[j] : best;
            bj = g ? j : bj;
        }
        const float rm = row_fmax(best);
        const float m = fmaxf(fmaxf(__int_as_float(__builtin_amdgcn_readlane(__float_as_int(rm), 0)),
                                    __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rm), 16))),
                              fmaxf(__int_as_float(__builtin_amdgcn_readlane(__float_as_int(rm), 32)),
                                    __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rm), 48))));
        const uint32_t cand = best == m ? (uint32_t) (base + 32 * bj) : 0xffffffffu;
        const uint32_t rmin = row_umin(cand);
        const int expert = (int) min(min((uint32_t) __builtin_amdgcn_readlane((int) rmin, 0), (uint32_t) __builtin_amdgcn_readlane((int) rmin, 16)),
                                     min((uint32_t) __builtin_amdgcn_readlane((int) rmin, 32), (uint32_t) __builtin_amdgcn_readlane((int) rmin, 48)));
        if (base + 32 * bj == expert) v8[bj] = -INFINITY;   // the owner lane: its scan picked this expert
        if (!up && (expert & 31) == lane) selected_sum += m;
        if (!up && rank == lane) selected = m;
        if (tl == 0) ids[rank] = expert;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (!up && lane < 10) weights[lane] = selected * inverse_selected_sum;
}
}  // namespace strata::kernels::rv2
#endif
