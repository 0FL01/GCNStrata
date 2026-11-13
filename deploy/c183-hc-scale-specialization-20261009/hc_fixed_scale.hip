// Isolated, uncompiled candidate. The original c183 source/kernel is unchanged.
// No fast-math, approximate reciprocal, guard aggregation, fusion, or host-route changes.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>

template <int SCALE_SHIFT>
__device__ __forceinline__ void hc_fixed_scale_lane(const uint16_t* in, __half* out,
                                                    int64_t n, unsigned* stats) {
    static_assert(SCALE_SHIFT == 12 || SCALE_SHIFT == 9, "Only the two actual HC scales are supported");
    constexpr float scale = SCALE_SHIFT == 12 ? 4096.0f : 512.0f;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float f = __uint_as_float((uint32_t)in[i] << 16);
        const float converted = f * scale;
        const __half h = __float2half_rn(converted);
        out[i] = h;
        // Verbatim active non-AUDIT, SCALED, EXACT predicates from c183.
        if (!isfinite(f)) atomicOr(stats + 1, 1u);
        if (!isfinite(converted) || fabsf(converted) > 65504.0f) atomicOr(stats + 2, 1u);
        const float back = __half2float(h) / scale;
        if (__float_as_uint(f) != __float_as_uint(back)) atomicOr(stats + 3, 1u);
    }
}

extern "C" __global__ void hc_exact_scale4096(const uint16_t* in, __half* out,
                                              int64_t n, unsigned* stats) {
    hc_fixed_scale_lane<12>(in, out, n, stats);
}

extern "C" __global__ void hc_exact_scale512(const uint16_t* in, __half* out,
                                             int64_t n, unsigned* stats) {
    hc_fixed_scale_lane<9>(in, out, n, stats);
}
