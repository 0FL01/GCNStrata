// Inkling-2-Mini's loop visits, short convolutions and relative attention. The GGUF has 24 sets of weights;
// the three core traversals reuse those weights and expert slots, not the sequence state.
#include "inkling2.hpp"

#include "inkling_attend.hpp"
#include "inkling_dspark.hpp"
#include "inkling_dflash2.hpp"
#include "inkling_swa_layout.hpp"
#include "mmq.hpp"
#include "runtime.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(STRATA_HIP_GFX906)
#include "inkling_prefill.hpp"
#include "inkling_route_gfx906.cuh"
#include <cfenv>
#if defined(__x86_64__) && defined(__GLIBC__)
#include <xmmintrin.h>
#endif
#endif

namespace strata::poly {
namespace {

constexpr int kThreads = 128;
#ifdef STRATA_HIP_GFX906
constexpr int kMixThreads = 576;  // 9 wave64s x float4 = 2304 floats/row; avoids 108-VGPR/thread spill
#else
constexpr int kMixThreads = kThreads;
#endif
constexpr int kRowWidth = 2304;   // Inkling2::kWidth, for the kernels that keep a whole row in registers
constexpr int kKvWidth = 1024;    // Inkling2::kKey
constexpr int kValueTile = 128;
int div_up(int a, int b) { return (a + b - 1) / b; }

#if defined(STRATA_HIP_GFX906)
static_assert(sizeof(uint4) == inkling::kSwaCopyAlign && alignof(uint4) <= inkling::kSwaCopyAlign);
struct SwaCopyPlanes {
    uint4* ring[4];
    uint4* stage[4];
    int stride[4];   // uint4 vectors per row
};

template <bool Gather>
__global__ void swa_copy_kernel(SwaCopyPlanes p, int ring, int ring_head, int stage_head, int rows) {
    const int j = blockIdx.y, stride = p.stride[j], i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * stride) return;
    const int row = i / stride;
    int slot = ring_head + row;
    if (slot >= ring) slot -= ring;   // this copy visits at most one whole ring
    const int src = slot * stride + i - row * stride, dst = stage_head * stride + i;
    if constexpr (Gather) p.stage[j][dst] = p.ring[j][src];
    else p.ring[j][src] = p.stage[j][dst];
}

bool gpu_route_host_fp() {
#if defined(__x86_64__) && defined(__GLIBC__)
    // Match the x86 glibc FMA expf, with nearest rounding and no FTZ or DAZ on the CPU.
    return __builtin_cpu_supports("fma") && std::fegetround() == FE_TONEAREST &&
           (_mm_getcsr() & 0xe040u) == 0;
#else
    return false;
#endif
}

bool gpu_route_host_expf() {
    if (!gpu_route_host_fp()) return false;
    // These inputs distinguish the dispatched x86 FMA expf from the unfused one (e.g. FMA disabled by a libc
    // tunable), so check the actual libm used by route(), not just the processor's capability.
    const volatile float x = -0x1.f8cbb2p+5f, y = 0x1.04845ep+5f;
    return std::exp(x) == 0x1.f45326p-92f && std::exp(y) == 0x1.f93e38p+46f;
}

__global__ void gpu_route_sum(const float* down, const float* weight, float* fo, int width) {
    const int col = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (col >= width) return;
    const int row = (int) blockIdx.y, first = row * inkling_route::kPicks;
    const int at = row * width + col;
    float sum = fo[at];
    // Same starting value, order and multiply-add expression as add_scaled().
    for (int k = 0; k < inkling_route::kPicks; ++k)
        sum = sum + down[(first + k) * width + col] * weight[first + k];
    fo[at] = sum;
}
#endif

template <bool Maximum> __device__ float wave_reduce(float x) {
    for (int d = 16; d > 0; d /= 2) {
        const float y = __shfl_down(x, d, 32);
        if constexpr (Maximum) x = fmaxf(x, y);
        else x += y;
    }
    return x;
}

template <bool Maximum = false> __device__ float block_reduce(float x, float* waves) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
    x = wave_reduce<Maximum>(x);
    if (lane == 0) waves[wave] = x;
    __syncthreads();
    if (wave == 0) {
        x = lane < kThreads / 32 ? waves[lane] : Maximum ? -INFINITY : 0.f;
        x = wave_reduce<Maximum>(x);
        if (lane == 0) waves[0] = x;
    }
    __syncthreads();
    const float result = waves[0];
    __syncthreads();
    return result;
}

struct MixInputs { const float* x[5]; };

#ifdef STRATA_HIP_GFX906
__device__ __forceinline__ float wave64_sum_mix(float x) {
#pragma unroll
    for (int d = 32; d > 0; d >>= 1) x += __shfl_down(x, d, 64);
    return x;
}
#endif

// A depth query attends all finished blocks and the current partial block. RMS-normalized states are values too.
// Each input's two sums take the same per-thread order and the same reduction tree as one block_reduce() each, but
// a thread loads its 18 values of every input at once and all ten sums go up the tree together: a single block (one
// token) waits for memory once, not 36 times. The products are fused as the compiler fuses them in the per-element
// loops (explicitly: with five inputs in registers it would not).
__device__ __forceinline__ void mix_row(const MixInputs& in, int count, const float* query, float* out, size_t row) {
#ifdef STRATA_HIP_GFX906
    constexpr int kMost = 5, kW = kMixThreads / 64;
    static_assert(kMixThreads * 4 == kRowWidth && kMixThreads % 64 == 0, "576 threads = 9 wave64s x float4");
    const int tid = threadIdx.x;
    const int lane = tid & 63, wave = tid >> 6;
    __shared__ float waves[2 * kMost][kW];

    const float4 q = reinterpret_cast<const float4*>(query)[tid];
    float4 xv[kMost];
    for (int j = 0; j < count; ++j) {
        const float4 x = reinterpret_cast<const float4*>(in.x[j] + row)[tid];
        xv[j] = x;
        float ss = __fmaf_rn(x.x, x.x, __fmaf_rn(x.y, x.y, __fmaf_rn(x.z, x.z, x.w * x.w)));
        float dot = __fmaf_rn(x.x, q.x, __fmaf_rn(x.y, q.y, __fmaf_rn(x.z, q.z, x.w * q.w)));
        ss = wave64_sum_mix(ss);
        dot = wave64_sum_mix(dot);
        if (lane == 0) {
            waves[2 * j][wave] = ss;
            waves[2 * j + 1][wave] = dot;
        }
    }
    __syncthreads();
    if (wave == 0 && lane < 2 * count) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < kW; ++w) s += waves[lane][w];
        waves[lane][0] = s;
    }
    __syncthreads();
    float scale[kMost], w[kMost], top = -INFINITY;
    for (int j = 0; j < count; ++j) {
        scale[j] = rsqrtf(waves[2 * j][0] / kRowWidth + 1.e-6f);
        w[j] = waves[2 * j + 1][0] * scale[j] * rsqrtf((float) kRowWidth);
        top = fmaxf(top, w[j]);
    }
    float sum = 0.f;
    for (int j = 0; j < count; ++j) sum += (w[j] = expf(w[j] - top));
    const float inv_sum = 1.f / sum;
    float4 res = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int j = 0; j < count; ++j) {
        const float c = w[j] * scale[j] * inv_sum;
        res.x = __fmaf_rn(c, xv[j].x, res.x);
        res.y = __fmaf_rn(c, xv[j].y, res.y);
        res.z = __fmaf_rn(c, xv[j].z, res.z);
        res.w = __fmaf_rn(c, xv[j].w, res.w);
    }
    reinterpret_cast<float4*>(out + row)[tid] = res;
#else
    constexpr int kMost = 5, kWaves = kThreads / 32, kPer = kRowWidth / kThreads;
    static_assert(kRowWidth % kThreads == 0, "a row is whole loads of the block");
    __shared__ float waves[2 * kMost][kWaves];
    float x[kMost][kPer], q[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) q[k] = query[threadIdx.x + k * kThreads];
#pragma unroll
    for (int j = 0; j < kMost; ++j)
        if (j < count)
#pragma unroll
            for (int k = 0; k < kPer; ++k) x[j][k] = in.x[j][row + threadIdx.x + k * kThreads];
    const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        if (j >= count) break;
        float ss = 0.f, dot = 0.f;
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            ss = __fmaf_rn(x[j][k], x[j][k], ss);
            dot = __fmaf_rn(x[j][k], q[k], dot);
        }
        ss = wave_reduce<false>(ss);
        dot = wave_reduce<false>(dot);
        if (lane == 0) {
            waves[2 * j][wave] = ss;
            waves[2 * j + 1][wave] = dot;
        }
    }
    __syncthreads();
    if (wave == 0)
#pragma unroll
        for (int v = 0; v < 2 * kMost; ++v) {
            if (v >= 2 * count) break;
            const float y = wave_reduce<false>(lane < kWaves ? waves[v][lane] : 0.f);
            if (lane == 0) waves[v][0] = y;
        }
    __syncthreads();
    float scale[kMost], w[kMost], top = -INFINITY;
#pragma unroll
    for (int j = 0; j < kMost; ++j)
        if (j < count) {
            scale[j] = rsqrtf(waves[2 * j][0] / kRowWidth + 1.e-6f);
            w[j] = waves[2 * j + 1][0] * scale[j] * rsqrtf((float) kRowWidth);
            top = fmaxf(top, w[j]);
        }
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kMost; ++j)
        if (j < count) sum += (w[j] = expf(w[j] - top));
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        float val = 0.f;
#pragma unroll
        for (int j = 0; j < kMost; ++j)
            if (j < count) val = __fmaf_rn(w[j] * scale[j], x[j][k], val);
        out[row + threadIdx.x + k * kThreads] = val / sum;
    }
#endif
}

__global__ void mix_kernel(MixInputs in, int count, const float* query, float* out) {
    mix_row(in, count, query, out, (size_t) blockIdx.x * kRowWidth);
}

// A visit's streams (Inkling2::heads()): stream blockIdx.y's query and output, each row as mix_kernel's.
struct Streams3 {
    const float* query[3];
    float* out[3];
};
__global__ void mix3_kernel(MixInputs in, int count, Streams3 s) {
    mix_row(in, count, s.query[blockIdx.y], s.out[blockIdx.y], (size_t) blockIdx.x * kRowWidth);
}

#if defined(STRATA_HIP_GFX906)
struct MixQ81 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(MixQ81) == 36 && alignof(MixQ81) == 4);
struct MixNormRows {
    const float* query[3];
    const float* weight[3];
    float* normalized[3];
    void* quant[3];
};

__device__ __forceinline__ float mix_norm_sum(float x) {
#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, 32);
    }
    return x;
}

__device__ __forceinline__ float mix_norm_max(float x) {
#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset, 32));
    }
    return x;
}

// One stock SUM thread per logical index. 576 physical threads in mix can
// emulate the other 448 logical SUM threads without changing any shuffles:
// 576 is a multiple of both 32 and 64.
template<bool Ffn>
__device__ __forceinline__ void mix_norm_from_lds(const float* values, const MixNormRows& out, int stream, int row, float eps) {
#pragma clang fp contract(off)
    constexpr int sum_threads = Ffn ? 128 : 1024;
    __shared__ float part[sum_threads / 32];
    const int tid = threadIdx.x;
    for (int logical = tid; logical < sum_threads; logical += kMixThreads) {
        float ss = 0.f;
        for (int i = logical; i < kRowWidth; i += sum_threads) {
            const float v = values[i];
            ss = __fmaf_rn(v, v, ss);
        }
        ss = mix_norm_sum(ss);
        if ((logical & 31) == 0) part[logical >> 5] = ss;
    }
    __syncthreads();
    float sum = 0.f;
    for (int k = 0; k < sum_threads / 32; ++k) sum = __fadd_rn(sum, part[k]);
    float scale;
    if constexpr (Ffn) scale = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(kRowWidth)), eps)));
    else scale = 1.f / sqrtf(sum / float(kRowWidth) + eps);
    float* y = out.normalized[stream] + row * kRowWidth;
    MixQ81* q = static_cast<MixQ81*>(out.quant[stream]) + row * (kRowWidth / 32);
    // kMixThreads and the stock 1024-thread quantizer both stride in multiples
    // of 32: each block's amax and raw FP32 sum have the same XOR tree.
    for (int i = tid; i < kRowWidth; i += kMixThreads) {
        float xi;
        if constexpr (Ffn) xi = __fmul_rn(__fmul_rn(values[i], scale), out.weight[stream][i]);
        else xi = values[i] * scale * out.weight[stream][i];
        y[i] = xi;
        const float amax = mix_norm_max(fabsf(xi));
        const float bsum = mix_norm_sum(xi);
        const float d = kernels::q8_1_finite(amax / 127.f);
        q[i / 32].qs[i % 32] = kernels::q8_1_quant(xi, d, amax);
        if (i % 32 == 0) q[i / 32].ds = kernels::q8_1_ds(d, bsum);
    }
}

// Same mixer arithmetic as the stock runtime-bound loops, but a constant
// upper bound lets LLVM scalarize all five float4s, scales and probabilities.
// Dynamic indexing of those per-thread arrays otherwise uses scratch.
__device__ __forceinline__ float4 mix_value_registers(const MixInputs& in, int count, const float* query, size_t row) {
    constexpr int kMost = 5, kW = kMixThreads / 64;
    static_assert(kMixThreads * 4 == kRowWidth && kMixThreads % 64 == 0, "576 threads = 9 wave64s x float4");
    const int tid = threadIdx.x;
    const int lane = tid & 63, wave = tid >> 6;
    __shared__ float waves[2 * kMost][kW];

    const float4 q = reinterpret_cast<const float4*>(query)[tid];
    float4 xv[kMost];

#pragma unroll
    for (int j = 0; j < kMost; ++j) if (j < count) {
        const float4 x = reinterpret_cast<const float4*>(in.x[j] + row)[tid];
        xv[j] = x;
        float ss = __fmaf_rn(x.x, x.x, __fmaf_rn(x.y, x.y, __fmaf_rn(x.z, x.z, x.w * x.w)));
        float dot = __fmaf_rn(x.x, q.x, __fmaf_rn(x.y, q.y, __fmaf_rn(x.z, q.z, x.w * q.w)));
        ss = wave64_sum_mix(ss);
        dot = wave64_sum_mix(dot);
        if (lane == 0) {
            waves[2 * j][wave] = ss;
            waves[2 * j + 1][wave] = dot;
        }
    }
    __syncthreads();
    if (wave == 0 && lane < 2 * count) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < kW; ++w) s += waves[lane][w];
        waves[lane][0] = s;
    }
    __syncthreads();
    float scale[kMost], w[kMost], top = -INFINITY;

#pragma unroll
    for (int j = 0; j < kMost; ++j) if (j < count) {
        scale[j] = rsqrtf(waves[2 * j][0] / kRowWidth + 1.e-6f);
        w[j] = waves[2 * j + 1][0] * scale[j] * rsqrtf((float) kRowWidth);
        top = fmaxf(top, w[j]);
    }
    float sum = 0.f;

#pragma unroll
    for (int j = 0; j < kMost; ++j) if (j < count) sum += (w[j] = expf(w[j] - top));
    const float inv_sum = 1.f / sum;
    float4 res = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
    for (int j = 0; j < kMost; ++j) if (j < count) {
        const float c = w[j] * scale[j] * inv_sum;
        res.x = __fmaf_rn(c, xv[j].x, res.x);
        res.y = __fmaf_rn(c, xv[j].y, res.y);
        res.z = __fmaf_rn(c, xv[j].z, res.z);
        res.w = __fmaf_rn(c, xv[j].w, res.w);
    }
    return res;
}

template<bool Ffn>
__global__ void __launch_bounds__(kMixThreads) fused_mix_norm(MixInputs in, int count, MixNormRows out, float eps) {
    __shared__ float4 mixed[kMixThreads];
    const int stream = blockIdx.y, row = blockIdx.x;
    const float4 v = mix_value_registers(in, count, out.query[stream], size_t(row) * kRowWidth);
    mixed[threadIdx.x] = v;
    __syncthreads();
    mix_norm_from_lds<Ffn>(reinterpret_cast<const float*>(mixed), out, stream, row, eps);
}

struct alignas(4) PrefillMmqBlock {
    union { float d4[4]; __half2 ds4[4]; };
    int8_t qs[128];
};
static_assert(sizeof(PrefillMmqBlock) == 144 && offsetof(PrefillMmqBlock, qs) == 16);
constexpr int kMmqPaddedWidth = ((kRowWidth + 511) / 512) * 512;

template<bool DS4>
__device__ __forceinline__ void pack_mmq_prefill(PrefillMmqBlock* dst, float4 x, int col, int rows, int row) {
#pragma clang fp contract(off)
    float amax = fabsf(x.x);
    amax = fmaxf(amax, fabsf(x.y));
    amax = fmaxf(amax, fabsf(x.z));
    amax = fmaxf(amax, fabsf(x.w));
#pragma unroll
    for (int d = 4; d > 0; d >>= 1) amax = fmaxf(amax, __shfl_xor(amax, d, 64));
    float sum = 0.f;
    if constexpr (DS4) {
        sum = x.x + x.y + x.z + x.w;
#pragma unroll
        for (int d = 4; d > 0; d >>= 1) sum += __shfl_xor(sum, d, 64);
    }
    const float inv = 127.f / amax;
    char4 q;
    q.x = roundf(x.x * inv);
    q.y = roundf(x.y * inv);
    q.z = roundf(x.z * inv);
    q.w = roundf(x.w * inv);
    const float scale = 1.f / inv;
    PrefillMmqBlock& b = dst[size_t(col / 128) * rows + row];
    reinterpret_cast<char4*>(b.qs)[(col % 128) / 4] = q;
    if (col % 32 == 0) {
        if constexpr (DS4) b.ds4[(col % 128) / 32] = __floats2half2_rn(scale, sum);
        else b.d4[(col % 128) / 32] = scale;
    }
}

template<bool DS4>
__global__ void __launch_bounds__(kMixThreads)
fused_mix_rms_mmq_prefill(MixInputs in, int count, const float* query, const float* weight, float4* y,
                          PrefillMmqBlock* packed, int rows, float eps, float pad_zero) {
#pragma clang fp contract(off)
    constexpr int sum_threads = 1024;
    __shared__ float4 mixed[kMixThreads];
    __shared__ float part[sum_threads / 32];
    const int tid = threadIdx.x, row = blockIdx.x;
    const float4 raw = mix_value_registers(in, count, query, size_t(row) * kRowWidth);
    mixed[tid] = raw;
    __syncthreads();
    const float* v = reinterpret_cast<const float*>(mixed);
    for (int logical = tid; logical < sum_threads; logical += kMixThreads) {
        float ss = 0.f;
        for (int i = logical; i < kRowWidth; i += sum_threads) ss = __fmaf_rn(v[i], v[i], ss);
        ss = mix_norm_sum(ss);
        if ((logical & 31) == 0) part[logical >> 5] = ss;
    }
    __syncthreads();
    float sum = 0.f;
    for (int k = 0; k < sum_threads / 32; ++k) sum = __fadd_rn(sum, part[k]);
    const float scale = 1.f / sqrtf(sum / float(kRowWidth) + eps);
    const int c = tid * 4;
    const float4 out = make_float4(raw.x * scale * weight[c], raw.y * scale * weight[c + 1],
                                   raw.z * scale * weight[c + 2], raw.w * scale * weight[c + 3]);
    y[size_t(row) * (kRowWidth / 4) + tid] = out;
    if (packed) {
        pack_mmq_prefill<DS4>(packed, out, c, rows, row);
        if (tid < (kMmqPaddedWidth - kRowWidth) / 4)
            pack_mmq_prefill<DS4>(packed, make_float4(pad_zero, pad_zero, pad_zero, pad_zero),
                                  kRowWidth + c, rows, row);
    }
}

__global__ void __launch_bounds__(kMixThreads)
fused_mix_rms_ffn_prefill(MixInputs in, int count, const float* query, const float* weight, float4* y, float eps) {
#pragma clang fp contract(off)
    constexpr int sum_threads = 128;
    __shared__ float4 mixed[kMixThreads];
    __shared__ float part[sum_threads / 32];
    const int tid = threadIdx.x, row = blockIdx.x;
    const float4 raw = mix_value_registers(in, count, query, size_t(row) * kRowWidth);
    mixed[tid] = raw;
    __syncthreads();
    const float* v = reinterpret_cast<const float*>(mixed);
    if (tid < sum_threads) {
        float ss = 0.f;
        for (int i = tid; i < kRowWidth; i += sum_threads) ss = __fmaf_rn(v[i], v[i], ss);
        ss = mix_norm_sum(ss);
        if ((tid & 31) == 0) part[tid >> 5] = ss;
    }
    __syncthreads();
    float sum = 0.f;
    for (int k = 0; k < sum_threads / 32; ++k) sum = __fadd_rn(sum, part[k]);
    const float scale = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(kRowWidth)), eps)));
    const int c = tid * 4;
    y[size_t(row) * (kRowWidth / 4) + tid] = make_float4(
        __fmul_rn(__fmul_rn(raw.x, scale), weight[c]),
        __fmul_rn(__fmul_rn(raw.y, scale), weight[c + 1]),
        __fmul_rn(__fmul_rn(raw.z, scale), weight[c + 2]),
        __fmul_rn(__fmul_rn(raw.w, scale), weight[c + 3]));
}

#endif

__global__ void control_kernel(float* cur, const float* msg, const float* gain, const float* forget, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) cur[i] = (1.f - tanhf(forget[i])) * cur[i] + tanhf(gain[i]) * msg[i];
}

#if defined(STRATA_HIP_GFX906)
// Same expression as control_kernel, but keep the raw depth mix intact for the next stream.
__global__ void control_from_shared(const float* depth, float* cur, const float* msg, const float* gain,
                                    const float* forget, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) cur[i] = (1.f - tanhf(forget[i])) * depth[i] + tanhf(gain[i]) * msg[i];
}
#endif

// One thread owns one channel and keeps its raw input history across the chunk. In GGUF the static taps
// run oldest to newest; the input-dependent taps of each KV head start with the current token.
// With `snap`, a window's (Engine::window()): the state after each of its tokens but the last as well, token t's
// `stride` floats after token t - 1's, at the place the state has in state_ (Engine::keep() puts one back).
__device__ __forceinline__ void conv_channel(float* x, float* state, const float* w, const float* dynamic, int width,
                                             int n, int i, float* snap = nullptr, size_t stride = 0) {
    float* h = state + 3 * i;
    float a = h[0], b = h[1], c = h[2];
    w += 4 * i;
    const float scale = dynamic ? 1.f / 3.f : 0.4472135954999579f;
    for (int t = 0; t < n; ++t) {
        const size_t at = (size_t) t * width + i;
        const float d = x[at];
        float s = a * w[0] + b * w[1] + c * w[2] + d * w[3];
        if (dynamic) {
            const float* p = dynamic + 4 * (t * (width / 128) + i / 128);
            s += d * p[0] + c * p[1] + b * p[2] + a * p[3];
        }
        x[at] = (d + s) * scale;
        a = b;
        b = c;
        c = d;
        if (snap && t + 1 < n) {
            float* k = snap + (size_t) t * stride + 3 * i;
            k[0] = a;
            k[1] = b;
            k[2] = c;
        }
    }
    h[0] = a;
    h[1] = b;
    h[2] = c;
}

__global__ void conv_kernel(float* x, float* state, const float* w, const float* dynamic, int width, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) conv_channel(x, state, w, dynamic, width, n, i);
}

// The decode switch's launches that each do what two to four of the ones above do, with the same sums.
struct Conv {
    float* x;
    float* state;
    const float* w;
    const float* dynamic;
    float* snap = nullptr;   // conv_channel()'s
};
// K's and V's convolutions (kKey channels each) in one launch.
__global__ void conv2_kernel(Conv k, Conv v, int width, int n, size_t stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool second = i >= width;
    if (second) i -= width;
    if (i >= width) return;
    const Conv& c = second ? v : k;
    conv_channel(c.x, c.state, c.w, c.dynamic, width, n, i, c.snap, stride);
}

// A static convolution of x (plus `add`, the CPU experts' sum, when there is one) added to the residual r, as
// axpy(add, 1, x), conv() and axpy(x, 1, r) do. With `depth`, a block of layers starts: depth takes r and r takes x,
// as axpy(r, 1, depth) and axpy(x, 1, r) without adding do. With `moe`, the routed experts' rows in the router's
// order (STRATA_INKLING_TAIL_MOE, one token's: n == 1) are the x itself, as the add_scaled() chain wrote it into fo_:
// the same sums, one launch less.
__global__ void conv_add_kernel(float* x, const float* add, float* state, const float* w, float* r, float* depth,
                                int width, int n, bool clean, TailSum moe, float* snap = nullptr, size_t stride = 0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    if (moe.n)
        for (int t = 0; t < n; ++t) {
            const size_t at = (size_t) t * width + i;
            // fo_ was 0 where add_scaled() started (moe_shared()), so this chain is its sums bit for bit
            float v = 0.f;
            for (int k = 0; k < moe.n; ++k) v = v + moe.x[k][at] * moe.w[k];
            x[at] = v;
        }
    if (add)
        for (int t = 0; t < n; ++t) x[(size_t) t * width + i] = x[(size_t) t * width + i] + add[(size_t) t * width + i];
    conv_channel(x, state, w, nullptr, width, n, i, snap, stride);
    for (int t = 0; t < n; ++t) {
        const size_t at = (size_t) t * width + i;
        if (depth) {
            depth[at] = r[at];
            r[at] = x[at];
        } else r[at] = r[at] + x[at];
        if (clean) x[at] = 0.f;   // fo_: the next layer's experts add into it (moe_shared())
    }
}

// A prompt batch's convolution in spans of kConvSpan tokens, a thread a channel and span: conv_halo_kernel keeps the
// three raw inputs before each span (the state's before the first) and conv_span_kernel computes each span as
// conv_kernel does, with the same sums, so the results are conv_kernel's bit for bit.
constexpr int kConvSpan = 64;
__global__ void conv_halo_kernel(const float* x, const float* state, float* halo, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, span = blockIdx.y;
    if (i >= width) return;
    float* to = halo + ((size_t) span * width + i) * 3;
    for (int j = 0; j < 3; ++j) {
        const int t = span * kConvSpan - 3 + j;
        to[j] = t >= 0 ? x[(size_t) t * width + i] : state[3 * i + 3 + t];
    }
}

__global__ void conv_span_kernel(float* x, float* state, const float* halo, const float* w, const float* dynamic,
                                 int width, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, span = blockIdx.y;
    if (i >= width) return;
    const float* from = halo + ((size_t) span * width + i) * 3;
    float a = from[0], b = from[1], c = from[2];
    w += 4 * i;
    const float scale = dynamic ? 1.f / 3.f : 0.4472135954999579f;
    const int first = span * kConvSpan, end = min(n, first + kConvSpan);
    for (int t = first; t < end; ++t) {
        const size_t at = (size_t) t * width + i;
        const float d = x[at];
        float s = a * w[0] + b * w[1] + c * w[2] + d * w[3];
        if (dynamic) {
            const float* p = dynamic + 4 * (t * (width / 128) + i / 128);
            s += d * p[0] + c * p[1] + b * p[2] + a * p[3];
        }
        x[at] = (d + s) * scale;
        a = b;
        b = c;
        c = d;
    }
    if (end == n) {
        float* h = state + 3 * i;
        h[0] = a;
        h[1] = b;
        h[2] = c;
    }
}

#if defined(STRATA_HIP_GFX906)
// Out-of-place ShortConv for prefill batches (n > kConvSpan): reads immutable raw input x and incoming
// history `before`, writes filtered output to `y` and outgoing history to `after` (eliminates conv_halo_kernel).
__global__ void conv_fused_prefill_kernel(const float* __restrict__ x, const float* __restrict__ before,
                                          const float* __restrict__ w, const float* __restrict__ dynamic,
                                          float* __restrict__ y, float* __restrict__ after, int width, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, first = blockIdx.y * kConvSpan;
    if (i >= width) return;
    const int end = min(n, first + kConvSpan);
    float a = first ? x[(size_t) (first - 3) * width + i] : before[3 * i];
    float b = first ? x[(size_t) (first - 2) * width + i] : before[3 * i + 1];
    float c = first ? x[(size_t) (first - 1) * width + i] : before[3 * i + 2];
    w += 4 * i;
    const float scale = dynamic ? 1.f / 3.f : 0.4472135954999579f;
    for (int t = first; t < end; ++t) {
        const size_t at = (size_t) t * width + i;
        const float d = x[at];
        float s = a * w[0] + b * w[1] + c * w[2] + d * w[3];
        if (dynamic) {
            const float* p = dynamic + 4 * (t * (width / 128) + i / 128);
            s += d * p[0] + c * p[1] + b * p[2] + a * p[3];
        }
        y[at] = (d + s) * scale;
        a = b;
        b = c;
        c = d;
    }
    if (end == n) {
        after[3 * i] = a;
        after[3 * i + 1] = b;
        after[3 * i + 2] = c;
    }
}

// Out-of-place static ShortConv fused with residual update (and optional depth_ snapshot) for prefill batches.
__global__ void conv_add_prefill_kernel(const float* __restrict__ x, const float* __restrict__ before,
                                        const float* __restrict__ w, float* __restrict__ r,
                                        float* __restrict__ depth_copy, float* __restrict__ after, int width, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, first = blockIdx.y * kConvSpan;
    if (i >= width) return;
    const int end = min(n, first + kConvSpan);
    float a = first ? x[(size_t) (first - 3) * width + i] : before[3 * i];
    float b = first ? x[(size_t) (first - 2) * width + i] : before[3 * i + 1];
    float c = first ? x[(size_t) (first - 1) * width + i] : before[3 * i + 2];
    w += 4 * i;
    constexpr float scale = 0.4472135954999579f;
    for (int t = first; t < end; ++t) {
        const size_t at = (size_t) t * width + i;
        const float d = x[at];
        const float s = a * w[0] + b * w[1] + c * w[2] + d * w[3];
        const float y = (d + s) * scale;
        if (depth_copy) {
            depth_copy[at] = r[at];
            r[at] = y;
        } else {
            r[at] = r[at] + y;
        }
        a = b;
        b = c;
        c = d;
    }
    if (end == n) {
        after[3 * i] = a;
        after[3 * i + 1] = b;
        after[3 * i + 2] = c;
    }
}
#endif

__device__ __forceinline__ void head_norm(float* x, const float* w, int width, int heads, float clip, float eps,
                                          int64_t pos, int floor, float alpha, int head, int row, float* waves) {
    const int i = threadIdx.x;
    float* dest = x + ((size_t) row * heads + head) * width;
    const float v = i < width ? dest[i] * (w ? w[head * width + i] : 1.f) : 0.f;
    const float ss = block_reduce(v * v, waves);
    const float s = clip > 0.f ? fminf(1.f, clip / (sqrtf(ss) + 1.e-6f)) : rsqrtf(ss / width + eps);
    const float tau = floor > 0 ? 1.f + alpha * logf(fmaxf(1.f, float(pos + row + 1) / floor)) : 1.f;
    if (i < width) dest[i] = v * s * tau;
}

__global__ void head_norm_kernel(float* x, const float* w, int width, int heads, float clip, float eps,
                                 int64_t pos, int floor, float alpha) {
    __shared__ float waves[kThreads / 32];
    head_norm(x, w, width, heads, clip, eps, pos, floor, alpha, blockIdx.x, blockIdx.y, waves);
}

// Q's, K's, V's and the relative terms' norms in one launch, a block a head as head_norm_kernel.
struct HeadNorm {
    float* x;
    const float* w;
    int width, heads;
    float clip;
    int floor;
};
struct HeadNorms { HeadNorm s[4]; };
__global__ void head_norms_kernel(HeadNorms h, float eps, int64_t pos, float alpha) {
    __shared__ float waves[kThreads / 32];
    int head = blockIdx.x, j = 0;
    while (j < 3 && head >= h.s[j].heads) head -= h.s[j++].heads;
    const HeadNorm& s = h.s[j];
    head_norm(s.x, s.w, s.width, s.heads, s.clip, eps, pos, s.floor, alpha, head, blockIdx.y, waves);
}

// One decode row's conv2_kernel, head_norms_kernel and store2_q8_kernel in one launch, a block a head as
// head_norms_kernel: a K or V head's block first convolves its channels (a thread a channel, as conv2_kernel), then
// norms the head, then each wave writes its 32 values' Q8_0 block as store2_q8_kernel does (the block's maximum by
// shuffles: the same maximum, so the same scale and codes).  The normed K and V stay in k and v as well.
struct KvStore {
    int8_t *kc, *vc;
    __half *ks, *vs;
    int64_t at;   // the row's place in the cache
};
__global__ void heads_kv_kernel(Conv k, Conv v, HeadNorms h, KvStore c, float eps, int64_t pos, float alpha,
                                size_t stride, int kv_width) {
    __shared__ float waves[kThreads / 32];
    int head = blockIdx.x, j = 0;
    while (j < 3 && head >= h.s[j].heads) head -= h.s[j++].heads;
    static_assert(kThreads == 128, "a thread a value of a 128-wide K or V head");
    const int i = head * kThreads + threadIdx.x;
    if (j == 1 || j == 2) {
        const Conv& cv = j == 2 ? v : k;
        conv_channel(cv.x, cv.state, cv.w, cv.dynamic, kv_width, 1, i, cv.snap, stride);
    }
    const HeadNorm& s = h.s[j];
    head_norm(s.x, s.w, s.width, s.heads, s.clip, eps, pos, s.floor, alpha, head, 0, waves);
    if (j != 1 && j != 2) return;
    const float x = s.x[i];
    float amax = fmaxf(0.f, fabsf(x));   // store2_q8_kernel's maximum starts at 0: a NaN does not count
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    const float d = amax / 127.0f;
    const float id = d > 0.f ? 1.0f / d : 0.0f;
    const int nb = kv_width / 32;
    if (threadIdx.x % 32 == 0) (j == 2 ? c.vs : c.ks)[(size_t) c.at * nb + i / 32] = __float2half(d);
    float q = rintf(x * id);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    (j == 2 ? c.vc : c.kc)[(size_t) c.at * kv_width + i] = (int8_t) q;
}

// Multi-row version of heads_kv_kernel for a fused Q8_0 decode/verify window (n > 1):
//   - Q and rel blocks (2 * n_heads * n blocks) run head_norm in parallel across all n rows;
//   - K and V blocks (2 * n_kv blocks) run conv_channel -> head_norm -> Q8_0 store in a single register pass over
//     t = 0..n-1 with zero extra global reads of k_/v_.
// Every output word equals conv2_kernel + head_norms_kernel + store2_q8_kernel bitwise.
__global__ void heads_kv_rows_kernel(Conv k, Conv v, HeadNorms h, KvStore c, float eps, int64_t pos, int ring,
                                     float alpha, size_t stride, int kv_width, int n) {
    __shared__ float waves[kThreads / 32];
    static_assert(kThreads == 128, "a thread a value of a 128-wide K or V head");
    const int n_q = h.s[0].heads * n;
    const int n_rel = h.s[3].heads * n;
    const int b = blockIdx.x;
    if (b < n_q) {
        const HeadNorm& s = h.s[0];
        head_norm(s.x, s.w, s.width, s.heads, s.clip, eps, pos, s.floor, alpha, b % s.heads, b / s.heads, waves);
        return;
    }
    if (b < n_q + n_rel) {
        const int r = b - n_q;
        const HeadNorm& s = h.s[3];
        head_norm(s.x, s.w, s.width, s.heads, s.clip, eps, pos, s.floor, alpha, r % s.heads, r / s.heads, waves);
        return;
    }
    const int kv_idx = b - (n_q + n_rel);
    const bool is_v = kv_idx >= h.s[1].heads;
    const int head = is_v ? kv_idx - h.s[1].heads : kv_idx;
    const Conv& cv = is_v ? v : k;
    const HeadNorm& s = is_v ? h.s[2] : h.s[1];
    const int i = head * kThreads + threadIdx.x;
    float* hist = cv.state + 3 * i;
    float ha = hist[0], hb = hist[1], hc = hist[2];
    const float* w = cv.w + 4 * i;
    const float scale = cv.dynamic ? 1.f / 3.f : 0.4472135954999579f;
    const int nb = kv_width / 32;
    for (int t = 0; t < n; ++t) {
        const size_t at_x = (size_t) t * kv_width + i;
        const float d = cv.x[at_x];
        float sv = ha * w[0] + hb * w[1] + hc * w[2] + d * w[3];
        if (cv.dynamic) {
            const float* p = cv.dynamic + 4 * (t * (kv_width / 128) + i / 128);
            sv += d * p[0] + hc * p[1] + hb * p[2] + ha * p[3];
        }
        const float conv_out = (d + sv) * scale;
        ha = hb;
        hb = hc;
        hc = d;
        if (cv.snap && t + 1 < n) {
            float* sn = cv.snap + (size_t) t * stride + 3 * i;
            sn[0] = ha;
            sn[1] = hb;
            sn[2] = hc;
        }
        // head_norm for K/V (s.w == nullptr, s.width == 128, s.floor == 0 -> tau == 1.f)
        const float ss = block_reduce(conv_out * conv_out, waves);
        const float ns = s.clip > 0.f ? fminf(1.f, s.clip / (sqrtf(ss) + 1.e-6f)) : rsqrtf(ss / kThreads + eps);
        const float x = conv_out * ns;
        cv.x[at_x] = x;
        float amax = fmaxf(0.f, fabsf(x));   // store2_q8_kernel's maximum starts at 0: a NaN does not count
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
        const float dq = amax / 127.0f;
        const float id = dq > 0.f ? 1.0f / dq : 0.0f;
        const int64_t at_kv = ring ? (pos + t) % ring : pos + t;
        if (threadIdx.x % 32 == 0) (is_v ? c.vs : c.ks)[(size_t) at_kv * nb + i / 32] = __float2half(dq);
        float q = rintf(x * id);
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        (is_v ? c.vc : c.kc)[(size_t) at_kv * kv_width + i] = (int8_t) q;
    }
    hist[0] = ha;
    hist[1] = hb;
    hist[2] = hc;
}

// K and V into their Q8_0 caches in one launch, as store_q8_kernel writes them, a row at (pos + row) % ring (or
// pos + row in a full layer's cache).
__global__ void store2_q8_kernel(const float* k, const float* v, int8_t* kc, __half* ks, int8_t* vc, __half* vs,
                                 int64_t pos, int ring, int rows, int kv_width) {
    const int nb = kv_width / 32;
    int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= 2 * nb * rows) return;
    const bool second = gb >= nb * rows;
    if (second) gb -= nb * rows;
    const int row = gb / nb, blk = gb % nb;
    const int64_t at = ring ? (pos + row) % ring : pos + row;
    const float* xr = (second ? v : k) + (size_t) row * kv_width + blk * 32;
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xr[j]));
    const float d = amax / 127.0f;
    const float id = d > 0.f ? 1.0f / d : 0.0f;
    (second ? vs : ks)[(size_t) at * nb + blk] = __float2half(d);
    int8_t* cr = (second ? vc : kc) + (size_t) at * kv_width + blk * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float q = rintf(xr[j] * id);
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        cr[j] = (int8_t) q;
    }
}

// A loop stream's control (Inkling2::control) in three launches instead of seven, the first: the message as
// mix_kernel mixes the banks, and the RMS norms of the stream and the message side by side in pair.
__device__ __forceinline__ void control_in_row(const float* bank0, const float* bank1, const float* query,
                                               const float* in, float* msg, float* pair, size_t row) {
#ifdef STRATA_HIP_GFX906
    constexpr int kW = kMixThreads / 64;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    __shared__ float waves[6][kW];

    const float4 q = reinterpret_cast<const float4*>(query)[tid];
    const float4 b0 = reinterpret_cast<const float4*>(bank0 + row)[tid];
    const float4 b1 = reinterpret_cast<const float4*>(bank1 + row)[tid];
    const float4 x = reinterpret_cast<const float4*>(in + row)[tid];

    float s[6];
    s[0] = __fmaf_rn(b0.x, b0.x, __fmaf_rn(b0.y, b0.y, __fmaf_rn(b0.z, b0.z, b0.w * b0.w)));
    s[1] = __fmaf_rn(b0.x, q.x, __fmaf_rn(b0.y, q.y, __fmaf_rn(b0.z, q.z, b0.w * q.w)));
    s[2] = __fmaf_rn(b1.x, b1.x, __fmaf_rn(b1.y, b1.y, __fmaf_rn(b1.z, b1.z, b1.w * b1.w)));
    s[3] = __fmaf_rn(b1.x, q.x, __fmaf_rn(b1.y, q.y, __fmaf_rn(b1.z, q.z, b1.w * q.w)));
    s[4] = __fmaf_rn(x.x, x.x, __fmaf_rn(x.y, x.y, __fmaf_rn(x.z, x.z, x.w * x.w)));
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const float y = wave64_sum_mix(s[j]);
        if (lane == 0) waves[j][wave] = y;
    }
    __syncthreads();
    if (wave == 0 && lane < 5) {
        float acc = 0.f;
#pragma unroll
        for (int w = 0; w < kW; ++w) acc += waves[lane][w];
        waves[lane][0] = acc;
    }
    __syncthreads();
    const float s0 = rsqrtf(waves[0][0] / kRowWidth + 1.e-6f);
    const float s1 = rsqrtf(waves[2][0] / kRowWidth + 1.e-6f);
    float w0 = waves[1][0] * s0 * rsqrtf((float) kRowWidth);
    float w1 = waves[3][0] * s1 * rsqrtf((float) kRowWidth);
    const float top = fmaxf(w0, w1);
    w0 = expf(w0 - top);
    w1 = expf(w1 - top);
    const float inv_sum = 1.f / (w0 + w1);
    const float c0 = w0 * s0 * inv_sum, c1 = w1 * s1 * inv_sum;

    float4 m;
    m.x = __fmaf_rn(c1, b1.x, c0 * b0.x);
    m.y = __fmaf_rn(c1, b1.y, c0 * b0.y);
    m.z = __fmaf_rn(c1, b1.z, c0 * b0.z);
    m.w = __fmaf_rn(c1, b1.w, c0 * b0.w);
    reinterpret_cast<float4*>(msg + row)[tid] = m;

    float ms = __fmaf_rn(m.x, m.x, __fmaf_rn(m.y, m.y, __fmaf_rn(m.z, m.z, m.w * m.w)));
    ms = wave64_sum_mix(ms);
    if (lane == 0) waves[5][wave] = ms;
    __syncthreads();
    if (wave == 0 && lane == 0) {
        float acc = 0.f;
#pragma unroll
        for (int w = 0; w < kW; ++w) acc += waves[5][w];
        waves[5][0] = acc;
    }
    __syncthreads();
    const float sx = rsqrtf(waves[4][0] / float(kRowWidth) + 1.e-6f);
    const float sm = rsqrtf(waves[5][0] / float(kRowWidth) + 1.e-6f);
    float4* out4 = reinterpret_cast<float4*>(pair + 2 * row);
    out4[tid] = make_float4(x.x * sx, x.y * sx, x.z * sx, x.w * sx);
    out4[kRowWidth / 4 + tid] = make_float4(m.x * sm, m.y * sm, m.z * sm, m.w * sm);
#else
    constexpr int kPer = kRowWidth / kThreads, kWaves = kThreads / 32;
    __shared__ float waves[6][kWaves];
    float b0[kPer], b1[kPer], x[kPer], q[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = threadIdx.x + k * kThreads;
        q[k] = query[i];
        b0[k] = bank0[row + i];
        b1[k] = bank1[row + i];
        x[k] = in[row + i];
    }
    float sums[6] = {};   // the banks' squares and products with the query, the stream's squares, the message's
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        sums[0] = __fmaf_rn(b0[k], b0[k], sums[0]);
        sums[1] = __fmaf_rn(b0[k], q[k], sums[1]);
        sums[2] = __fmaf_rn(b1[k], b1[k], sums[2]);
        sums[3] = __fmaf_rn(b1[k], q[k], sums[3]);
        sums[4] += x[k] * x[k];
    }
    const int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
    const auto reduce = [&](int first, int count) {
        for (int j = first; j < first + count; ++j) {
            const float y = wave_reduce<false>(sums[j]);
            if (lane == 0) waves[j][wave] = y;
        }
        __syncthreads();
        if (wave == 0)
            for (int j = first; j < first + count; ++j) {
                const float y = wave_reduce<false>(lane < kWaves ? waves[j][lane] : 0.f);
                if (lane == 0) waves[j][0] = y;
            }
        __syncthreads();
        for (int j = first; j < first + count; ++j) sums[j] = waves[j][0];
    };
    reduce(0, 5);
    const float s0 = rsqrtf(sums[0] / kRowWidth + 1.e-6f), s1 = rsqrtf(sums[2] / kRowWidth + 1.e-6f);
    float w0 = sums[1] * s0 * rsqrtf((float) kRowWidth), w1 = sums[3] * s1 * rsqrtf((float) kRowWidth);
    const float top = fmaxf(w0, w1);
    w0 = expf(w0 - top);
    w1 = expf(w1 - top);
    const float sum = w0 + w1;
    float m[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        float val = __fmaf_rn(w0 * s0, b0[k], 0.f);
        val = __fmaf_rn(w1 * s1, b1[k], val);
        m[k] = val / sum;
        msg[row + threadIdx.x + k * kThreads] = m[k];
        sums[5] += m[k] * m[k];
    }
    __syncthreads();
    reduce(5, 1);
    const float sx = 1.0f / sqrtf(sums[4] / float(kRowWidth) + 1.e-6f), sm = 1.0f / sqrtf(sums[5] / float(kRowWidth) + 1.e-6f);
    float* out = pair + 2 * row;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = threadIdx.x + k * kThreads;
        out[i] = x[k] * sx;
        out[kRowWidth + i] = m[k] * sm;
    }
#endif
}

__global__ void control_in_kernel(const float* bank0, const float* bank1, const float* query, const float* in,
                                  float* msg, float* pair) {
    control_in_row(bank0, bank1, query, in, msg, pair, (size_t) blockIdx.x * kRowWidth);
}

// The streams' control (Inkling2::heads()): stream blockIdx.y's, each row as the one-stream kernels'.
struct Control3 {
    const float* query[3];
    float* cur[3];
    float* msg[3];
    float* pair[3];
    const float* gain[3];
    const float* forget[3];
    const float* rank[3];
};
__global__ void control_in3_kernel(const float* bank0, const float* bank1, Control3 c) {
    const int s = blockIdx.y;
    control_in_row(bank0, bank1, c.query[s], c.cur[s], c.msg[s], c.pair[s], (size_t) blockIdx.x * kRowWidth);
}

#ifdef STRATA_HIP_GFX906
// STRATA_INKLING_MIX_CTRL_FUSE=1: fuse mix_row() + control_in_row() on loop visits (exact same 576-thread
// float4 ownership, keeps cur in register without a second launch or global round-trip; bitwise identical).
__device__ __forceinline__ void mix_control_in_row(const MixInputs& in, int count, const float* mquery, float* cur_out,
                                                   const float* bank0, const float* bank1, const float* cquery,
                                                   float* msg, float* pair, size_t row) {
    constexpr int kMost = 5, kW = kMixThreads / 64;
    const int tid = threadIdx.x;
    const int lane = tid & 63, wave = tid >> 6;
    __shared__ float waves[2 * kMost][kW];
    __shared__ float cwaves[6][kW];

    // 1. Exact mix_row()
    const float4 mq = reinterpret_cast<const float4*>(mquery)[tid];
    float4 xv[kMost];
    for (int j = 0; j < count; ++j) {
        const float4 xj = reinterpret_cast<const float4*>(in.x[j] + row)[tid];
        xv[j] = xj;
        float ss = __fmaf_rn(xj.x, xj.x, __fmaf_rn(xj.y, xj.y, __fmaf_rn(xj.z, xj.z, xj.w * xj.w)));
        float dot = __fmaf_rn(xj.x, mq.x, __fmaf_rn(xj.y, mq.y, __fmaf_rn(xj.z, mq.z, xj.w * mq.w)));
        ss = wave64_sum_mix(ss);
        dot = wave64_sum_mix(dot);
        if (lane == 0) {
            waves[2 * j][wave] = ss;
            waves[2 * j + 1][wave] = dot;
        }
    }
    __syncthreads();
    if (wave == 0 && lane < 2 * count) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < kW; ++w) s += waves[lane][w];
        waves[lane][0] = s;
    }
    __syncthreads();
    float scale[kMost], wm[kMost], mtop = -INFINITY;
    for (int j = 0; j < count; ++j) {
        scale[j] = rsqrtf(waves[2 * j][0] / kRowWidth + 1.e-6f);
        wm[j] = waves[2 * j + 1][0] * scale[j] * rsqrtf((float) kRowWidth);
        mtop = fmaxf(mtop, wm[j]);
    }
    float msum = 0.f;
    for (int j = 0; j < count; ++j) msum += (wm[j] = expf(wm[j] - mtop));
    const float minv_sum = 1.f / msum;
    float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int j = 0; j < count; ++j) {
        const float c = wm[j] * scale[j] * minv_sum;
        x.x = __fmaf_rn(c, xv[j].x, x.x);
        x.y = __fmaf_rn(c, xv[j].y, x.y);
        x.z = __fmaf_rn(c, xv[j].z, x.z);
        x.w = __fmaf_rn(c, xv[j].w, x.w);
    }
    reinterpret_cast<float4*>(cur_out + row)[tid] = x;

    // 2. Exact control_in_row() with x already in register
    const float4 q = reinterpret_cast<const float4*>(cquery)[tid];
    const float4 b0 = reinterpret_cast<const float4*>(bank0 + row)[tid];
    const float4 b1 = reinterpret_cast<const float4*>(bank1 + row)[tid];

    float s[5];
    s[0] = __fmaf_rn(b0.x, b0.x, __fmaf_rn(b0.y, b0.y, __fmaf_rn(b0.z, b0.z, b0.w * b0.w)));
    s[1] = __fmaf_rn(b0.x, q.x, __fmaf_rn(b0.y, q.y, __fmaf_rn(b0.z, q.z, b0.w * q.w)));
    s[2] = __fmaf_rn(b1.x, b1.x, __fmaf_rn(b1.y, b1.y, __fmaf_rn(b1.z, b1.z, b1.w * b1.w)));
    s[3] = __fmaf_rn(b1.x, q.x, __fmaf_rn(b1.y, q.y, __fmaf_rn(b1.z, q.z, b1.w * q.w)));
    s[4] = __fmaf_rn(x.x, x.x, __fmaf_rn(x.y, x.y, __fmaf_rn(x.z, x.z, x.w * x.w)));
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const float y = wave64_sum_mix(s[j]);
        if (lane == 0) cwaves[j][wave] = y;
    }
    __syncthreads();
    if (wave == 0 && lane < 5) {
        float acc = 0.f;
#pragma unroll
        for (int w = 0; w < kW; ++w) acc += cwaves[lane][w];
        cwaves[lane][0] = acc;
    }
    __syncthreads();
    const float s0 = rsqrtf(cwaves[0][0] / kRowWidth + 1.e-6f);
    const float s1 = rsqrtf(cwaves[2][0] / kRowWidth + 1.e-6f);
    float w0 = cwaves[1][0] * s0 * rsqrtf((float) kRowWidth);
    float w1 = cwaves[3][0] * s1 * rsqrtf((float) kRowWidth);
    const float top = fmaxf(w0, w1);
    w0 = expf(w0 - top);
    w1 = expf(w1 - top);
    const float inv_sum = 1.f / (w0 + w1);
    const float c0 = w0 * s0 * inv_sum, c1 = w1 * s1 * inv_sum;

    float4 m;
    m.x = __fmaf_rn(c1, b1.x, c0 * b0.x);
    m.y = __fmaf_rn(c1, b1.y, c0 * b0.y);
    m.z = __fmaf_rn(c1, b1.z, c0 * b0.z);
    m.w = __fmaf_rn(c1, b1.w, c0 * b0.w);
    reinterpret_cast<float4*>(msg + row)[tid] = m;

    float ms = __fmaf_rn(m.x, m.x, __fmaf_rn(m.y, m.y, __fmaf_rn(m.z, m.z, m.w * m.w)));
    ms = wave64_sum_mix(ms);
    if (lane == 0) cwaves[5][wave] = ms;
    __syncthreads();
    if (wave == 0 && lane == 0) {
        float acc = 0.f;
#pragma unroll
        for (int w = 0; w < kW; ++w) acc += cwaves[5][w];
        cwaves[5][0] = acc;
    }
    __syncthreads();
    const float sx = rsqrtf(cwaves[4][0] / float(kRowWidth) + 1.e-6f);
    const float sm = rsqrtf(cwaves[5][0] / float(kRowWidth) + 1.e-6f);
    float4* out4 = reinterpret_cast<float4*>(pair + 2 * row);
    out4[tid] = make_float4(x.x * sx, x.y * sx, x.z * sx, x.w * sx);
    out4[kRowWidth / 4 + tid] = make_float4(m.x * sm, m.y * sm, m.z * sm, m.w * sm);
}

__global__ void mix_control_in_kernel(MixInputs in, int count, const float* mquery, float* cur_out,
                                      const float* bank0, const float* bank1, const float* cquery,
                                      float* msg, float* pair) {
    mix_control_in_row(in, count, mquery, cur_out, bank0, bank1, cquery, msg, pair, (size_t) blockIdx.x * kRowWidth);
}

__global__ void mix_control_in3_kernel(MixInputs in, int count, Streams3 ms,
                                       const float* bank0, const float* bank1, Control3 c) {
    const int s = blockIdx.y;
    mix_control_in_row(in, count, ms.query[s], ms.out[s], bank0, bank1, c.query[s], c.msg[s], c.pair[s],
                       (size_t) blockIdx.x * kRowWidth);
}
#endif

// The third: the gain and the forget gate of each value from the rank, eight lanes an output, applied to the stream.
__device__ __forceinline__ void control_out_body(const float* gain, const float* forget, const float* rank,
                                                 const float* msg, float* cur, int ranks) {
    const int lane = threadIdx.x % 8, o = blockIdx.x * 32 + threadIdx.x / 8, t = blockIdx.y;
    const float4* g = reinterpret_cast<const float4*>(gain + (size_t) o * ranks);
    const float4* f = reinterpret_cast<const float4*>(forget + (size_t) o * ranks);
    const float4* r = reinterpret_cast<const float4*>(rank + (size_t) t * ranks);
    float ga = 0.f, fa = 0.f;
    for (int i = lane; i < ranks / 4; i += 8) {
        const float4 a = g[i], b = f[i], x = r[i];
        ga += a.x * x.x + a.y * x.y + a.z * x.z + a.w * x.w;
        fa += b.x * x.x + b.y * x.y + b.z * x.z + b.w * x.w;
    }
#pragma unroll
    for (int d = 4; d > 0; d /= 2) {
        ga += __shfl_xor(ga, d, 32);
        fa += __shfl_xor(fa, d, 32);
    }
    if (lane == 0) {
        const size_t at = (size_t) t * kRowWidth + o;
        cur[at] = (1.f - tanhf(fa)) * cur[at] + tanhf(ga) * msg[at];
    }
}

__global__ void __launch_bounds__(256) control_out_kernel(const float* gain, const float* forget, const float* rank,
                                                          const float* msg, float* cur, int ranks) {
    control_out_body(gain, forget, rank, msg, cur, ranks);
}

__global__ void __launch_bounds__(256) control_out3_kernel(Control3 c, int ranks) {
    const int s = blockIdx.z;
    control_out_body(c.gain[s], c.forget[s], c.rank[s], c.msg[s], c.cur[s], ranks);
}

}  // namespace

void Inkling2::load_geometry() {
    int device = 0;
    cudaDeviceProp p{};
    ck(cudaGetDevice(&device), "Inkling device");
    ck(cudaGetDeviceProperties(&p, device), "Inkling device properties");
    const bool gfx906 = kernels::gfx_arch_is(p.gcnArchName, "gfx906");
    bool supported_arch = gfx906;
    for (const char* gcn : {"gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"})
        supported_arch = supported_arch || kernels::gfx_arch_is(p.gcnArchName, gcn);
    if (!supported_arch) throw std::runtime_error("Inkling-2-Mini requires a HIP gfx906 or RDNA2 card");
    if (opt_.batch || !opt_.layer_split.empty() || opt_.tail_skip || opt_.skip_weight > 0.f)
        throw std::runtime_error("Inkling-2-Mini requires one sequence and one card, without expert skipping");
    // --draft-model is the DSpark drafter, a model of its own and not a file with a block load_shape() would look
    // for.  Without it the model checks no guesses: a window of one token.
    std::unique_ptr<Shards> drafter = std::move(dg_);
    opt_.draft = false;
    const char* flash = std::getenv("STRATA_INKLING_FLASH");
    flash_ = flash && flash[0] == '1' && flash[1] == '\0';
    if (flash_) std::fputs("strata-poly: STRATA_INKLING_FLASH=1: fused attention for prompt batches\n", stderr);
    const char* prefetch = std::getenv("STRATA_INKLING_PREFETCH");
    prefetch_early_ = prefetch && prefetch[0] == '1' && prefetch[1] == '\0';
    if (prefetch_early_)
        std::fputs("strata-poly: STRATA_INKLING_PREFETCH=1: prompt experts queued before router readback\n", stderr);
    const char* decode = std::getenv("STRATA_INKLING_DECODE");
    decode_ = decode && decode[0] == '1' && decode[1] == '\0';
#if defined(STRATA_HIP_GFX906)
    const char* factored = std::getenv("STRATA_INKLING_DECODE_FACTORED");
    factored_decode_ = gfx906 && tp_size_ == 2 && q8() && decode_ && factored && std::strcmp(factored, "1") == 0;
#endif
    wave_gemv_ = head_warps_ = situ_q8_ = decode_;
    // A miss in a once-visited layer (0-5, 18-23) stalls the card: the CPU's experts there are 2-3 a visit and the
    // card waited 200-340 us a visit for them, against 5-70 us in the core visits, whose card experts hide one or two.
    // Counted per visit, the core's uses (three a token) held ~1,200 of the ~1,330 slots.  Weighed twice, forced decode
    // 39.07 -> 40.15 tok/s and 16K 30.87 -> 32.65 (RX 6700 XT; 1.5 to 3 measured the same).
    if (decode_) {
        use_w_.assign(kPhysical, 1.f);
        for (int il = 0; il < kPhysical; ++il)
            if (il < kStart || il >= kStart + kCore) use_w_[(size_t) il] = 2.f;
    }
    if (decode_) std::fputs("strata-poly: STRATA_INKLING_DECODE=1: decode kernels for a few tokens at a time\n", stderr);
    const char* norm_q8 = std::getenv("STRATA_INKLING_NORM_Q8");
    norm_q8_ = decode_ && norm_q8 && norm_q8[0] == '1' && norm_q8[1] == '\0';
#if defined(STRATA_HIP_GFX906)
    const char* mix_rms = std::getenv("STRATA_INKLING_MIX_RMS_FUSE");
    mix_rms_fuse_ = decode_ && gfx906 && p.warpSize == 64 && mix_rms && std::strcmp(mix_rms, "1") == 0;
    const char* pfuse = std::getenv("STRATA_INKLING_PREFILL_FUSE");
    prefill_fuse_ = gfx906 && p.warpSize == 64 && !opt_.dump && pfuse && std::strcmp(pfuse, "1") == 0;
    if (prefill_fuse_ && tp_rank_ == 0 && !prefill_only_)
        std::fputs("strata-poly: STRATA_INKLING_PREFILL_FUSE=1: fused prefill mix+rms+mmq, Q8_1 reuse & OOP ShortConv\n",
                   stderr);
    const char* group_ballot = std::getenv("STRATA_INKLING_MOE_GROUP_BALLOT");
    moe_group_ballot_ = gfx906 && p.warpSize == 64 && group_ballot && std::strcmp(group_ballot, "1") == 0;
#endif
    if (norm_q8_)
        std::fputs("strata-poly: STRATA_INKLING_NORM_Q8=1: a visit's stream norms and their Q8_1 in one launch\n",
                   stderr);
    const char* tail_moe = std::getenv("STRATA_INKLING_TAIL_MOE");
    moe_tail_ = decode_ && tail_moe && tail_moe[0] == '1' && tail_moe[1] == '\0';
    if (moe_tail_)
        std::fputs("strata-poly: STRATA_INKLING_TAIL_MOE=1: a fused visit's routed experts summed in its tail launch\n",
                   stderr);
    const char* q8_cols = std::getenv("STRATA_INKLING_Q8_COLS");
    q8_cols_ = decode_ && q8_cols && q8_cols[0] == '1' && q8_cols[1] == '\0';
    if (q8_cols_)
        std::fputs("strata-poly: STRATA_INKLING_Q8_COLS=1: a window's Q8_0 projections sum as one token's do\n", stderr);
    const char* route_dual = std::getenv("STRATA_INKLING_ROUTE_DUAL");
    route_dual_ = decode_ && route_dual && route_dual[0] == '1' && route_dual[1] == '\0';
    if (route_dual_)
        std::fputs("strata-poly: STRATA_INKLING_ROUTE_DUAL=1: the router's copies to the host ride its producers\n",
                   stderr);
    const char* heads_kv_rows = std::getenv("STRATA_INKLING_HEADS_KV_ROWS");
    heads_kv_rows_ = decode_ && heads_kv_rows && heads_kv_rows[0] == '1' && heads_kv_rows[1] == '\0';
    if (heads_kv_rows_)
        std::fputs("strata-poly: STRATA_INKLING_HEADS_KV_ROWS=1: fused K/V conv, head norms and Q8 store for windows\n",
                   stderr);
    const char* q4_dense_cols = std::getenv("STRATA_INKLING_Q4_0_DENSE_COLS");
    q4_0_dense_cols_ = decode_ && q4_dense_cols && q4_dense_cols[0] == '1' && q4_dense_cols[1] == '\0';
    if (q4_0_dense_cols_)
        std::fputs("strata-poly: STRATA_INKLING_Q4_0_DENSE_COLS=1: grouped multi-row Q4_0 dense projections\n",
                   stderr);
    const char* mix_ctrl = std::getenv("STRATA_INKLING_MIX_CTRL_FUSE");
    mix_ctrl_fuse_ = decode_ && mix_ctrl && mix_ctrl[0] == '1' && mix_ctrl[1] == '\0';
    if (mix_ctrl_fuse_)
        std::fputs("strata-poly: STRATA_INKLING_MIX_CTRL_FUSE=1: fused depth_mix + control_in on loop visits\n",
                   stderr);
    const char* gemv_opt = std::getenv("STRATA_GEMV_ROWS_SHARED_OPT");
    kvproj_fuse_ = decode_ && gemv_opt && gemv_opt[0] == '1' && gemv_opt[1] == '\0';
    const char* df_fast = std::getenv("STRATA_DFLASH2_FAST");
    df_fast_ = df_fast && df_fast[0] == '1' && df_fast[1] == '\0';
    if (!q8() && opt_.kv_format != KvFmt::kFp16)
        throw std::runtime_error("Inkling-2-Mini supports Q8_0 or FP16 KV");
    const int window = opt_.window;
    opt_.window = 1;
    load_shape();

    const MetaValue* arch_mv = gg_->get("general.architecture");
    const bool loop_arch = arch_mv && arch_mv->s == "inkling2_loop";
    const auto need = [](bool ok, const char* key) {
        if (!ok) throw std::runtime_error(std::string("not an Inkling-2-Mini layout: ") + key);
    };
    const auto integer = [&](const char* key, int expected) { need(shape_i(key) == expected, key); };
    const int expected_blocks = loop_arch ? kPhysical : kVisits;
    need(n_layer_ == expected_blocks && n_embd_ == kWidth && n_expert_ == 128 && n_used_ == 8 && n_ff_exp_ == 1152,
         "layers, width or experts");
    integer(loop_arch ? "loop.count" : "num_loops", 3);
    integer(loop_arch ? "loop.start" : "loop.core_start", kStart);
    integer(loop_arch ? "loop.end" : "loop.core_end", kStart + kCore);
    integer("loop.gate_rank", kRank);
    integer("attn_res.block_size", kBlock);
    integer("attention.sliding_window", kWindow);
    integer("attention.relative_length", kRelative);
    integer("attention.relative_extent", kWindow);
    integer("shortconv.l_cache", kTaps);
    integer("rope.dimension_count", 0);
    for (const char* key : {"attention.key_length", "attention.value_length", "attention.key_length_swa",
                            "attention.value_length_swa"}) integer(key, kHead);
    const auto array = [&](const char* key, auto expected) {
        const auto* value = meta_in(*gg_, key);
        need(value && value->count == (uint64_t) expected_blocks && value->items.size() == (size_t) expected_blocks,
             key);
        for (int i = 0; i < expected_blocks; ++i) need(value->items[i].u == (uint64_t) expected(i), key);
    };
    array("attention.head_count", [](int) { return kHeads; });
    array("attention.head_count_kv", [](int) { return kKv; });
    array("attention.sliding_window_pattern", [](int i) { return local(i); });
    padded_vocab_ = n_vocab_;
    n_vocab_ = shape_i("vocab_size_unpadded");
    const auto* tokens = gg_->get("tokenizer.ggml.tokens");
    need(n_vocab_ == 200058 && padded_vocab_ == 201024 && tokens && tokens->count == (uint64_t) n_vocab_, "vocabulary");
    need(gg_->find("output.weight") && !draft_, "output or draft");
    qk_clip_ = shape_f("attention.qk_clip");
    r_clip_ = shape_f("attention.relative_clip");
    floor_ = shape_i("attention.log_scale_floor");
    alpha_ = shape_f("attention.log_scale_alpha");
    expert_scale_ = shape_f("expert_weights_scale");
    cap_ = shape_f("final_logit_softcapping");
    situ_ = {shape_f("activation.situ_beta"), shape_f("activation.situ_linear_beta")};
    for (float value : {rms_eps_, qk_clip_, r_clip_, expert_scale_, cap_, situ_.beta, situ_.linear_beta})
        need(std::isfinite(value) && value > 0.f, "numerical parameters");
    need(std::isfinite(alpha_) && alpha_ >= 0.f && floor_ > 0, "log scale");
    n_layer_ = kPhysical;
    looped_ = true;
    clamp_exp_.assign(kPhysical + 1, INFINITY);
    if (drafter) {
        // a window's guesses need the short convolutions' state after each of its tokens, which the decode switch's
        // kernels leave (layer_head(), layer_tail())
        if (!decode_) throw std::runtime_error("the DSpark drafter of Inkling-2-Mini needs STRATA_INKLING_DECODE=1");
        // its block of 8 rows goes through the scratch that holds a call's rows
        if (opt_.prompt_batch < kDsBlock)
            throw std::runtime_error("the DSpark drafter of Inkling-2-Mini needs --prompt-batch 8 or more");
        load_drafter_shape(*drafter);
        dg_ = std::move(drafter);
        block_ = dg_.get();   // the file the drafter's tensors are read from (home())
        draft_ = true;
        block_layer_ = false;
        cols_kernels_ = true;
        opt_.draft = true;
        opt_.window = window > 0 ? std::min(window, kDsBlock) : kDsBlock;
        ds_rows_ = std::min(kDsBlock - 1, opt_.window - 1);
        drafter_block_ = df2_;
        std::fprintf(stderr, "strata-poly: %s drafter %s: up to %d guesses a window\n", df2_ ? "DFlash2" : "DSpark",
                     opt_.draft_model.c_str(), ds_rows_);
    }
    if (tp_size_ == 2) {
        if (tp_rank_ == 1 && window > 1) opt_.window = std::min(window, kDsBlock);
        n_heads_ = kHeads / 2;
        n_kv_ = kKv / 2;
        n_query_ = n_heads_ * kHead;
        n_key_ = n_kv_ * kHead;
        conv_floats_ = (kTaps - 1) * (2 * n_key_ + 2 * kWidth);
        n_ff_exp_ /= 2;
        moe_tail_ = false;   // all_reduce #2 needs the summed local MoE output in fo_ before conv_add
    }
}

namespace {
// All draft launches switch devices on the calling thread. The target stream and its workspace never move.
struct DfOnDevice {
    int previous = 0;
    explicit DfOnDevice(int device) {
        ck(cudaGetDevice(&previous), "DFlash2 current device");
        if (device != previous) ck(cudaSetDevice(device), "DFlash2 select device");
    }
    ~DfOnDevice() { (void) cudaSetDevice(previous); }
};

// upload() uses the Engine's two-pass dense arena and stream. Point it at a separate arena during startup only.
struct DfWeightArena {
    uint8_t* data = nullptr;
    size_t bytes = 0, at = 0, bf16_max = 0;
};
}  // namespace

struct Inkling2::DfGpu23 {
    cudaStream_t s2 = nullptr, s3 = nullptr, tx0 = nullptr;
    struct TapsEvents {
        cudaEvent_t ready0 = nullptr, pushed = nullptr, read2 = nullptr, done2 = nullptr;
        bool used = false;
    };
    // A 2048-row tail uses up to eight 256-row pushes per call. Do not re-record a HIP event while it is pending.
    std::array<TapsEvents, 16> taps;
    size_t next_taps = 0;
    std::array<const TapsEvents*, 2> last_taps{};  // each staging pair has its own lifetime (kept, provisional)
    cudaEvent_t h2_ready = nullptr, h3_done = nullptr;
    DfWeightArena weights;
    LinWorkspace lin{};
    std::unique_ptr<Mmq> mmq;
    // On their allocating devices, independent of the target Engine's allocation list.
    std::array<std::vector<void*>, 4> memory;
    float *tap_stage0 = nullptr, *prov_taps0 = nullptr, *prov_taps2 = nullptr;
    __half *prov_k2 = nullptr, *prov_v2 = nullptr;
    bool async_ctx = false, prov_valid = false;
    int64_t prov_pos = -1;
    int prov_n = 0;

    void* alloc(size_t bytes, const char* what) {
        int device = 0;
        ck(cudaGetDevice(&device), "DFlash2 allocation device");
        auto& owned = memory.at((size_t) device);
        owned.push_back(nullptr);
        ck(cudaMalloc(&owned.back(), std::max<size_t>(bytes, 256)), what);
        return owned.back();
    }

    ~DfGpu23() {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        // Drain the entire dependency chain before releasing any peer-visible storage or pinned results.
        for (auto [device, stream] : {std::pair{0, tx0}, std::pair{2, s2}, std::pair{3, s3}})
            if (stream && cudaSetDevice(device) == cudaSuccess) (void) cudaStreamSynchronize(stream);
        for (int device : {0, 2, 3}) {
            if (cudaSetDevice(device) != cudaSuccess) continue;
            if (device == 0) {
                for (const auto& t : taps)
                    for (auto ev : {t.ready0, t.pushed}) if (ev) (void) cudaEventDestroy(ev);
                if (tx0) (void) cudaStreamDestroy(tx0);
            } else if (device == 2) {
                for (const auto& t : taps)
                    for (auto ev : {t.read2, t.done2}) if (ev) (void) cudaEventDestroy(ev);
                if (h2_ready) (void) cudaEventDestroy(h2_ready);
                mmq.reset();
                blas_destroy(lin.blas);
                if (s2) (void) cudaStreamDestroy(s2);
            } else {
                if (h3_done) (void) cudaEventDestroy(h3_done);
                if (s3) (void) cudaStreamDestroy(s3);
            }
            for (void* ptr : memory[(size_t) device]) if (ptr) (void) cudaFree(ptr);
        }
        (void) cudaSetDevice(previous);
    }
};

Inkling2::Inkling2() = default;

cudaStream_t Inkling2::df_stream() const { return df_gpu23_ ? df_gpu23_->s2 : cs(st_); }

Inkling2::~Inkling2() {
#if defined(STRATA_HIP_GFX906)
    stage_b_.reset();
    destroy_phase_dualtp_handles();
#endif
    stop_peer_thread();
    if (df_gpu23_) {
        int previous = 0;
        if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device_) == cudaSuccess) {
            if (st_) (void) cudaStreamSynchronize(cs(st_));
            df_gpu23_.reset();
            (void) cudaSetDevice(previous);
        }
    }
}

std::unique_ptr<Engine> Inkling2::open_tp2(const Options& o) {
    struct DevGuard {
        int prev = 0;
        bool ok = false;
        DevGuard() { ok = (cudaGetDevice(&prev) == cudaSuccess); }
        ~DevGuard() { if (ok) { (void) cudaSetDevice(prev); } }
    } guard;
    const int max_floats = std::max(o.prompt_batch, 8) * kWidth;
    auto comm = Tp2Comm::create(0, 1, max_floats);

    auto r0 = std::make_unique<Inkling2>();
    r0->device_ = 0;
    r0->split_ = true;
    r0->tp_size_ = 2;
    r0->tp_rank_ = 0;
    r0->tp2_ = comm;

    auto r1 = std::make_unique<Inkling2>();
    r1->device_ = 1;
    r1->split_ = true;
    r1->tp_size_ = 2;
    r1->tp_rank_ = 1;
    r1->tp2_ = comm;

    Options o0 = o;
    o0.tp = 2;
    Options o1 = o;
    o1.tp = 2;
    o1.draft = false;
    o1.draft_model.clear();
    if (o.draft && !o.draft_model.empty()) o1.window = o.window > 0 ? std::min(o.window, kDsBlock) : kDsBlock;

#if defined(STRATA_HIP_GFX906)
    const char* compact = std::getenv("STRATA_INKLING_COMPACT_SWA");
    if (compact && std::strcmp(compact, "1") == 0) {
        const auto gfx906 = [](int device) {
            cudaDeviceProp props{};
            return cudaGetDeviceProperties(&props, device) == cudaSuccess && props.warpSize == 64 &&
                   kernels::gfx_arch_is(props.gcnArchName, "gfx906");
        };
        // Match the resident production TP2 path, not growing-VMM expert/cache policy.
        const bool supported = o.kv_format == KvFmt::kQ8_0 && !o.kv_grow && !o.dump &&
                               gfx906(r0->device_) && gfx906(r1->device_);
        r0->compact_swa_asked_ = r1->compact_swa_asked_ = supported;
        if (!supported)
            std::fputs("strata-poly: STRATA_INKLING_COMPACT_SWA=1 requires two gfx906 owners, Q8_0 --kv-whole "
                       "and no layer dump; using stock rings\n", stderr);
    }
#endif
    ck(cudaSetDevice(0), "tp2 setDevice 0");
    r0->load(o0);
    ck(cudaSetDevice(1), "tp2 setDevice 1");
    r1->load(o1);
    if (r0->df2_ && r0->df_private_head_) {   // each owner loads its private half; rank 1 has no drafter GGUF
        const int lo1 = r1->tp2_vocab_lo_;
        r1->df_head_q4_ = r0->df_head_q4_;
        r1->df_head_fuse_ = r0->df_head_fuse_;
        if (r0->df_gpu23_) {
            { DfOnDevice on3(3); r1->df_head_load(lo1, kDfVocab - lo1, r0.get()); }
            { DfOnDevice on2(2); r0->df_head_load(0, lo1, r0.get()); }
        } else {
            r1->df_head_load(lo1, kDfVocab - lo1);
            ck(cudaSetDevice(0), "tp2 setDevice 0");
            r0->df_head_load(0, lo1);
        }
        ck(cudaSetDevice(0), "tp2 setDevice 0");
        std::fprintf(stderr, "strata-poly: DFlash2 private head rows [0, %d) and [%d, %d) as %s%s\n",
                     lo1, lo1, kDfVocab, r0->df_head_q4_ ? "Q4_0" : "int8", r0->df_head_fuse_ ? " with fused top-16" : "");
        ck(cudaSetDevice(1), "tp2 setDevice 1");
    }
#if defined(STRATA_HIP_GFX906)
    const auto enabled = [](const char* name) {
        const char* v = std::getenv(name);
        return v && !std::strcmp(v, "1");
    };
    const bool want_decode = enabled("STRATA_INKLING_GPU_ROUTE");
    const bool want_prefill_any = enabled("STRATA_INKLING_GPU_PREFILL_ANY");
    const bool want_prefill = enabled("STRATA_INKLING_GPU_PREFILL") || want_prefill_any;
    if (want_decode || want_prefill) {
        const bool shared = gpu_route_host_expf() && r0->gpu_route_supported() && r1->gpu_route_supported();
        const bool decode = shared && want_decode && Tp2Comm::small_push_enabled() &&
                            r0->gpu_decode_supported() && r1->gpu_decode_supported();
        const int min_rows = want_prefill_any ? kStageRows : 2047;
        const bool prefill = shared && want_prefill && r0->rows_ >= min_rows && r1->rows_ >= min_rows;
        if (decode) {
            r0->gpu_down_sum_ = r1->gpu_down_sum_ = enabled("STRATA_INKLING_DOWN_SUM");
            // The ordinary down path materializes at most kSumRows; only the CTA sum bypasses that plane.
            // Short scratch (<8 rows) can make the host path use single-expert rather than slots Q4 dots.
            if (r0->gpu_down_sum_ && r0->rows_ >= 8 && r1->rows_ >= 8 && enabled("STRATA_INKLING_GPU_ROUTE_WIDE"))
                r0->gpu_route_rows_ = r1->gpu_route_rows_ = 8;
        }
        if (decode || prefill) {
            r1->init_gpu_route(decode, prefill);
            ck(cudaSetDevice(0), "tp2 setDevice 0");
            r0->init_gpu_route(decode, prefill);
        }
        if (decode) {
            std::fprintf(stderr, "strata-poly: STRATA_INKLING_GPU_ROUTE=1: exact GPU routing for TP2 T=1..%d\n",
                         r0->gpu_route_rows_);
        } else if (want_decode) {
            std::fputs("strata-poly: STRATA_INKLING_GPU_ROUTE=1 unavailable (needs FP32 small push, fully resident "
                       "Q4 TP2 decode and default x86 glibc FMA FP mode)\n", stderr);
        }
        if (want_prefill && !prefill)
            std::fputs("strata-poly: STRATA_INKLING_GPU_PREFILL=1 exact selector unavailable; using host routing\n", stderr);
    }
    if (enabled("STRATA_INKLING_PREFILL_DEPTH_SHARE")) {
        const auto gfx906 = [](int device) {
            cudaDeviceProp props{};
            return cudaGetDeviceProperties(&props, device) == cudaSuccess &&
                   kernels::gfx_arch_is(props.gcnArchName, "gfx906");
        };
        const bool share = gfx906(r0->device_) && gfx906(r1->device_);
        r0->prefill_depth_share_ = r1->prefill_depth_share_ = share;
        if (share)
            std::fputs("strata-poly: STRATA_INKLING_PREFILL_DEPTH_SHARE=1: TP2 SWA message input reuse at T=2047/2048\n",
                       stderr);
    }
    if (enabled("STRATA_INKLING_KV_HELPERS")) {
        if (!r0->factored_decode_ || !r1->factored_decode_ || !enabled("STRATA_INKLING_DECODE_DPP") ||
            r0->opt_.max_context != r1->opt_.max_context || !r0->attq_ || !r1->attq_) {
            std::fputs("strata-poly: STRATA_INKLING_KV_HELPERS=1 unavailable: both TP2 ranks need gfx906, Q8_0 KV, "
                       "DECODE, DECODE_DPP, DECODE_FACTORED and matching context limits; using ordinary TP2\n", stderr);
        } else {
            try {
                const bool wide_decode = enabled("STRATA_INKLING_KV_HELPERS_W8");
                auto mirrors = inkling::kv_helpers::create_pair(r0->opt_.max_context, {cs(r0->st_), cs(r1->st_)},
                                                                wide_decode);
                if (wide_decode)
                    std::fputs("strata-poly: STRATA_INKLING_KV_HELPERS_W8=1: up to eight full-attention decode rows\n",
                               stderr);
                if (enabled("STRATA_INKLING_FLASH_HELPERS") && r0->flash_ && r1->flash_ &&
                    enabled("STRATA_INKLING_FLASH_LEAN") && enabled("STRATA_INKLING_LANE_PROB") &&
                    !enabled("STRATA_INKLING_FLASH_H2")) {
                    try {
                        inkling::kv_helpers::enable_flash_rows(mirrors);
                        std::fputs("strata-poly: STRATA_INKLING_FLASH_HELPERS=1: split bulk full Flash query tiles\n", stderr);
                    } catch (const std::runtime_error& e) {
                        (void) cudaGetLastError();
                        std::fprintf(stderr, "strata-poly: bulk Flash helpers unavailable: %s; keeping decode helpers\n", e.what());
                    }
                }
                r0->kv_helper_ = std::move(mirrors[0]);
                r1->kv_helper_ = std::move(mirrors[1]);
                std::fputs("strata-poly: STRATA_INKLING_KV_HELPERS=1: TP2 full-attention decode on GPUs 0/2 and 1/3\n",
                           stderr);
            } catch (const std::runtime_error& e) {
                (void) cudaGetLastError();
                std::fprintf(stderr, "strata-poly: STRATA_INKLING_KV_HELPERS=1 unavailable: %s; using ordinary TP2\n",
                             e.what());
            }
        }
    }
#endif
    ck(cudaSetDevice(0), "tp2 setDevice 0");
    r0->peer_ = std::move(r1);
    r0->start_peer_thread();
#if defined(STRATA_HIP_GFX906)
    r0->phase_dualtp_init(o);
#endif
    return r0;
}

void Inkling2::start_peer_thread() {
    if (!peer_) return;
    bool started = false;
    std::exception_ptr start_err;
    peer_thread_ = std::thread([this, &started, &start_err] {
        try {
            ck(cudaSetDevice(peer_->device_), "tp2 peer setDevice");
        } catch (...) {
            {
                std::lock_guard<std::mutex> lk(peer_mu_);
                start_err = std::current_exception();
                started = true;
            }
            peer_cv_.notify_one();
            return;
        }
        {
            std::lock_guard<std::mutex> lk(peer_mu_);
            started = true;
        }
        peer_cv_.notify_one();
        while (true) {
            PeerCmd cmd = PeerCmd::kIdle;
            int n = 0;
            size_t snap_idx = 0;
            {
                std::unique_lock<std::mutex> lk(peer_mu_);
                peer_cv_.wait(lk, [&] { return peer_cmd_ != PeerCmd::kIdle; });
                cmd = peer_cmd_;
                n = peer_arg_n_;
                snap_idx = peer_arg_snap_;
                peer_cmd_ = PeerCmd::kIdle;
            }
            if (cmd == PeerCmd::kStop) break;
            std::exception_ptr err;
            try {
                if (cmd == PeerCmd::kRunVisits || cmd == PeerCmd::kRunGpuVisits) {
                    peer_->pos_ = pos_;
                    peer_->run_last_visit_ = run_last_visit_;
                    peer_->windowed_ = windowed_;
                    peer_->snap_bits_ = snap_bits_;
                    peer_->snap_at_ = windowed_ && snap_bits_ ? peer_->snap_ + (snap_at_ - snap_) : nullptr;
                    peer_->cover(pos_ + n);
                    if (!first_visit_) tp2_->broadcast_from_0(1, peer_->r_, n * kWidth, peer_->st_);
                    peer_->run_visits_local(n, cmd == PeerCmd::kRunGpuVisits);
                } else if (cmd == PeerCmd::kLogits) {
                    tp2_->broadcast_from_0(1, peer_->xn_, n * kWidth, peer_->st_);
                    peer_->lin(peer_->output_, peer_->xn_, peer_->tp2_logits_half_, n);
                    tp2_->gather_from_1(1, nullptr, peer_->tp2_logits_half_, n * peer_->tp2_vocab_n_, peer_->st_);
                } else if (cmd == PeerCmd::kDraftHead) {
                    tp2_->broadcast_from_0(1, peer_->df_hx_, n * kWidth, peer_->st_);
                    peer_->df_head_run(peer_->df_hx_, cs(peer_->st_));
                    dflash2::pack_top(peer_->df_hids_, peer_->df_hvals_, n, peer_->df_head_lo_, peer_->df_hpack_,
                                      cs(peer_->st_));
                    tp2_->gather_from_1(1, nullptr, peer_->df_hpack_, n * 2 * dflash2::kTop, peer_->st_);
                } else if (cmd == PeerCmd::kKeepSnap) {
                    ck(cudaMemcpyAsync(peer_->state_, peer_->snap_ + snap_idx * peer_->rec_n_,
                                       peer_->rec_n_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(peer_->st_)),
                       "tp2 peer keep");
                } else if (cmd == PeerCmd::kReset) {
                    peer_->reset();
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata-poly: TP2 peer exception: %s\n", e.what());
                tp2_->abort_barrier();
                err = std::current_exception();
            } catch (...) {
                tp2_->abort_barrier();
                err = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lk(peer_mu_);
                peer_err_ = err;
                peer_done_ = true;
            }
            peer_cv_.notify_one();
        }
    });
    std::unique_lock<std::mutex> lk(peer_mu_);
    peer_cv_.wait(lk, [&] { return started; });
    if (start_err) {
        lk.unlock();
        if (peer_thread_.joinable()) peer_thread_.join();
        std::rethrow_exception(start_err);
    }
}

void Inkling2::stop_peer_thread() {
    if (!peer_thread_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(peer_mu_);
        peer_cmd_ = PeerCmd::kStop;
    }
    peer_cv_.notify_one();
    peer_thread_.join();
}

void Inkling2::peer_dispatch(PeerCmd cmd, int n, size_t snap_idx) {
    {
        std::lock_guard<std::mutex> lk(peer_mu_);
        peer_done_ = false;
        peer_err_ = nullptr;
        peer_arg_n_ = n;
        peer_arg_snap_ = snap_idx;
        peer_cmd_ = cmd;
    }
    peer_cv_.notify_one();
}

void Inkling2::peer_wait() {
    std::exception_ptr err;
    {
        std::unique_lock<std::mutex> lk(peer_mu_);
        peer_cv_.wait(lk, [&] { return peer_done_; });
        err = peer_err_;
        peer_err_ = nullptr;
    }
    if (err) std::rethrow_exception(err);
}

void Inkling2::on_loaded() {
    if (df_gpu23_) df_gpu23_scratch();
#if defined(STRATA_HIP_GFX906)
    if (tp_size_ != 2) {
        const char* helpers = std::getenv("STRATA_INKLING_KV_HELPERS");
        if (helpers && !std::strcmp(helpers, "1"))
            std::fputs("strata-poly: STRATA_INKLING_KV_HELPERS=1 unavailable without TP2\n", stderr);
    }
#endif
    if (tp_size_ == 2 && all_cached()) {
        for (size_t il = 0; il < layers_.size(); ++il) {
            layers_[il].exp_gate = nullptr;
            layers_[il].exp_up = nullptr;
            layers_[il].exp_down = nullptr;
        }
        for (auto& v : tp2_exp_gu_) { v.clear(); v.shrink_to_fit(); }
        for (auto& v : tp2_exp_down_) { v.clear(); v.shrink_to_fit(); }
    }
}

#if defined(STRATA_HIP_GFX906)
bool Inkling2::gpu_route_supported() const {
    using namespace inkling_route;
    const bool valid_range = (lo_ == 0 || (prefill_only_ && lo_ == kStart)) && hi_ == n_layer_;
    if (!all_cached() || !opt_.gpu_experts || opt_.dump || opt_.cache_bias != 0.f || expert_scale_ != 8.f ||
        n_layer_ != kPhysical || !valid_range ||
        n_embd_ != kWidth || n_ff_exp_ != 576 || n_expert_ != kExperts || n_used_ != kPicks)
        return false;
    cudaDeviceProp props{};
    ck(cudaGetDeviceProperties(&props, device_), "Inkling GPU route device properties");
    if (!kernels::gfx_arch_is(props.gcnArchName, "gfx906")) return false;
    for (int il = lo_; il < hi_; ++il) {
        const Layer& L = layers_[il];
        const auto& bias = blocks_[il].bias;
        if (!L.moe || L.gate_inp.type != kTypeF32 || bias.size() != kExperts ||
            std::any_of(bias.begin(), bias.end(), [](float x) { return !std::isfinite(x); }))
            return false;
    }
    return true;
}

bool Inkling2::gpu_decode_supported() const {
    using namespace inkling_route;
    if (prefill_only_ || !decode_ || !situ_q8_ || moe_tail_ || !(situ_.beta > 0.f) || prefetch_k_ > 0) return false;
    const auto q4 = [](uint32_t type) { return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1; };
    for (int il = lo_; il < hi_; ++il) {
        const Layer& L = layers_[il];
        if (L.mixed() || L.gu_bytes != L.u_bytes || !q4(L.gu_type) || !q4(L.d_type) ||
            L.gu_bytes != (size_t) n_ff_exp_ * ggml_row_size((ggml_type) L.gu_type, n_embd_) ||
            L.d_bytes != (size_t) n_embd_ * ggml_row_size((ggml_type) L.d_type, n_ff_exp_) ||
            !L.h_s_gate.empty() || !L.h_s_up.empty() || !L.h_s_down.empty() ||
            L.slot.size() != kExperts || std::any_of(L.slot.begin(), L.slot.end(), [](auto p) { return !p; }))
            return false;
    }
    return true;
}

void Inkling2::init_gpu_route(bool decode, bool prefill) {
    using namespace inkling_route;
    std::vector<float> bias((size_t) kPhysical * kExperts, 0.f);
    for (int il = lo_; il < hi_; ++il)
        std::copy(blocks_[il].bias.begin(), blocks_[il].bias.end(), bias.begin() + (size_t) il * kExperts);
    gpu_route_bias_ = (float*) dmalloc(bias.size() * sizeof(float), "Inkling GPU route bias");
    ck(cudaMemcpy(gpu_route_bias_, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice),
       "Inkling GPU route bias");
    gpu_prefill_route_ = prefill;
    if (!decode) return;
    static_assert(kPicks == 8 && 4 * kPicks <= kSumRows);
    const int rr = std::min(rows_, gpu_route_rows_) * kPicks;
    if (!d_slots_)
        d_slots_ = (void**) dmalloc((size_t) n_layer_ * kExperts * sizeof(void*), "Inkling GPU route slots");
    if (!d_wts_) d_wts_ = (float*) dmalloc((size_t) rr * sizeof(float), "Inkling GPU route weights");
    const size_t bytes = (1 + size_t(kVisits) * rr) * sizeof(int32_t);
    gpu_route_dev_ = (int32_t*) dmalloc(bytes, "Inkling GPU route picks");
    gpu_route_host_ = (int32_t*) hmalloc(bytes, "Inkling GPU route report");
    gpu_route_gu_ = (float*) dmalloc((size_t) rr * 2 * n_ff_exp_ * sizeof(float), "Inkling GPU route gate/up");
    gpu_route_q8_ = dmalloc((size_t) rr * kernels::native_q8_1_bytes(n_ff_exp_), "Inkling GPU route activation Q8_1");
    if (const char* e = std::getenv("STRATA_INKLING_MOE_GROUPED"); e && !std::strcmp(e, "1")) {
        grp_part_ = (float*) dmalloc((size_t) rr * n_embd_ * sizeof(float), "Inkling grouped down partials");
        grp_ids_ = (int32_t*) dmalloc((1 + 64 * 10) * sizeof(int32_t), "Inkling grouped expert table");
    }
    if (!card_moe(1)) throw std::logic_error("Inkling GPU route could not initialize resident slots");
}

bool Inkling2::route_inkling_exact(int il, int n, int32_t* ids, float* w, int32_t* invalid) {
    if (!gpu_prefill_route_ || !all_cached() || il < 0 || il >= kPhysical || n < kStageRows || n > rows_ ||
        block_pass_ || !gpu_route_host_fp())
        return false;
    inkling_route::select(rl_, gpu_route_bias_ + il * inkling_route::kExperts, expert_scale_, ids, w, n, invalid, st_);
    return true;
}

void Inkling2::gpu_route_moe(Layer& L, int il, int n) {
    using namespace inkling_route;
    const int rr = n * kPicks;
    int32_t* ids = gpu_route_dev_ + 1 + visit_ * rr;
    const void* const* table = d_slots_ + (size_t) il * kExperts;
    select(rl_, gpu_route_bias_ + il * kExperts, expert_scale_, ids, d_wts_, n, gpu_route_dev_, st_);
    const size_t ff = (size_t) n_ff_exp_;
    const kernels::MmvqIds gu{table, ids, 0, xq_, kernels::native_q8_1_bytes(n_embd_), kPicks, gpu_route_gu_, 2 * ff};
    static const bool gu_r4 = [] {
        const char* e = std::getenv("STRATA_INKLING_GU_R4");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    if (grp_part_ && gpu_down_sum_ && n_ff_exp_ == 576 && n_embd_ == 2304) {
        // STRATA_INKLING_MOE_GROUPED: the same words in a schedule that reads each distinct expert of a window once
        if (!(n == 1 ? (gu_r4 && kernels::inkling_gu_q4_0_r4((int) L.gu_type, gu, rr, n_embd_, 2 * n_ff_exp_, st_)) ||
                           kernels::native_mmvq_ids((int) L.gu_type, gu, rr, n_embd_, 2 * n_ff_exp_, st_)
                     : kernels::inkling_gu_window((int) L.gu_type, gu, n, grp_ids_, st_, moe_group_ballot_ && kExperts <= 128)))
            throw std::logic_error("Inkling grouped gate/up rejected a validated shape");
        situ_rows_q8_1(gpu_route_gu_, gpu_route_q8_, n_ff_exp_, rr, situ_.beta, situ_.linear_beta, st_, expert_scales(L, ids));
        const kernels::MmvqIds dn{table, ids, L.gu2(), gpu_route_q8_, kernels::native_q8_1_bytes(n_ff_exp_), 1, d_,
                                  (size_t) n_embd_};
        if (!kernels::inkling_down_window((int) L.d_type, dn, n, d_wts_, fo_, grp_part_, grp_ids_, st_))
            throw std::logic_error("Inkling grouped down rejected a validated shape");
        return;
    }
    if (!(n == 1 && gu_r4 && kernels::inkling_gu_q4_0_r4((int) L.gu_type, gu, rr, n_embd_, 2 * n_ff_exp_, st_)) &&
        !kernels::native_mmvq_ids((int) L.gu_type, gu, rr, n_embd_, 2 * n_ff_exp_, st_))
        throw std::logic_error("Inkling GPU route gate/up MMVQ rejected a validated shape");
    situ_rows_q8_1(gpu_route_gu_, gpu_route_q8_, n_ff_exp_, rr, situ_.beta, situ_.linear_beta, st_, expert_scales(L, ids));
    const kernels::MmvqIds down{table, ids, L.gu2(), gpu_route_q8_, kernels::native_q8_1_bytes(n_ff_exp_),
                               1, d_, (size_t) n_embd_};
    if (gpu_down_sum_ && n_ff_exp_ == 576 && n_embd_ == 2304 &&
        kernels::inkling_down_sum((int)L.d_type, down, d_wts_, fo_, n, st_)) return;
    if (rr > kSumRows) throw std::logic_error("wide Inkling GPU route requires the fused down sum");
    if (!kernels::native_mmvq_ids((int) L.d_type, down, rr, n_ff_exp_, n_embd_, st_))
        throw std::logic_error("Inkling GPU route down MMVQ rejected a validated shape");
    gpu_route_sum<<<dim3(div_up(n_embd_, 256), n), 256, 0, cs(st_)>>>(d_, d_wts_, fo_, n_embd_);
    ck(cudaGetLastError(), "Inkling GPU route sum");
}

void Inkling2::count_gpu_route(int n) {
    // Count each visit (including repeat visits to a physical block), just as the stock route does. Keep the
    // compact picks on the GPU throughout the call; the single validation copy supplies the usage counts too.
    const int rr = n * inkling_route::kPicks;
    for (int visit = 0; visit < kVisits; ++visit) {
        const int il = physical(visit);
        for (int p = 0; p < rr; ++p) {
            const size_t i = (size_t) il * n_expert_ + gpu_route_host_[1 + visit * rr + p];
            ++use_[i];
            if (n_swap_ > 0) recent_[i] += use_weight(il);
        }
    }
    stats_.gpu_experts += kVisits * rr;
}
#endif

#if defined(STRATA_HIP_GFX906)
void Inkling2::request_budget(int64_t prompt_rows, int64_t max_new) {
    // DFlash2: the first block (at the prompt's end) attends its last 2047 positions, later ones only newer rows
    df_skip_ = df2_ ? std::max<int64_t>(0, prompt_rows - dflash2::kWindow - 8) : 0;
    if (!kv_helper_ || !peer_) return;
    // Both ranks are idle between requests. Ordinary TP2 and the speculation controller need no hint.
    inkling::kv_helpers::set_budget(*kv_helper_, prompt_rows, max_new);
    inkling::kv_helpers::set_budget(*peer_->kv_helper_, prompt_rows, max_new);
}
#endif

void Inkling2::on_reset() {
    if (df_gpu23_) df_gpu23_->prov_valid = false;
    df_lo_ = df_hi_ = df_skip_ = 0;
#if defined(STRATA_HIP_GFX906)
    if (kv_helper_) inkling::kv_helpers::invalidate(*kv_helper_, 0);
    stage_b_kv_valid_ = 0;
    if (stage_b_) {
        int prev = 0; (void) cudaGetDevice(&prev);
        ck(cudaSetDevice(stage_b_->device_), "PhaseDualTP reset dev 2");
        stage_b_->reset();
        (void) cudaSetDevice(prev);
    }
#endif
    if (!peer_) return;
    peer_dispatch(PeerCmd::kReset);
    peer_wait();
}

__global__ void interleave_logits_kernel(const float* __restrict__ h0, const float* __restrict__ h1,
                                         float* __restrict__ out, int n0, int n1, int rows) {
    const int n_vocab = n0 + n1;
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) rows * n_vocab;
    if (idx >= total) return;
    const int r = (int) (idx / n_vocab);
    const int c = (int) (idx % n_vocab);
    out[idx] = c < n0 ? h0[(size_t) r * n0 + c] : h1[(size_t) r * n1 + (c - n0)];
}

void Inkling2::on_keep_snap(size_t snap_idx) {
#if defined(STRATA_HIP_GFX906)
    stage_b_kv_valid_ = std::min(stage_b_kv_valid_, pos_);
#endif
    if (!peer_) return;
    peer_dispatch(PeerCmd::kKeepSnap, 0, snap_idx);
    peer_wait();
}

Dev Inkling2::load_output_matrix(const TensorInfo* t, int64_t file_vocab) {
    if (prefill_only_) return Dev{};
    if (tp_size_ != 2) return Engine::load_output_matrix(t, file_vocab);
    const int half0 = n_vocab_ / 2;
    tp2_vocab_lo_ = tp_rank_ == 0 ? 0 : half0;
    tp2_vocab_n_ = tp_rank_ == 0 ? half0 : (n_vocab_ - half0);
    return upload_row_range(t, "output.weight", n_embd_, file_vocab, tp2_vocab_lo_, tp2_vocab_n_, kAnyMat);
}

void Inkling2::project_logits(const float* h, float* out, int rows) {
    if (!peer_ || rows > std::max(kDsBlock, opt_.window)) {
        Engine::project_logits(h, out, rows);
        return;
    }
    peer_dispatch(PeerCmd::kLogits, rows);
    try {
        tp2_->broadcast_from_0(0, const_cast<float*>(h), rows * kWidth, st_);
        if (rows == 1) {
            lin(output_, h, out, 1);
            tp2_->gather_from_1(0, out + tp2_vocab_n_, nullptr, n_vocab_ - tp2_vocab_n_, st_);
        } else {
            lin(output_, h, tp2_logits_half_, rows);
            tp2_->gather_from_1(0, tp2_logits_peer_, nullptr, rows * (n_vocab_ - tp2_vocab_n_), st_);
            const size_t total = (size_t) rows * n_vocab_;
            interleave_logits_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, cs(st_)>>>(
                tp2_logits_half_, tp2_logits_peer_, out, tp2_vocab_n_, n_vocab_ - tp2_vocab_n_, rows);
        }
    } catch (...) {
        tp2_->abort_barrier();
        peer_wait();
        throw;
    }
    peer_wait();
}

// The drafter's shape must be the one graph_inkling2 runs for this model (llama.cpp's dspark.cpp): three layers of
// this width and heads, the 8-token block, taps at visits 5, 23, 41 and the output, a recurrent head of rank 256.
void Inkling2::load_drafter_shape(const Shards& g) {
    if (g.find("selector_hidden.weight")) return df_shape(g);
    const std::string file = opt_.draft_model;
    const auto need = [&](bool ok, const std::string& what) {
        if (!ok) throw std::runtime_error(file + " is not the DSpark drafter of Inkling-2-Mini: " + what);
    };
    const MetaValue* arch = g.get("general.architecture");
    need(arch && arch->s == "dflash", "its architecture is not 'dflash'");
    const bool precomp_head = g.find("markov_w2.weight") && g.find("markov_w2.bias") && g.find("markov_ctx_mix.weight");
    const bool legacy_head = g.find("dspark_hidden_out.weight") && g.find("dspark_fuse.weight");
    need((precomp_head || legacy_head) && g.find("markov_w1.weight") && g.find("joint_proj.weight"),
         "no Inkling2 DSpark head (markov_w1, joint_proj, and markov_w2 or dspark_hidden_out)");
    const auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = meta_in(g, key);
        need(v != nullptr, std::string("no dflash.") + key);
        return *v;
    };
    const auto is = [&](const char* key, uint64_t value) { need(must(key).u == value, std::string("dflash.") + key); };
    is("block_count", kDsLayers);
    is("embedding_length", kWidth);
    is("feed_forward_length", kDsFfn);
    is("attention.head_count", kHeads);
    is("attention.head_count_kv", kKv);
    is("attention.key_length", kHead);
    is("attention.value_length", kHead);
    is("rope.dimension_count", kHead);
    is("block_size", kDsBlock);
    const uint64_t sw = must("attention.sliding_window").u;
    need(sw == (uint64_t) kWindow || sw == (uint64_t) (kWindow + 1), "dflash.attention.sliding_window");
    if (const MetaValue* vu = meta_in(g, "vocab_size_unpadded")) need(vu->u == (uint64_t) n_vocab_, "dflash.vocab_size_unpadded");
    need(must("sample_from_anchor").u == 0, "it samples from the anchor");
    if (const MetaValue* ca = meta_in(g, "attention.causal")) need(ca->u == 0, "it is causal");
    const MetaValue& taps = must("target_layers");
    need(taps.items.size() == 4 && taps.items[0].u == (uint64_t) kTap[0] && taps.items[1].u == (uint64_t) kTap[1] &&
             taps.items[2].u == (uint64_t) kTap[2] && taps.items[3].u == (uint64_t) kVisits,
         "its target_layers are not [5, 23, 41, 48]");
    const MetaValue& pattern = must("attention.sliding_window_pattern");
    need(pattern.items.size() == kDsLayers, "dflash.attention.sliding_window_pattern");
    for (int l = 0; l < kDsLayers; ++l) need((pattern.items[(size_t) l].u != 0) == ds_swa(l), "its window pattern");
    ds_eps_ = (float) must("attention.layer_norm_rms_epsilon").f;
    ds_cap_ = (float) must("final_logit_softcapping").f;
    ds_theta_ = std::pow((float) must("rope.freq_base").f, -2.f / kHead);
    need(std::isfinite(ds_eps_) && ds_eps_ > 0.f && std::isfinite(ds_cap_) && ds_cap_ > 0.f && ds_theta_ > 0.f,
         "its epsilon, logit cap or rope base");
    const TensorInfo* w1 = g.find("markov_w1.weight");
    need((w1->type == GGML_TYPE_F16 || w1->type == GGML_TYPE_BF16) && w1->shape.size() == 2 &&
             w1->shape[0] == (uint64_t) kDsRank && w1->shape[1] >= (uint64_t) n_vocab_,
         "markov_w1 is not F16/BF16 [256, vocabulary]");
    ds_w1_bf16_ = (w1->type == GGML_TYPE_BF16);
    const TensorInfo* mask = g.find("mask_embd.weight");
    need(mask && mask->shape.size() >= 1 && mask->shape[0] == (uint64_t) kWidth, "mask_embd.weight");
    ds_mask_rows_ = (mask->shape.size() >= 2 && mask->shape[1] == (uint64_t) (kDsBlock - 1)) ? (kDsBlock - 1) : 1;
    if (precomp_head) {
        const TensorInfo* w2 = g.find("markov_w2.weight");
        const TensorInfo* b2 = g.find("markov_w2.bias");
        need((w2->type == GGML_TYPE_F16 || w2->type == GGML_TYPE_BF16) && w2->shape.size() == 2 &&
                 w2->shape[0] == (uint64_t) kDsRank && w2->shape[1] >= (uint64_t) n_vocab_,
             "markov_w2 is not F16/BF16 [256, vocabulary]");
        need(b2->type == kTypeF32 && b2->elements() >= (uint64_t) n_vocab_, "markov_w2.bias is not F32 [vocabulary]");
        ds_w2_bf16_ = (w2->type == GGML_TYPE_BF16);
    } else {
        const TensorInfo* fuse = g.find("dspark_fuse.weight");
        need(fuse->type == kTypeF32 && fuse->elements() == 4 * (kDsLayers + 1), "dspark_fuse is not F32 [4, 4]");
        const TensorInfo* hidden = g.find("dspark_hidden_out.weight");
        const TensorInfo* hidden_b = g.find("dspark_hidden_out.bias");
        need(hidden->type == GGML_TYPE_F16 && hidden->elements() == (uint64_t) kDsRank * kWidth && hidden_b &&
                 hidden_b->type == kTypeF32 && hidden_b->elements() == (uint64_t) kWidth,
             "dspark_hidden_out is not F16 [256, 2304] with an F32 bias");
    }
}

void Inkling2::begin_pass() {
    const auto part = [&](const char* name, int rows) { return upload(gg_->find(name), name, kWidth, rows, kF32); };
    token_norm_ = part("token_embd_norm.weight", 1);
    output_query_ = part("output_res_query.weight", 1);
    if (gg_->find("loop_res_query.weight")) {
        loop_query_ = part("loop_res_query.weight", 2 * kCore);
        loop_kv_query_ = part("loop_kv_res_query.weight", 2 * kCore);
    }
}

void Inkling2::load_moe_tp2(int il) {
    Layer& L = layers_[(size_t) il];
    L.moe = true;
    L.gate_inp = upload(tensor(il, "ffn_gate_inp.weight"), blk(il, "ffn_gate_inp.weight"), n_embd_, n_expert_, kAnyMat);
    const int full_ff = n_ff_exp_ * 2;
    const TensorInfo* td = tensor(il, "ffn_down_exps.weight");
    const TensorInfo* gu = tensor(il, "ffn_gate_up_exps.weight");
    if (!td || !gu) throw std::runtime_error(blk(il, "TP2 requires ffn_gate_up_exps and ffn_down_exps"));
    const size_t full_d_bytes = home(*td).payload(*td) / (size_t) n_expert_;
    const size_t full_gu_stride = home(*gu).payload(*gu) / (size_t) n_expert_;
    const size_t half_gu_exp = full_gu_stride / 2;   // all gate rows (or all up rows) of one expert
    const size_t quarter_gu = half_gu_exp / 2;       // half of gate (or half of up) for this rank

    L.d_type = td->type;
    L.d_bytes = full_d_bytes / 2;
    L.gu_type = gu->type;
    L.gu_bytes = quarter_gu;
    L.gu_stride = 2 * quarter_gu;
    L.u_type = L.gu_type;
    L.u_bytes = L.gu_bytes;
    L.u_stride = L.gu_stride;
    L.gu_step = L.gu2();

    if (dense_) {
        const uint8_t* gu_src = home(*gu).data(*gu);
        auto& gu_dst = tp2_exp_gu_[(size_t) il];
        gu_dst.resize((size_t) n_expert_ * L.gu_stride);

        const uint8_t* d_src = home(*td).data(*td);
        auto& d_dst = tp2_exp_down_[(size_t) il];
        d_dst.resize((size_t) n_expert_ * L.d_bytes);
        const size_t full_row = full_d_bytes / (size_t) n_embd_;
        const size_t half_row = full_row / 2;
        if (full_row % 2 != 0) throw std::runtime_error(blk(il, "TP2 ffn_down_exps odd row bytes"));

        const int n_thr = std::min(8, std::max(1, n_expert_));
        std::vector<std::thread> pool;
        pool.reserve((size_t) n_thr);
        for (int t = 0; t < n_thr; ++t) {
            pool.emplace_back([&, t] {
                const int e0 = t * n_expert_ / n_thr;
                const int e1 = (t + 1) * n_expert_ / n_thr;
                for (int e = e0; e < e1; ++e) {
                    const uint8_t* es = gu_src + (size_t) e * full_gu_stride;
                    uint8_t* ed = gu_dst.data() + (size_t) e * L.gu_stride;
                    std::memcpy(ed, es + (size_t) tp_rank_ * quarter_gu, quarter_gu);
                    std::memcpy(ed + quarter_gu, es + half_gu_exp + (size_t) tp_rank_ * quarter_gu, quarter_gu);

                    const uint8_t* ds = d_src + (size_t) e * full_d_bytes;
                    uint8_t* dd = d_dst.data() + (size_t) e * L.d_bytes;
                    for (int r = 0; r < n_embd_; ++r)
                        std::memcpy(dd + (size_t) r * half_row,
                                    ds + (size_t) r * full_row + (size_t) tp_rank_ * half_row, half_row);
                }
            });
        }
        for (auto& th : pool) th.join();
        L.exp_gate = gu_dst.data();
        L.exp_up = gu_dst.data() + L.gu_bytes;
        L.exp_down = d_dst.data();
    }

    const auto one = [&](const char* n, uint32_t type, Dev& d, std::vector<float>& h) {
        const TensorInfo* t = tensor(il, n);
        h.clear();
        if (type != GGML_TYPE_NVFP4 || !t) return;
        if (t->type != kTypeF32) throw std::runtime_error(t->name + " is not F32");
        d = upload(t, blk(il, n), n_expert_, 1, kF32);
        h.resize((size_t) n_expert_);
        std::memcpy(h.data(), home(*t).data(*t), h.size() * sizeof(float));
    };
    if (L.gu_type == GGML_TYPE_NVFP4 && tensor(il, "ffn_gate_up_exps.scale")) {
        one("ffn_gate_up_exps.scale", L.gu_type, L.s_gate, L.h_s_gate);
        L.s_up = L.s_gate;
        L.h_s_up = L.h_s_gate;
    } else {
        one("ffn_gate_exps.scale", L.gu_type, L.s_gate, L.h_s_gate);
        one("ffn_up_exps.scale", L.u_type, L.s_up, L.h_s_up);
    }
    one("ffn_down_exps.scale", L.d_type, L.s_down, L.h_s_down);
    (void) full_ff;
    L.slot.assign((size_t) n_expert_, nullptr);
    ++n_moe_;
}

void Inkling2::load_layer(int il) {
    Block& b = blocks_[il];
    Layer& L = layers_[il];
    const auto part = [&](const char* name, int cols, int rows = 1, Want want = kF32) {
        return upload(tensor(il, name), blk(il, name), cols, rows, want);
    };
    const auto row_sh = [&](const char* name, int cols, int rows, Want want = kF32) {
        return tp_size_ == 2 ? upload_slice(tensor(il, name), blk(il, name), cols, rows, want, tp_rank_, false)
                             : part(name, cols, rows, want);
    };
    const auto col_sh = [&](const char* name, int cols, int rows, Want want = kF32) {
        return tp_size_ == 2 ? upload_slice(tensor(il, name), blk(il, name), cols, rows, want, tp_rank_, true)
                             : part(name, cols, rows, want);
    };
    b.q = row_sh("attn_q.weight", kWidth, kQuery, kAnyMat);
    b.k = row_sh("attn_k.weight", kWidth, kKey, kAnyMat);
    b.v = row_sh("attn_v.weight", kWidth, kKey, kAnyMat);
    b.o = col_sh("attn_output.weight", kQuery, kWidth, kAnyMat);
    b.gate = row_sh("attn_gate.weight", kWidth, kQuery, kAnyMat);
    b.rel = row_sh("attn_rel.weight", kWidth, kRelative * kHeads, kAnyMat);
    b.sink = row_sh("attn_sink_proj.weight", kWidth, kHeads);
    b.qclip = row_sh("attn_q_clip.weight", kHead, kHeads);
    b.rclip = row_sh("attn_rel_clip.weight", kRelative, kHeads);
    b.table = part("attn_rel_bias.weight", kRelative, kWindow);
    b.kconv = row_sh("attn_k_conv.weight", kTaps, kKey);
    b.vconv = row_sh("attn_v_conv.weight", kTaps, kKey);
    b.aconv = part("attn_post_conv.weight", kTaps, kWidth);
    b.fconv = part("ffn_post_conv.weight", kTaps, kWidth);
    if (local(il)) L.attn_norm = part("attn_norm.weight", kWidth);
    else {
        L.attn_norm = part("attn_q_inp_norm.weight", kWidth);
        b.knorm = part("attn_k_inp_norm.weight", kWidth);
        b.vnorm = part("attn_v_inp_norm.weight", kWidth);
        b.kproj = row_sh("attn_k_conv_proj.weight", kWidth, kTaps * kKv);
        b.vproj = row_sh("attn_v_conv_proj.weight", kWidth, kTaps * kKv);
    }
    if (il > 0) b.aquery = part("attn_res_query.weight", kWidth, local(il) ? 1 : 3);
    b.fquery = part("ffn_res_query.weight", kWidth);
    L.ffn_norm = part("ffn_norm.weight", kWidth);
    if (!tensor(il, "ffn_gate_up_exps.weight"))
        throw std::runtime_error(blk(il, "ffn_gate_up_exps.weight is required for Inkling-2-Mini"));
    if (tp_size_ == 2) load_moe_tp2(il);
    else load_moe(il, false);
    if ((L.gu_type == GGML_TYPE_NVFP4 && L.h_s_gate.empty()) ||
        (L.d_type == GGML_TYPE_NVFP4 && L.h_s_down.empty()))
        throw std::runtime_error(blk(il, "NVFP4 experts require their scales"));
    const auto* bias = tensor(il, "exp_probs_b.bias");
    if (!bias || bias->type != kTypeF32 || bias->elements() != (uint64_t) n_expert_)
        throw std::runtime_error(blk(il, "exp_probs_b.bias must be an F32 expert vector"));
    b.bias.resize(n_expert_);
    home(*bias).payload(*bias);
    std::memcpy(b.bias.data(), home(*bias).data(*bias), b.bias.size() * sizeof(float));

    if (il >= kStart && il < kStart + kCore) {
        const bool ctrl_names = tensor(il, "loop_q_ctrl_down.weight") != nullptr;
        if (tensor(il, "loop_msg_query.weight")) b.msg_query = part("loop_msg_query.weight", kWidth, 4);
        else if (!gg_->find("loop_res_query.weight") || !gg_->find("loop_kv_res_query.weight"))
            throw std::runtime_error(blk(il, "loop_msg_query.weight (or global loop_res_query.weight) is required"));
        const char* stream_old[4] = {"attn", "k", "v", "mlp"};
        const char* stream_new[4] = {"q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"};
        for (int i = 0; i < 4; ++i) {
            const std::string pre = std::string("loop_") + (ctrl_names ? stream_new[i] : stream_old[i]);
            b.control[i] = {part((pre + "_down.weight").c_str(), 2 * kWidth, kRank),
                            part((pre + "_gain.weight").c_str(), kRank, kWidth),
                            part((pre + "_forget.weight").c_str(), kRank, kWidth)};
        }
    }
}

// The drafter's weights (both passes of load(): the first adds up the arena).  The recurrent head's token table and
// hidden_out stay on the host: a step reads one row of the table, and hidden_out goes into W' (prepare_head()).
void Inkling2::load_drafter() {
    if (df2_) return df_load();
    const Shards& g = *dg_;
    const auto up = [&](const std::string& name, int64_t n_in, int64_t n_out, Want want) {
        const TensorInfo* t = g.find(name);
        if (!t) throw std::runtime_error(opt_.draft_model + " has no " + name);
        return upload(t, name, n_in, n_out, want);
    };
    for (int l = 0; l < kDsLayers; ++l) {
        DsLayer& L = ds_[(size_t) l];
        const auto name = [&](const char* n) { return "blk." + std::to_string(l) + "." + n; };
        L.attn_norm = up(name("attn_norm.weight"), kWidth, 1, kF32);
        L.ffn_norm = up(name("ffn_norm.weight"), kWidth, 1, kF32);
        L.ctx_norm = up(g.find(name("ctx_norm.weight")) ? name("ctx_norm.weight") : name("dflare_norm.weight"), kWidth, 1, kF32);
        L.q = up(name("attn_q.weight"), kWidth, kQuery, kAnyMat);
        L.k = up(name("attn_k.weight"), kWidth, kKey, kAnyMat);
        L.v = up(name("attn_v.weight"), kWidth, kKey, kAnyMat);
        L.o = up(name("attn_output.weight"), kQuery, kWidth, kAnyMat);
        L.k_ctx = up(name("attn_k_ctx.weight"), kWidth, kKey, kAnyMat);
        L.v_ctx = up(name("attn_v_ctx.weight"), kWidth, kKey, kAnyMat);
        L.q_norm = up(name("attn_q_norm.weight"), kHead, 1, kF32);
        L.k_norm = up(name("attn_k_norm.weight"), kHead, 1, kF32);
        L.k_ctx_norm = up(name("attn_k_ctx_norm.weight"), kHead, 1, kF32);
        L.gate = up(name("ffn_gate.weight"), kWidth, kDsFfn, kAnyMat);
        L.up = up(name("ffn_up.weight"), kWidth, kDsFfn, kAnyMat);
        L.down = up(name("ffn_down.weight"), kDsFfn, kWidth, kAnyMat);
    }
    ds_out_norm_ = up("output_norm.weight", kWidth, 1, kF32);
    ds_mask_ = up("mask_embd.weight", kWidth, ds_mask_rows_, kF32);
    ds_rnn_norm_ = up(g.find("markov_ctx_norm.weight") ? "markov_ctx_norm.weight" : "dspark_norm.weight", kWidth, 1, kF32);
    ds_joint_ = up("joint_proj.weight", 2 * kDsRank + 2 * kWidth, 3 * kDsRank, kF32);
    ds_joint_b_ = up("joint_proj.bias", 3 * kDsRank, 1, kF32);

    const auto host = [&](const std::string& name) {
        const TensorInfo* t = g.find(name);
        g.payload(*t);
        return g.data(*t);
    };
    const bool precomp_head = g.find("markov_w2.weight") && g.find("markov_w2.bias") && g.find("markov_ctx_mix.weight");
    if (!precomp_head) {
        const float* fuse = (const float*) host("dspark_fuse.weight");
        for (int r = 0; r <= kDsLayers; ++r) {   // a row's weights over the four taps: softmax(dspark_fuse[r])
            float most = -INFINITY, sum = 0.f;
            for (int j = 0; j < 4; ++j) most = std::max(most, fuse[r * 4 + j]);
            for (int j = 0; j < 4; ++j) sum += ds_mix_[r][j] = std::exp(fuse[r * 4 + j] - most);
            for (int j = 0; j < 4; ++j) ds_mix_[r][j] /= sum;
        }
    } else {
        for (int l = 0; l < kDsLayers; ++l) {
            const float* m = (const float*) host("blk." + std::to_string(l) + ".ctx_mix.weight");
            std::memcpy(ds_mix_[l], m, 4 * sizeof(float));
        }
        const float* m = (const float*) host("markov_ctx_mix.weight");
        std::memcpy(ds_mix_[kDsLayers], m, 4 * sizeof(float));
    }
    ds_w1_ = host("markov_w1.weight");
    if (precomp_head) {
        ds_w2_host_ = host("markov_w2.weight");
        ds_b2_host_ = (const float*) host("markov_w2.bias");
    } else {
        const uint16_t* hidden = (const uint16_t*) host("dspark_hidden_out.weight");
        ds_hidden_.resize((size_t) kWidth * kDsRank);
        for (size_t i = 0; i < ds_hidden_.size(); ++i) ds_hidden_[i] = ggml_fp16_to_fp32(hidden[i]);
        const float* bias = (const float*) host("dspark_hidden_out.bias");
        ds_hidden_b_.assign(bias, bias + kWidth);
    }
}

size_t Inkling2::recurrent_floats() const { return (size_t) (last_visit_ - first_visit_) * conv_floats_; }
size_t Inkling2::kv_bytes() const {
    return 2 * (q8() ? n_key_ + (n_key_ / 32) * sizeof(uint16_t) : n_key_ * sizeof(uint16_t));
}
size_t Inkling2::card_row_bytes() const {
    return owned_full() * kv_bytes() * (peer_ ? 2 : 1) + (ds_on() ? ds_row_bytes() : 0);
}

void Inkling2::alloc_caches() {
    const int local_visits = owned_swa();
    ring_ = std::min(opt_.max_context, kWindow + rows_ - 1);
#if defined(STRATA_HIP_GFX906)
    if (compact_swa_asked_) {
        const int swa_win = std::min(std::max(opt_.window, swa_ring_win_), rows_);
        const auto plan = inkling::plan_swa(opt_.max_context, kWindow, rows_, swa_win,
                                            local_visits, kv_bytes(), (size_t) rows_ * kWidth * sizeof(float),
                                            tp_size_ == 2 && n_key_ == 512 && q8());
        if (plan.stage_rows)
            std::fprintf(stderr, "strata-poly: STRATA_INKLING_COMPACT_SWA=1 rank %d (dev %d): %d -> %d rows per SWA visit; "
                         "%zu-byte borrowed stage, %zu bytes saved\n", tp_rank_, device_, ring_, plan.ring_rows,
                         (size_t) plan.stage_rows * kv_bytes(), plan.saved_bytes);
        else
            std::fprintf(stderr, "strata-poly: STRATA_INKLING_COMPACT_SWA=1 rank %d (dev %d): no borrowable savings; "
                         "using stock rings\n", tp_rank_, device_);
        ring_ = plan.ring_rows;
        swa_stage_rows_ = plan.stage_rows;
    }
#endif
    auto* rings = (uint8_t*) seq_alloc((size_t) local_visits * ring_ * kv_bytes(), "Inkling SWA KV");
    const size_t wide = n_key_ * (q8() ? 1 : sizeof(uint16_t));
    const size_t scale = q8() ? n_key_ / 32 * sizeof(uint16_t) : 0;
    for (int i = first_visit_; i < last_visit_; ++i) {
        for (int j = 0; j < 4; ++j) {
            const size_t stride = j % 2 ? scale : wide;
            auto& p = caches_[i].planes[j];
            p = {nullptr, stride};
            if (!stride) continue;
            if (local(i)) {
                p.data = rings;
                rings += (size_t) ring_ * stride;
            } else p.data = (uint8_t*) kv_alloc(stride, 1);
        }
    }
    if (!draft_) return;
    if (df2_) {   // DFlash2: five half rings of 2048 positions, K and V
        if (!df_gpu23_) {
            const size_t ring = (size_t) kDfLayers * dflash2::kWindow * dflash2::kKey * sizeof(__half);
            df_kc_ = (__half*) seq_alloc(ring, "DFlash2 K rings");
            df_vc_ = (__half*) seq_alloc(ring, "DFlash2 V rings");
        }
        df_lo_ = df_hi_ = 0;
        return;
    }
    // the drafter's K/V, Q8_0 whatever --kv is: its window layers keep the last kWindow positions, the other every one
    for (int l = 0; l < kDsLayers; ++l)
        for (int j = 0; j < 4; ++j) {
            const size_t stride = j % 2 ? kKey / 32 * sizeof(uint16_t) : kKey;
            auto& p = ds_cache_[(size_t) l].planes[j];
            p = {nullptr, stride};
            p.data = ds_swa(l) ? (uint8_t*) seq_alloc((size_t) kWindow * stride, "DSpark window KV")
                               : (uint8_t*) kv_alloc(stride, 1);
        }
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
#if defined(STRATA_HIP_GFX906)
    if (gpu_route_failed_) throw std::runtime_error("cannot save rows from a failed Inkling GPU route");
#endif
    if (first < 0 || n < 0 || first + n > opt_.max_context) throw std::logic_error("save_rows: outside context");
    if (!n) return;
    ck(cudaStreamSynchronize(cs(st_)), "save Inkling rows");
    for (int i = 0; i < kVisits; ++i) if (!local(i)) {
        for (const auto& p : caches_[i].planes) if (p.stride) {
            const size_t bytes = (size_t) n * p.stride;
            ck(cudaMemcpy(dst, p.data + first * p.stride, bytes, cudaMemcpyDeviceToHost), "save Inkling rows");
            dst += bytes;
        }
    }
    if (ds_on())
        for (const auto& p : ds_cache_[1].planes) {
            const size_t bytes = (size_t) n * p.stride;
            ck(cudaMemcpy(dst, p.data + first * p.stride, bytes, cudaMemcpyDeviceToHost), "save DSpark rows");
            dst += bytes;
        }
    if (peer_) {
        int prev = 0; cudaGetDevice(&prev);
        ck(cudaSetDevice(peer_->device_), "save peer rows");
        peer_->save_card_rows(first, n, dst);
        cudaSetDevice(prev);
    }
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: outside saved rows");
    if (!n) return;
#if defined(STRATA_HIP_GFX906)
    if (kv_helper_) inkling::kv_helpers::invalidate(*kv_helper_, first);
    stage_b_kv_valid_ = std::min(stage_b_kv_valid_, first);
#endif
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load Inkling rows");
    for (int i = 0; i < kVisits; ++i) if (!local(i)) {
        for (const auto& p : caches_[i].planes) if (p.stride) {
            ck(cudaMemcpy(p.data + first * p.stride, src + (first - from) * p.stride, (size_t) n * p.stride,
                          cudaMemcpyHostToDevice), "load Inkling rows");
            src += (size_t) count * p.stride;
        }
    }
    if (ds_on())
        for (const auto& p : ds_cache_[1].planes) {
            ck(cudaMemcpy(p.data + first * p.stride, src + (first - from) * p.stride, (size_t) n * p.stride,
                          cudaMemcpyHostToDevice), "load DSpark rows");
            src += (size_t) count * p.stride;
        }
    if (peer_) {
        int prev = 0; cudaGetDevice(&prev);
        ck(cudaSetDevice(peer_->device_), "load peer rows");
        peer_->load_card_rows(src, from, count, first, n);
        cudaSetDevice(prev);
    }
}

size_t Inkling2::extra_bytes() const {
    return (kVisits - kVisits / kBlock) * (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes() +
           (ds_on() ? 2 * (size_t) std::min(opt_.max_context, kWindow) * ds_row_bytes() : 0) +
           (peer_ ? peer_->card_state_bytes() : 0);
}

void Inkling2::ring_state(int visit, int64_t pos, const uint8_t* src, uint8_t* dst) {
    const int64_t keep = std::min(opt_.max_context, kWindow - 1);
    const int64_t first = std::max<int64_t>(0, pos - keep);
    for (const auto& p : caches_[visit].planes) if (p.stride) {
        if (dst) std::memset(dst, 0, (size_t) (keep - (pos - first)) * p.stride);
        for (int64_t i = first; i < pos;) {
            // Saved images are chronological plane bytes, independent of the allocation's modulus.
            const auto [at, n] = inkling::swa_ring_span(i, pos, ring_);
            const size_t off = (size_t) (i - (pos - keep)) * p.stride, bytes = (size_t) n * p.stride;
            if (dst) ck(cudaMemcpy(dst + off, p.data + at * p.stride, bytes, cudaMemcpyDeviceToHost), "save Inkling SWA");
            else ck(cudaMemcpy(p.data + at * p.stride, src + off, bytes, cudaMemcpyHostToDevice), "load Inkling SWA");
            i += n;
        }
        if (dst) dst += keep * p.stride;
        else src += keep * p.stride;
    }
}

// A drafter window layer's last kWindow rows before `pos`, from the ring (dst) or into it (src): a plane after another,
// the oldest row first, zeros before the sequence's first row.
void Inkling2::ds_ring(int l, int64_t pos, const uint8_t* src, uint8_t* dst) {
    const int64_t keep = std::min(opt_.max_context, kWindow);
    const int64_t first = std::max<int64_t>(0, pos - keep);
    for (const auto& p : ds_cache_[(size_t) l].planes) {
        if (dst) std::memset(dst, 0, (size_t) (keep - (pos - first)) * p.stride);
        for (int64_t i = first; i < pos;) {
            const int64_t at = i % kWindow, n = std::min(pos - i, kWindow - at);
            const size_t off = (size_t) (i - (pos - keep)) * p.stride, bytes = (size_t) n * p.stride;
            if (dst) ck(cudaMemcpy(dst + off, p.data + at * p.stride, bytes, cudaMemcpyDeviceToHost), "save DSpark");
            else ck(cudaMemcpy(p.data + at * p.stride, src + off, bytes, cudaMemcpyHostToDevice), "load DSpark");
            i += n;
        }
        if (dst) dst += keep * p.stride;
        else src += keep * p.stride;
    }
}

void Inkling2::save_card_state(uint8_t* dst) {
#if defined(STRATA_HIP_GFX906)
    if (gpu_route_failed_) throw std::runtime_error("cannot save state from a failed Inkling GPU route");
#endif
    static const bool pack = [] {
        const char* p = std::getenv("STRATA_INKLING_STATE_PACK");
        return p && p[0] == '1' && p[1] == '\0';
    }();
    if (pack && looped_ && try_pack_state(dst)) {
        static const bool logged = [] {
            std::fputs("strata-poly: STRATA_INKLING_STATE_PACK=1: packed checkpoints in expert stage\n", stderr);
            return true;
        }();
        (void) logged;
        return;
    }
    Engine::save_card_state(dst);
    if (peer_) {
        const size_t r0_extra = (kVisits - kVisits / kBlock) * (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes() +
                                (ds_on() ? 2 * (size_t) std::min(opt_.max_context, kWindow) * ds_row_bytes() : 0);
        int prev = 0; (void) cudaGetDevice(&prev);
        ck(cudaSetDevice(peer_->device_), "save peer state");
        ck(cudaStreamSynchronize(cs(peer_->st_)), "save peer state");
        peer_->pos_ = pos_;
        peer_->save_card_state(dst + state_n_ * sizeof(float) + r0_extra);
        (void) cudaSetDevice(prev);
    }
}

bool Inkling2::try_pack_state(uint8_t* dst) {
    // save_state() joined st_. Use only the complete one-card, non-DSpark layout and the expert stage's
    // currently mapped capacity; never allocate or widen it here.
    if (!dst || draft_ || !direct_stage_ || !stage_ || split_ || next_ || recording_ ||
        lo_ != 0 || hi_ != n_layer_ || n_layer_ != kPhysical || layers_.size() != kPhysical ||
        state_n_ != rec_n_ || rec_n_ != (size_t) kVisits * kConvFloats || !state_ ||
        ring_ <= 0 || pos_ < 0 || pos_ > opt_.max_context)
        return false;
    const size_t state_bytes = state_n_ * sizeof(float), bytes = card_state_bytes();
    if (stage_bytes_ < bytes) return false;
    const int64_t keep = std::min(opt_.max_context, kWindow - 1);
    const int64_t first = std::max<int64_t>(0, pos_ - keep);
    const size_t one = (size_t) keep * kv_bytes();
    if (bytes != state_bytes + (kVisits - kVisits / kBlock) * one ||
        ring_ != std::min(opt_.max_context, kWindow + rows_ - 1))
        return false;
    for (int i = 0; i < kVisits; ++i) if (local(i))
        for (int j = 0; j < 4; ++j) {
            const auto& p = caches_[i].planes[j];
            const size_t stride = j % 2 ? (q8() ? kKey / 32 * sizeof(uint16_t) : 0)
                                       : kKey * (q8() ? 1 : sizeof(uint16_t));
            if (p.stride != stride || (stride && !p.data)) return false;
        }

    // Fault this checkpoint in parallel before the bulk D2H.
    if (workers_ && workers_->threads() > 1) {
        constexpr size_t kTouchJob = size_t(2) << 20;
        const int jobs = (int) (bytes / kTouchJob + (bytes % kTouchJob != 0));
        const auto touch_pages = [dst, bytes](int job) {
            constexpr size_t kJob = size_t(2) << 20, kStep = 4096;
            const size_t first = (size_t) job * kJob, end = std::min(bytes, first + kJob);
            volatile uint8_t* const out = dst;
            for (size_t at = first; at < end; at += kStep) out[at] = 0;
            if (first < end) out[end - 1] = 0;
        };
        workers_->run(jobs, std::ref(touch_pages));
    }

    // stage_ no longer holds weights, including if an enqueue below throws. clear() retains capacity.
    staged_.clear();
    staged_il_ = -1;
    ck(cudaMemsetAsync(stage_, 0, bytes, cs(st_)), "pack Inkling state padding");
    ck(cudaMemcpyAsync(stage_, state_, state_bytes, cudaMemcpyDeviceToDevice, cs(st_)),
       "pack Inkling convolution state");
    for (int i = 0; i < kVisits; ++i) if (local(i)) {
        uint8_t* plane = stage_ + state_bytes + (i - i / kBlock) * one;
        for (const auto& p : caches_[i].planes) if (p.stride) {
            // Same source row, destination padding and oldest-first order as ring_state().
            for (int64_t row = first; row < pos_;) {
                const int64_t at = row % ring_, count = std::min(pos_ - row, ring_ - at);
                const size_t off = (size_t) (row - (pos_ - keep)) * p.stride;
                ck(cudaMemcpyAsync(plane + off, p.data + at * p.stride, (size_t) count * p.stride,
                                   cudaMemcpyDeviceToDevice, cs(st_)), "pack Inkling SWA state");
                row += count;
            }
            plane += (size_t) keep * p.stride;
        }
    }
    ck(cudaMemcpyAsync(dst, stage_, bytes, cudaMemcpyDeviceToHost, cs(st_)), "save packed Inkling state");
    ck(cudaStreamSynchronize(cs(st_)), "save packed Inkling state");
    // No caller can start its next prefetch or narrow/unmap stage_ until all copies above have completed.
    return true;
}

void Inkling2::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (ds_on() && il == kPhysical - 1) {   // the drafter's rings, behind the model's
        uint8_t* at = dst + state_n_ * sizeof(float) +
                      (kVisits - kVisits / kBlock) * (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
        for (int l = 0; l < kDsLayers; ++l)
            if (ds_swa(l)) {
                ds_ring(l, pos, nullptr, at);
                at += (size_t) std::min(opt_.max_context, kWindow) * ds_row_bytes();
            }
    }
    for (int i = 0; i < kVisits; ++i) if (physical(i) == il) {
        const size_t at = (size_t) i * conv_floats_;
        ck(cudaMemcpy(dst + at * sizeof(float), state_ + at, conv_floats_ * sizeof(float), cudaMemcpyDeviceToHost),
           "save Inkling convolution");
        if (local(i)) {
            const size_t offset = (i - i / kBlock) * (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
            ring_state(i, pos, nullptr, dst + state_n_ * sizeof(float) + offset);
        }
    }
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
#if defined(STRATA_HIP_GFX906)
    // A restored state can refer to a different prefix at the same position. Rebuild only from canonical rows.
    if (kv_helper_) inkling::kv_helpers::invalidate(*kv_helper_, 0);
    stage_b_kv_valid_ = 0;
#endif
    const size_t one = (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
    for (int i = 0; i < kVisits; ++i) if (local(i)) {
        ring_state(i, pos, src, nullptr);
        src += one;
    }
    if (df_gpu23_) df_gpu23_->prov_valid = false;
    df_lo_ = df_hi_ = pos;   // DFlash2: no context for a restored state
    if (ds_on())
        for (int l = 0; l < kDsLayers; ++l)
            if (ds_swa(l)) {
                ds_ring(l, pos, src, nullptr);
                src += (size_t) std::min(opt_.max_context, kWindow) * ds_row_bytes();
            }
    if (peer_) {
        int prev = 0; (void) cudaGetDevice(&prev);
        ck(cudaSetDevice(peer_->device_), "load peer state");
        peer_->load_state(src, pos);
        (void) cudaSetDevice(prev);
    }
}

void Inkling2::assign_scratch() {
    const size_t n = rows_, width = n * kWidth;
    q_ = fbuf(n * n_query_);
    k_ = fbuf(n * n_key_);
    v_ = fbuf(n * n_key_);
    att_ = fbuf(n * n_query_);
    gate_ = fbuf(n * n_query_);
    rel_ = fbuf(n * n_heads_ * kRelative);
    sink_ = fbuf(n * n_heads_);
    taps_ = fbuf(n * n_kv_ * kTaps);
    halo_ = fbuf((size_t) div_up(rows_, kConvSpan) * 3 * kWidth);
    // A tile takes kValueTile scores and one kHead-wide partial value vector per query head. Keep room
    // for a single query at the configured limit, but share that room across queries on shorter prefixes.
    constexpr int target_tiles = (1 << 20) / (kHeads * kValueTile);
    attend_tiles_ = std::max(div_up(opt_.max_context, kValueTile),
                             std::min(target_tiles, rows_ * div_up(opt_.max_context, kValueTile)));
    scores_ = fbuf(inkling::plain_score_floats(attend_tiles_));
    partial_ = fbuf(inkling::plain_partial_floats(attend_tiles_));
    if (decode_) {
        chunks_ = fbuf(inkling::decode_floats());
        taps2_ = fbuf(n * n_kv_ * kTaps);
        attq_ = fbuf((size_t) kStageRows * (n_query_ / 32) * 36 / sizeof(float));   // Q8_1: 36 bytes a block of 32
        kq8_ = fbuf((size_t) (kWidth / 32) * 36 / sizeof(float));
        vq8_ = fbuf((size_t) (kWidth / 32) * 36 / sizeof(float));
        // heads(): the three streams of up to kStageRows - 1 decode rows
        const size_t w3 = 3 * (size_t) kStageRows * kWidth;
        s3cur_ = fbuf(w3);
        s3msg_ = fbuf(w3);
        s3pair_ = fbuf(2 * w3);
        s3rank_ = fbuf(3 * (size_t) kStageRows * kRank);
        s3in_ = fbuf(w3);
        s3q8_ = fbuf(3 * (size_t) kStageRows * kernels::native_q8_1_bytes(kWidth) / sizeof(float));
    }
    for (auto& x : input_) x = fbuf(width);
    for (auto& x : depth_) x = fbuf(width);
    prelude_ = fbuf(width);
    bank0_ = fbuf(width);
    bank1_ = fbuf(width);
    msg_ = fbuf(width);
    pair_ = fbuf(2 * width);
    rank_ = fbuf(n * kRank);
    gain_ = fbuf(width);
    forget_ = fbuf(width);
    if (tp_size_ == 2 && !prefill_only_) {
        const size_t max_r = (size_t) std::max(kDsBlock, opt_.window);
        tp2_logits_half_ = fbuf(max_r * (size_t) tp2_vocab_n_);
        if (tp_rank_ == 0) tp2_logits_peer_ = fbuf(max_r * (size_t) (n_vocab_ - tp2_vocab_n_));
    }
#if defined(STRATA_HIP_GFX906)
    if (prefill_fuse_ && rec_n_ > 0) state_next_ = fbuf(rec_n_);
    if (prefill_only_) {
        cut_depth0_ = fbuf(width);
        cut_prelude_ = fbuf(width);
        cut_bank0_ = fbuf(width);
        cut_bank1_ = fbuf(width);
        if (stage_b_want_df2_) {
            sb_tap0_ = fbuf(width);
            sb_tap1_ = fbuf(width);
            sb_tap2_ = fbuf(width);
            sb_hn_ = fbuf(width);
        }
    }
#endif
    if (!draft_) return;
    ds_taps_ = fbuf(3 * width);
    if (df2_) {
        if (!df_gpu23_) df_scratch();
        if (!ds_host_) ds_host_ = (float*) hmalloc(2 * kWidth * sizeof(float), "pinned memory");
        if (!h_block_) h_block_ = (Likeliest*) hmalloc(dflash2::kBlock * sizeof(Likeliest), "pinned memory");
        return;
    }
    const size_t b = kDsBlock;
    ds_x_ = fbuf(b * kWidth);
    ds_xa_ = fbuf(b * kWidth);
    ds_q_ = fbuf(b * kQuery);
    ds_qr_ = fbuf(b * kQuery);
    ds_k_ = fbuf(b * kKey);
    ds_kr_ = fbuf(b * kKey);
    ds_v_ = fbuf(b * kKey);
    ds_ao_ = fbuf(b * kQuery);
    ds_out_ = fbuf(b * kWidth);
    ds_g_ = fbuf(b * kDsFfn);
    ds_u_ = fbuf(b * kDsFfn);
    ds_h_ = fbuf(b * kDsFfn);
    ds_cur_ = fbuf(b * kWidth);
    ds_base_ = fbuf((b - 1) * (size_t) n_vocab_);
    ds_state_ = fbuf(kDsRank);
    ds_w1row_ = fbuf(kDsRank);
    ds_jx_ = fbuf(2 * kDsRank + 2 * kWidth);
    ds_jo_ = fbuf(3 * kDsRank);
    ds_lat_ = fbuf(kDsRank);
    ds_att_ = fbuf(dspark::attend_floats());
    ds_b2_ = fbuf((size_t) n_vocab_);
    ds_w2_ = (uint16_t*) fbuf(((size_t) kDsRank * n_vocab_ + 1) / 2);
    if (!ds_host_) ds_host_ = (float*) hmalloc(b * kWidth * sizeof(float), "pinned memory");
    ds_head_ready_ = false;
    if (scratch_ && ds_w2_host_ && ds_b2_host_ && ds_w2_ && ds_b2_) prepare_head();
}

void Inkling2::conv(float* x, float* state, const float* w, const float* dynamic, int width, int n) {
    if (n <= kConvSpan) {
        conv_kernel<<<div_up(width, kThreads), kThreads, 0, cs(st_)>>>(x, state, w, dynamic, width, n);
        return;
    }
    const dim3 grid(div_up(width, kThreads), div_up(n, kConvSpan));
    conv_halo_kernel<<<grid, kThreads, 0, cs(st_)>>>(x, state, halo_, width);
    conv_span_kernel<<<grid, kThreads, 0, cs(st_)>>>(x, state, halo_, w, dynamic, width, n);
}

void Inkling2::store(const float* x, Cache& c, int side, int n) {
    auto& values = c.planes[2 * side];
    auto& scales = c.planes[2 * side + 1];
    const bool swa = local(visit_);
    int64_t pos = swa ? pos_ % ring_ : pos_;
    for (int at = 0; at < n;) {
        const int m = swa ? std::min(n - at, ring_ - (int) pos) : n;
        if (q8()) store_q8(x, (int8_t*) values.data, (uint16_t*) scales.data, n_key_, pos, m, st_);
        else store_f16(x, (uint16_t*) values.data, n_key_, pos, m, st_);
        at += m;
        x += (size_t) m * n_key_;
        pos = 0;
    }
}

#if defined(STRATA_HIP_GFX906)
void Inkling2::copy_swa(const Cache& live, const Cache& stage, int ring_head, int stage_head, int n, bool gather) {
    if (!n) return;
    SwaCopyPlanes p{};
    for (int j = 0; j < 4; ++j) {
        p.ring[j] = reinterpret_cast<uint4*>(live.planes[j].data);
        p.stage[j] = reinterpret_cast<uint4*>(stage.planes[j].data);
        p.stride[j] = (int) (live.planes[j].stride / sizeof(uint4));
    }
    const dim3 grid(div_up(n * p.stride[0], kThreads), 4);
    if (gather) swa_copy_kernel<true><<<grid, kThreads, 0, cs(st_)>>>(p, ring_, ring_head, stage_head, n);
    else swa_copy_kernel<false><<<grid, kThreads, 0, cs(st_)>>>(p, ring_, ring_head, stage_head, n);
    ck(cudaGetLastError(), gather ? "gather Inkling SWA history" : "commit Inkling SWA rows");
}
#endif

void Inkling2::attention(const Block& b, int n, const float* k_in, const float* v_in) {
    const float* kp = k_in ? k_in : k_;
    const float* vp = v_in ? v_in : v_;
    Cache& live = caches_[visit_];
    const bool swa = local(visit_);
    int64_t pos = pos_;
    int ring = swa ? ring_ : 0;
    Cache* selected = &live;
#if defined(STRATA_HIP_GFX906)
    Cache stage;
    inkling::SwaStageLayout layout;
    if (swa && swa_stage_rows_ && pos_ + n > ring_) {
        layout = inkling::swa_stage_layout(pos_, kWindow);
        if (!layout.fits_ring(n, ring_)) {
            // No verifier needs the wide path; a staged commit cannot preserve arbitrary rejected tails.
            if (windowed_ || kv_stored_ || layout.append() + n > swa_stage_rows_)
                throw std::logic_error("Inkling SWA stage outside its bulk input geometry");
            // On a local visit the sink projection is the last reader of input_[0]. No clearing is needed:
            // only written history/append rows are read, and the unused Flash tile lanes are masked. All
            // stage accesses and the next writer of input_[0] (including drafter_kept) are ordered on st_.
            auto* base = reinterpret_cast<uint8_t*>(input_[0]);
            if (!base || reinterpret_cast<uintptr_t>(base) % inkling::kSwaCopyAlign)
                throw std::logic_error("Inkling SWA stage input scratch is unaligned");
            for (int j = 0; j < 4; ++j) {
                const auto stride = live.planes[j].stride;
                stage.planes[j] = {base, stride};
                base += (size_t) swa_stage_rows_ * stride;
            }
            selected = &stage;
            pos = layout.append();   // shifted by a whole Flash key tile, relative distances unchanged
            ring = 0;
            copy_swa(live, stage, (int) (layout.first % ring_), layout.head, layout.history, true);
        }
    }
#endif
    Cache& c = *selected;
    if (fused(n) && q8()) {
        const int blocks = 2 * (n_key_ / 32) * n;
        if (!kv_stored_)
            store2_q8_kernel<<<div_up(blocks, 256), 256, 0, cs(st_)>>>(
                kp, vp, (int8_t*) c.planes[0].data, (__half*) c.planes[1].data, (int8_t*) c.planes[2].data,
                (__half*) c.planes[3].data, pos, ring, n, n_key_);
        kv_stored_ = false;
    }
#if defined(STRATA_HIP_GFX906)
    else if (selected != &live) {
        store_q8(kp, (int8_t*) c.planes[0].data, (uint16_t*) c.planes[1].data, n_key_, pos, n, st_);
        store_q8(vp, (int8_t*) c.planes[2].data, (uint16_t*) c.planes[3].data, n_key_, pos, n, st_);
    }
#endif
    else {
        store(kp, c, 0, n);
        store(vp, c, 1, n);
    }
    inkling::Attend a;
    a.q = q_;
    a.rel = rel_;
    a.table = fp(b.table);
    a.sink = sink_;
    a.gate = gate_;
    a.k = {c.planes[0].data, (const uint16_t*) c.planes[1].data};
    a.v = {c.planes[2].data, (const uint16_t*) c.planes[3].data};
    a.pos = pos;
    a.n = n;
    a.window = swa ? kWindow : 0;
    a.ring = ring;
    a.kv_heads = n_kv_;
    a.out = att_;
    const auto attend_local = [&] {
#if defined(STRATA_HIP_GFX906)
        if (n >= 2047 && !swa && kv_helper_ && !phase_dualtp_active_ &&
            inkling::kv_helpers::try_flash_rows(*kv_helper_, a)) return;
#endif
        // Prompt batches only: a token's answer and short calls keep the usual kernels and their sums.
        if (flash_ && n >= kStageRows) inkling::attend_flash(a, cs(st_));
        else if (decode_ && n < kStageRows) {
            inkling::attend_decode(a, chunks_, cs(st_), attq_, factored_decode_ ? scores_ : nullptr,
                                   factored_decode_ ? inkling::plain_score_floats(attend_tiles_) : 0);
        } else inkling::attend_plain(a, scores_, partial_, attend_tiles_, cs(st_));
    };
#if defined(STRATA_HIP_GFX906)
    if (kv_helper_ && !swa && !phase_dualtp_active_) {
        using namespace inkling::kv_helpers;
        static_assert(kVisits / kBlock == kFullVisits && kKv / 2 == kOwnerKvHeads && kWindow == kTableRows &&
                      kRelative == kRelativeWidth && kHead == kHeadWidth && kMaxDecodeRows < kStageRows);
        const auto path = try_visit(*kv_helper_, a, visit_ / kBlock, chunks_, scores_,
                                    inkling::plain_score_floats(attend_tiles_), attq_);
        if (path == Visit::Offloaded) return;
        if (path == Visit::Seeding) {
            try {
                attend_local();
            } catch (...) {
                join_seed(*kv_helper_);
                throw;
            }
            join_seed(*kv_helper_);
            return;
        }
    }
#endif
    attend_local();
#if defined(STRATA_HIP_GFX906)
    if (selected != &live) {
        const int count = std::min(n, ring_), skip = n - count;
        copy_swa(live, stage, (int) ((pos_ + skip) % ring_), layout.append() + skip, count, false);
    }
#endif
}

void Inkling2::depth_mix(const float* query, float* dst, int first, int n) {
    static_assert(kWidth == kRowWidth);
    const size_t at = (size_t) first * kWidth;
    MixInputs m{};
    for (int i = 0; i < depth_n_; ++i) m.x[i] = depth_[i] + at;
    m.x[depth_n_] = r_ + at;
    mix_kernel<<<n, kMixThreads, 0, cs(st_)>>>(m, depth_n_ + 1, query, dst);
}

#if defined(STRATA_HIP_GFX906)
template<bool Ffn>
void Inkling2::depth_mix_norm(const float* query, const float* weight, float* y, void* q, int n) {
    MixInputs m{};
    for (int i = 0; i < depth_n_; ++i) m.x[i] = depth_[i];
    m.x[depth_n_] = r_;
    const MixNormRows rows{{query}, {weight}, {y}, {q}};
    fused_mix_norm<Ffn><<<n, kMixThreads, 0, cs(st_)>>>(m, depth_n_ + 1, rows, rms_eps_);
}
#endif

void Inkling2::control(const Block& b, int il, int stream, float* in, int n, const float* shared_depth) {
    const size_t row = 2 * (size_t) (il - kStart) + (stream >= 2 ? 1 : 0);
    const float* query = b.msg_query.p
                             ? fp(b.msg_query) + (size_t) stream * kWidth
                             : fp(stream == 0 || stream == 3 ? loop_query_ : loop_kv_query_) + row * kWidth;
    if (fused(n)) control_in_kernel<<<n, kMixThreads, 0, cs(st_)>>>(bank0_, bank1_, query, in, msg_, pair_);
    else {
        mix_kernel<<<n, kMixThreads, 0, cs(st_)>>>(MixInputs{{bank0_, bank1_}}, 2, query, msg_);
        if (!shared_depth) rms_norm(in, nullptr, pair_, kWidth, n, 1.e-6f, st_, 2 * kWidth);
        rms_norm(msg_, nullptr, pair_ + kWidth, kWidth, n, 1.e-6f, st_, 2 * kWidth);
    }
    const Control& c = b.control[stream];
    if (fused(n) && c.gain.type == kTypeF32 && c.forget.type == kTypeF32 && c.gain.scale == 1.f &&
        c.forget.scale == 1.f) {
        lin(c.down, pair_, rank_, n);
        control_out_kernel<<<dim3(kWidth / 32, n), 256, 0, cs(st_)>>>(fp(c.gain), fp(c.forget), rank_, msg_, in,
                                                                      kRank);
        return;
    }
    lin(c.down, pair_, rank_, n);
    lin(c.gain, rank_, gain_, n);
    lin(c.forget, rank_, forget_, n);
#if defined(STRATA_HIP_GFX906)
    if (shared_depth) {
        control_from_shared<<<div_up(n * kWidth, kThreads), kThreads, 0, cs(st_)>>>(
            shared_depth, in, msg_, gain_, forget_, n * kWidth);
        return;
    }
#endif
    control_kernel<<<div_up(n * kWidth, kThreads), kThreads, 0, cs(st_)>>>(in, msg_, gain_, forget_, n * kWidth);
}

// Decode rows of a visit with three streams (q, k, v): each stream's mix, control and norm as control() and
// layer_head()'s loop do them, all three in each launch, into s3in_ (the streams one after the other).  Five launches
// in place of fifteen with the control, two in place of six without; every row bit for bit the per-stream launches'.
// False, and nothing launched, where the per-stream path has to run (a prompt batch, the first layer, a control that
// is not the F32 products control() fuses).
bool Inkling2::heads(const Block& b, int il, bool swa, bool loop, int n, const Dev* const* norm) {
    if (!fused(n) || il == 0 || !s3in_ || depth_n_ + 1 > 5) return false;
    if (loop)
        for (const Control& c : {b.control[0], b.control[1], b.control[2]})
            if (c.down.type != kTypeF32 || c.down.scale != 1.f || c.gain.type != kTypeF32 || c.forget.type != kTypeF32 ||
                c.gain.scale != 1.f || c.forget.scale != 1.f || c.down.n_in != 2 * kWidth || c.down.n_out != kRank)
                return false;
    const size_t w = (size_t) n * kWidth;
    MixInputs m{};
    for (int i = 0; i < depth_n_; ++i) m.x[i] = depth_[i];
    m.x[depth_n_] = r_;
    Streams3 ms{};
    for (int i = 0; i < 3; ++i) {
        ms.query[i] = fp(b.aquery) + (swa ? 0 : i * kWidth);
        ms.out[i] = s3cur_ + i * w;
    }
#if defined(STRATA_HIP_GFX906)
    if (!loop && mix_rms_fuse_ && norm_q8_ && n <= 8) {
        MixNormRows rs{};
        const size_t wq = (size_t) n * kernels::native_q8_1_bytes(kWidth);
        for (int i = 0; i < 3; ++i) {
            rs.query[i] = ms.query[i];
            rs.weight[i] = fp(*norm[i]);
            rs.normalized[i] = s3in_ + i * w;
            rs.quant[i] = (uint8_t*) s3q8_ + i * wq;
        }
        // Only the normalized FP32 and Q8_1 are live on the non-loop path.
        fused_mix_norm<false><<<dim3(n, 3), kMixThreads, 0, cs(st_)>>>(m, depth_n_ + 1, rs, rms_eps_);
        return true;
    }
#endif
    if (loop) {
        Control3 c{};
        F32Segments down;
        for (int i = 0; i < 3; ++i) {
            const size_t row = 2 * (size_t) (il - kStart) + (i >= 2 ? 1 : 0);
            c.query[i] = b.msg_query.p ? fp(b.msg_query) + (size_t) i * kWidth
                                       : fp(i == 0 ? loop_query_ : loop_kv_query_) + row * kWidth;
            c.cur[i] = s3cur_ + i * w;
            c.msg[i] = s3msg_ + i * w;
            c.pair[i] = s3pair_ + 2 * i * w;
            c.gain[i] = fp(b.control[i].gain);
            c.forget[i] = fp(b.control[i].forget);
            c.rank[i] = s3rank_ + (size_t) i * n * kRank;
            down.w[i] = fp(b.control[i].down);
            down.x[i] = c.pair[i];
            down.y[i] = s3rank_ + (size_t) i * n * kRank;
        }
#if defined(STRATA_HIP_GFX906)
        if (mix_ctrl_fuse_) {
            mix_control_in3_kernel<<<dim3(n, 3), kMixThreads, 0, cs(st_)>>>(m, depth_n_ + 1, ms, bank0_, bank1_, c);
        } else
#endif
        {
            mix3_kernel<<<dim3(n, 3), kMixThreads, 0, cs(st_)>>>(m, depth_n_ + 1, ms);
            control_in3_kernel<<<dim3(n, 3), kMixThreads, 0, cs(st_)>>>(bank0_, bank1_, c);
        }
        // lin(c.down, pair_, rank_, n) of each stream: gemv_f32_waves() with a row of outputs to its input
        gemv_f32_waves_segments(down, 3, 2 * kWidth, kRank, kRank, n, st_);
        control_out3_kernel<<<dim3(kWidth / 32, n, 3), 256, 0, cs(st_)>>>(c, kRank);
    } else {
        mix3_kernel<<<dim3(n, 3), kMixThreads, 0, cs(st_)>>>(m, depth_n_ + 1, ms);
    }
    if (norm_q8_) {   // the three stream norms and their Q8_1 in one launch (s3in_ then already Q8_1 in s3q8_)
        kernels::NormQ8Row rs[3];
        // a stream's Q8_1 rows are n of them, back to back as the flat quantize laid them out: the stream's
        // slot in s3q8_ starts n rows in (layer_head's `wq` below reads at the same stride)
        const size_t wq = (size_t) n * kernels::native_q8_1_bytes(kWidth);
        for (int i = 0; i < 3; ++i)
            rs[i] = {s3cur_ + i * w, fp(*norm[i]), s3in_ + i * w, (uint8_t*) s3q8_ + i * wq};
        kernels::native_rms_norm_q8_1_rows(rs, 3, kWidth, n, rms_eps_, sum_threads(kWidth), st_);
        return true;
    }
    NormSegments ns;
    for (int i = 0; i < 3; ++i) {
        ns.x[i] = s3cur_ + i * w;
        ns.w[i] = fp(*norm[i]);
        ns.y[i] = s3in_ + i * w;
    }
    rms_norm_segments(ns, 3, kWidth, n, rms_eps_, st_);
    return true;
}

void Inkling2::run_visits(int n) {
    // Includes a keep() that needs to replay a prefix: its taps supersede the provisional window.
    if (df_gpu23_) df_gpu23_->prov_valid = false;
#if defined(STRATA_HIP_GFX906)
    if (gpu_route_failed_) throw std::runtime_error("Inkling GPU route failed; reopen the TP2 engine");
    if (gpu_route_dev_ && (!all_cached() || !peer_->all_cached()))
        throw std::runtime_error("Inkling GPU route lost TP2 residency; reopen the engine");
    const bool gpu_route = gpu_route_dev_ && n >= 1 && n <= gpu_route_rows_ && n <= rows_ &&
                           n <= peer_->rows_ && !block_pass_ && gpu_route_host_fp();
#else
    constexpr bool gpu_route = false;
#endif
    // Engine::input already put the projected patches in r_. Only token embeddings take the learned norm.
    const auto norm_text = [&](int first, int count) {
        const size_t at = (size_t) first * kWidth;
        rms_norm(r_ + at, fp(token_norm_), cur_ + at, kWidth, count, rms_eps_, st_);
        axpy(cur_ + at, 1.f, r_ + at, count * kWidth, false, st_);
    };
    if (!first_visit_) {
        int first = 0;
        for (const Image& im : images_) {   // sorted, non-overlapping; a prompt chunk can cut through a picture
            if (im.at >= pos_ + n) break;
            if (im.at + im.n <= pos_) continue;
            if (im.at > pos_ + first) norm_text(first, (int) (im.at - pos_) - first);
            first = (int) std::min<int64_t>(n, im.at + im.n - pos_);
        }
        if (first < n) norm_text(first, n - first);
    }
    // half a warp an NVFP4 expert row for this call's few tokens only (iq_set_slot_lanes() is the process's)
#if defined(STRATA_HIP_GFX906)
    const bool own_lanes = !phase_dualtp_active_ && !prefill_only_;
#else
    const bool own_lanes = true;
#endif
    if (own_lanes) kernels::iq_set_slot_lanes(fused(n));
    struct Lanes {
        bool enabled;
        ~Lanes() { if (enabled) kernels::iq_set_slot_lanes(false); }
    } lanes{own_lanes};
    if (peer_) {
        peer_dispatch(gpu_route ? PeerCmd::kRunGpuVisits : PeerCmd::kRunVisits, n);
        try {
            if (!first_visit_) tp2_->broadcast_from_0(0, r_, n * kWidth, st_);
            run_visits_local(n, gpu_route);
#if defined(STRATA_HIP_GFX906)
            if (gpu_route) {
                peer_wait();
                if (gpu_route_host_[0] || peer_->gpu_route_host_[0])
                    throw std::runtime_error("Inkling GPU route found invalid routing data; call rejected; reopen the engine");
                count_gpu_route(n);
                peer_->count_gpu_route(n);
            }
#endif
        } catch (const std::exception& e) {
#if defined(STRATA_HIP_GFX906)
            if (gpu_route) {
                gpu_route_failed_ = true;
                win_ = 0;  // no previous verifier result can be kept after provisional state was overwritten
            }
#endif
            std::fprintf(stderr, "strata-poly: TP2 rank0 exception: %s\n", e.what());
            tp2_->abort_barrier();
            peer_wait();
            throw;
        } catch (...) {
#if defined(STRATA_HIP_GFX906)
            if (gpu_route) {
                gpu_route_failed_ = true;
                win_ = 0;
            }
#endif
            tp2_->abort_barrier();
            peer_wait();
            throw;
        }
        if (!gpu_route) peer_wait();
    } else {
        run_visits_local(n);
    }
}

void Inkling2::run_visits_local(int n, [[maybe_unused]] bool gpu_route) {
#if defined(STRATA_HIP_GFX906)
    if (gpu_route) {
        // Reuse the engine's slot table with its ordinary upload-on-change guard, never a second frozen map.
        if (!refresh_card_slots()) throw std::logic_error("Inkling GPU route resident slot table unavailable");
        zero_vram(gpu_route_dev_, sizeof(int32_t));
    }
    static const bool gpu_prefill_any = [] {
        const char* v = std::getenv("STRATA_INKLING_GPU_PREFILL_ANY");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    const bool gpu_prefill_any_check =
        gpu_prefill_any && !gpu_route && inkling_prefill_ && h_inkling_prefill_ &&
        n >= kStageRows && n < inkling_prefill::max_tokens - 1 && !recording_ && !opt_.dump && all_cached();
    if (gpu_prefill_any_check)
        ck(cudaMemsetAsync(&inkling_prefill_->any_invalid, 0, sizeof(int32_t), cs(st_)),
           "Inkling prefill clear any_invalid");
#endif
    fo_clean_ = false;   // fo_ is what the last call left
    if (!first_visit_) depth_n_ = 0;
#if defined(STRATA_HIP_GFX906)
    const size_t cut_bytes = (size_t) n * kWidth * sizeof(float);
    const auto push_cut_async = [&](const float* src, float* dst) {
        ck(cudaEventRecord(cut_ev_, cs(st_)), "PhaseDualTP cut ready");
        ck(cudaStreamWaitEvent(cut_st_, cut_ev_, 0), "PhaseDualTP cut wait");
        ck(cudaMemcpyPeerAsync(dst, cut_dst_->device_, src, device_, cut_bytes, cut_st_), "PhaseDualTP stagger P2P");
    };
#endif
    for (visit_ = first_visit_; visit_ < run_last_visit_; ++visit_) {
        if (visit_ == kStart) {
            axpy(r_, 1.f, prelude_, n * kWidth, false, st_);
#if defined(STRATA_HIP_GFX906)
            if (cut_dst_) push_cut_async(prelude_, cut_dst_->cut_prelude_);
#endif
        }
        if (visit_ == kStart + kCore || visit_ == kStart + 2 * kCore) {
            // The old partial and the one just before it are the previous pass's two core blocks.
            axpy(depth_[2], 1.f, bank0_, n * kWidth, false, st_);
            axpy(r_, 1.f, bank1_, n * kWidth, false, st_);
#if defined(STRATA_HIP_GFX906)
            if (cut_dst_ && visit_ == kStart + kCore) {
                ck(cudaEventRecord(cut_ev_, cs(st_)), "PhaseDualTP banks ready");
                ck(cudaStreamWaitEvent(cut_st_, cut_ev_, 0), "PhaseDualTP banks wait");
                ck(cudaMemcpyPeerAsync(cut_dst_->cut_bank0_, cut_dst_->device_, bank0_, device_, cut_bytes, cut_st_),
                   "PhaseDualTP stagger bank0");
                ck(cudaMemcpyPeerAsync(cut_dst_->cut_bank1_, cut_dst_->device_, bank1_, device_, cut_bytes, cut_st_),
                   "PhaseDualTP stagger bank1");
            }
#endif
            axpy(prelude_, 1.f, r_, n * kWidth, false, st_);
            depth_n_ = 1;
        }
        if (draft_)   // the drafter's taps: the residual entering visits 5, 23 and 41 (drafter_kept())
            for (int j = 0; j < 3; ++j)
                if (visit_ == kTap[j])
                    ck(cudaMemcpyAsync(ds_taps_ + (size_t) j * rows_ * kWidth, r_, (size_t) n * kWidth * sizeof(float),
                                       cudaMemcpyDeviceToDevice, cs(st_)),
                       "DSpark taps");
#if defined(STRATA_HIP_GFX906)
        if (sb_capture_df2_ && sb_tap2_ && visit_ == kTap[2])
            ck(cudaMemcpyAsync(sb_tap2_, r_, (size_t) n * kWidth * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
               "PhaseDualTP Stage B tap41");
#endif
        const int il = physical(visit_);
#if defined(STRATA_HIP_GFX906)
        if (gpu_route) {
            Layer& L = layers_[il];
            layer_head_impl(L, il, n, true);
            moe_shared(L, il, n);
            gpu_route_moe(L, il, n);
            layer_tail(L, il, n);
        } else
#endif
            layer(il, n);
#if defined(STRATA_HIP_GFX906)
        if (cut_dst_ && visit_ == 0) push_cut_async(depth_[0], cut_dst_->cut_depth0_);
#endif
    }
#if defined(STRATA_HIP_GFX906)
    if (prefill_fuse_ && state_next_ && n > kConvSpan) {
        const size_t n_vis = (size_t) (run_last_visit_ - first_visit_);
        const size_t off = conv_at(first_visit_);
        ck(cudaMemcpyAsync(state_ + off, state_next_ + off, n_vis * conv_floats_ * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(st_)),
           "Inkling prefill conv state commit");
    }
    if (gpu_prefill_any_check) {
        ck(cudaMemcpyAsync(&h_inkling_prefill_->invalid, &inkling_prefill_->any_invalid, sizeof(int32_t),
                           cudaMemcpyDeviceToHost, cs(st_)),
           "Inkling prefill deferred check");
        ck(cudaStreamSynchronize(cs(st_)), "Inkling prefill deferred check sync");
        if (h_inkling_prefill_->invalid)
            throw std::runtime_error("Inkling GPU prefill routing failed (deferred check)");
    }
    if (gpu_route) {
        const size_t bytes = (1 + size_t(kVisits) * n * inkling_route::kPicks) * sizeof(int32_t);
        ck(cudaMemcpyAsync(gpu_route_host_, gpu_route_dev_, bytes, cudaMemcpyDeviceToHost, cs(st_)),
           "Inkling GPU route report");
        ck(cudaStreamSynchronize(cs(st_)), "Inkling GPU route ready");
    }
#endif
}

void Inkling2::layer_head(Layer& L, int il, int n) { layer_head_impl(L, il, n, false); }

void Inkling2::layer_head_impl(Layer& L, int il, int n, bool gpu_route) {
    const Block& b = blocks_[il];
    // STRATA_INKLING_TAIL_MOE: this visit's cached experts wait in tail_sum_ for the tail launch, or fo_ holds
    // their add_scaled() sums as before.  One token only: the record carries no rows.  Every visit starts fresh.
    tail_sum_.n = 0;
    tail_record_ = moe_tail_ && fused(n) && L.moe && n == 1;
    const bool swa = local(visit_), loop = messages();
    const Dev* norm[3] = {&L.attn_norm, swa ? &L.attn_norm : &b.knorm, swa ? &L.attn_norm : &b.vnorm};
    const int streams = swa && !loop ? 1 : 3;
    const float *xq = input_[0], *xk = xq, *xv = xq;
    const void *kq = nullptr, *vq = nullptr;   // K's and V's own inputs as Q8_1, when heads() quantized them
    const bool mmv = n < kStageRows;
    const bool three = streams == 3 && heads(b, il, swa, loop, n, norm);
    bool q8in = three;   // the stream inputs are Q8_1 already (heads(), or the loop below when it fuses)
    const float* mqx_src = nullptr;
    int mqx_bucket = -1;
#if defined(STRATA_HIP_GFX906)
    const bool can_pfuse = prefill_fuse_ && mmq_ && n >= kStageRows && !half_sum_guard_;
    const auto mmq_bucket = [](uint32_t type) -> int {
        switch (type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1: return 0;  // DS4
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q6_K: return 1;  // D4
            default: return 2 + (int) type;
        }
    };
#endif
    if (three) {
        const size_t w = (size_t) n * kWidth, wq = (size_t) n * kernels::native_q8_1_bytes(kWidth);
        xq = s3in_;
        xk = s3in_ + w;
        xv = s3in_ + 2 * w;
        // the three inputs lie one after the other, and a Q8_1 block never straddles two: one launch for all
        // (heads() wrote their Q8_1 too when norm_q8_)
        if (!norm_q8_) kernels::native_quantize_q8_1(s3in_, s3q8_, kWidth * n * 3, 1, st_);
        kq = (const uint8_t*) s3q8_ + wq;
        vq = (const uint8_t*) s3q8_ + 2 * wq;
    } else {
        const float* shared_depth = nullptr;
#if defined(STRATA_HIP_GFX906)
        if (prefill_depth_share_ && swa && loop && !windowed_ && n >= kStageRows) {
            // Only SWA shares a depth query. Keep its raw mix in input_[2] until V's control reads it,
            // then V's final norm overwrites it on the same stream. Each bank norm writes only pair_'s second half.
            depth_mix(fp(b.aquery), input_[2], 0, n);
            rms_norm(input_[2], nullptr, pair_, kWidth, n, 1.e-6f, st_, 2 * kWidth);
            shared_depth = input_[2];
        }
#endif
        for (int i = 0; i < streams; ++i) {
#if defined(STRATA_HIP_GFX906)
            if (mix_rms_fuse_ && norm_q8_ && n <= 8 && (streams == 1 || n == 1) && il != 0 && !loop &&
                depth_n_ + 1 <= 5) {
                void* q = i == 0 ? xq_ : i == 1 ? kq8_ : vq8_;
                depth_mix_norm<false>(fp(b.aquery) + (swa ? 0 : i * kWidth), fp(*norm[i]), input_[i], q, n);
                q8in = true;
                continue;
            }
            if (can_pfuse && il != 0 && !loop && depth_n_ + 1 <= 5) {
                MixInputs fm{};
                for (int j = 0; j < depth_n_; ++j) fm.x[j] = depth_[j];
                fm.x[depth_n_] = r_;
                const float* qptr = fp(b.aquery) + (swa ? 0 : i * kWidth);
                const int qb = mmq_bucket(b.q.type);
                const bool pack_this =
                    (i == 0 && (qb == 0 || qb == 1) && Mmq::fits((int) b.q.type, b.q.n_out, b.q.n_in));
                if (pack_this) {
                    if (qb == 0)
                        fused_mix_rms_mmq_prefill<true><<<n, kMixThreads, 0, cs(st_)>>>(
                            fm, depth_n_ + 1, qptr, fp(*norm[i]), reinterpret_cast<float4*>(input_[i]),
                            reinterpret_cast<PrefillMmqBlock*>(mqx_), n, rms_eps_, 0.f);
                    else
                        fused_mix_rms_mmq_prefill<false><<<n, kMixThreads, 0, cs(st_)>>>(
                            fm, depth_n_ + 1, qptr, fp(*norm[i]), reinterpret_cast<float4*>(input_[i]),
                            reinterpret_cast<PrefillMmqBlock*>(mqx_), n, rms_eps_, 0.f);
                    mqx_src = input_[0];
                    mqx_bucket = qb;
                } else {
                    fused_mix_rms_mmq_prefill<false><<<n, kMixThreads, 0, cs(st_)>>>(
                        fm, depth_n_ + 1, qptr, fp(*norm[i]), reinterpret_cast<float4*>(input_[i]),
                        nullptr, n, rms_eps_, 0.f);
                }
                continue;
            }
#endif
            if (il == 0) axpy(r_, 1.f, cur_, n * kWidth, false, st_);
            else if (!shared_depth) depth_mix(fp(b.aquery) + (swa ? 0 : i * kWidth), cur_, 0, n);
            if (loop) control(b, il, i, cur_, n, shared_depth);
            // this stream's norm and its Q8_1 in one launch; K and V write their own single-row buffers
            if (norm_q8_ && mmv && (streams == 1 || n == 1)) {
                void* q = streams == 1 || i == 0 ? xq_ : i == 1 ? (void*) kq8_ : (void*) vq8_;
                const kernels::NormQ8Row r{cur_, fp(*norm[i]), input_[i], q};
                kernels::native_rms_norm_q8_1_rows(&r, 1, kWidth, n, rms_eps_, sum_threads(kWidth), st_);
                q8in = true;
            } else rms_norm(cur_, fp(*norm[i]), input_[i], kWidth, n, rms_eps_, st_);
        }
        if (streams == 3) {
            xk = input_[1];
            xv = input_[2];
            if (q8in && n == 1) kq = kq8_, vq = vq8_;
        }
    }
    // MMVQ uses the same Q8_1 input for Q, relative bias and the attention gate (also K and V when
    // they share the input). The matrix-matrix prompt path uses its own layout instead.
    if (mmv && !q8in) kernels::native_quantize_q8_1(xq, xq_, kWidth * n, 1, st_);
    const void* shared = three ? s3q8_ : mmv ? xq_ : nullptr;
    if (fused(n) && n == 1) project(b, xq, xk, xv, streams == 1, shared, kq, vq);
    else if (q8_cols_ && fused(n) && n <= 8) project_cols(b, xq, xk, xv, streams == 1, shared, kq, vq, n);
#if defined(STRATA_HIP_GFX906)
    else if (can_pfuse) {
        struct PPart {
            const Dev* w;
            const float* x;
            float* y;
            bool done;
        } parts[5] = {{&b.q, xq, q_, false},
                      {&b.rel, xq, rel_, false},
                      {&b.gate, xq, gate_, false},
                      {&b.k, xk, k_, false},
                      {&b.v, xv, v_, false}};
        if (mqx_src) {
            for (PPart& p : parts) {
                if (p.x == mqx_src && Mmq::fits((int) p.w->type, p.w->n_out, p.w->n_in) &&
                    mmq_bucket(p.w->type) == mqx_bucket) {
                    lin(*p.w, p.x, p.y, n, mqx_);
                    p.done = true;
                }
            }
        }
        for (int i = 0; i < 5; ++i) {
            if (parts[i].done) continue;
            const Dev& w = *parts[i].w;
            if (Mmq::fits((int) w.type, w.n_out, w.n_in)) {
                const int bk = mmq_bucket(w.type);
                Mmq::quantize(parts[i].x, nullptr, mqx_, (int) w.type, w.n_in, w.n_in, n, st_);
                for (int j = i; j < 5; ++j) {
                    if (!parts[j].done && parts[j].x == parts[i].x &&
                        Mmq::fits((int) parts[j].w->type, parts[j].w->n_out, parts[j].w->n_in) &&
                        parts[j].w->n_in == w.n_in && mmq_bucket(parts[j].w->type) == bk) {
                        lin(*parts[j].w, parts[j].x, parts[j].y, n, mqx_);
                        parts[j].done = true;
                    }
                }
            } else {
                lin(w, parts[i].x, parts[i].y, n);
                parts[i].done = true;
            }
        }
    }
#endif
    else {
        lin(b.q, xq, q_, n, shared);
        lin(b.k, xk, k_, n, streams == 1 ? shared : kq);
        lin(b.v, xv, v_, n, streams == 1 ? shared : vq);
        lin(b.rel, xq, rel_, n, shared);
        lin(b.gate, xq, gate_, n, shared);
    }
    lin(b.sink, xq, sink_, n);

    float* hist = state_ + conv_at(visit_);
    float* k_cur = k_;
    float* v_cur = v_;
#if defined(STRATA_HIP_GFX906)
    const bool oop_conv = prefill_fuse_ && state_next_ && n > kConvSpan;
#else
    constexpr bool oop_conv = false;
#endif
    const auto kv_conv = [&](float* x, float* state, const Dev& w, const Dev& dyn, const float* input) {
        if (!swa) lin(dyn, input, taps_, n);
        conv(x, state, fp(w), swa ? nullptr : taps_, n_key_, n);
    };
    const int floor = swa ? 0 : floor_;
    if (fused(n)) {
        if (!swa) {
            if (kvproj_fuse_ && b.kproj.type == kTypeF32 && b.vproj.type == kTypeF32 &&
                b.kproj.scale == 1.f && b.vproj.scale == 1.f &&
                b.kproj.n_in == b.vproj.n_in && b.kproj.n_out == b.vproj.n_out) {
                F32Segments kvp;
                kvp.w[0] = fp(b.kproj); kvp.x[0] = xk; kvp.y[0] = taps_;
                kvp.w[1] = fp(b.vproj); kvp.x[1] = xv; kvp.y[1] = taps2_;
                gemv_f32_waves_segments(kvp, 2, b.kproj.n_in, b.kproj.n_out, b.kproj.n_out, n, st_);
            } else {
                lin(b.kproj, xk, taps_, n);
                lin(b.vproj, xv, taps2_, n);
            }
        }
        const Conv ck{k_, hist, fp(b.kconv), swa ? nullptr : taps_, snap_of(hist)};
        const Conv cv{v_, hist + 3 * n_key_, fp(b.vconv), swa ? nullptr : taps2_, snap_of(hist + 3 * n_key_)};
        HeadNorms h{{{q_, fp(b.qclip), kHead, n_heads_, qk_clip_, floor}, {k_, nullptr, kHead, n_kv_, qk_clip_, 0},
                     {v_, nullptr, kHead, n_kv_, 0.f, 0}, {rel_, fp(b.rclip), kRelative, n_heads_, r_clip_, floor}}};
        const bool can_fuse_kv =
            n == 1 || (heads_kv_rows_ &&
                       (!swa || !swa_stage_rows_ || pos_ + n <= ring_ ||
                        inkling::swa_stage_layout(pos_, kWindow).fits_ring(n, ring_)));
        if (q8() && can_fuse_kv) {   // attention() finds K and V stored
            Cache& c = caches_[visit_];
            const KvStore to{(int8_t*) c.planes[0].data, (int8_t*) c.planes[2].data, (__half*) c.planes[1].data,
                             (__half*) c.planes[3].data, swa ? pos_ % ring_ : pos_};
            if (n == 1)
                heads_kv_kernel<<<2 * n_heads_ + 2 * n_kv_, kThreads, 0, cs(st_)>>>(ck, cv, h, to, rms_eps_, pos_,
                                                                                    alpha_, rec_n_, n_key_);
            else
                heads_kv_rows_kernel<<<2 * n_heads_ * n + 2 * n_kv_, kThreads, 0, cs(st_)>>>(
                    ck, cv, h, to, rms_eps_, pos_, swa ? ring_ : 0, alpha_, rec_n_, n_key_, n);
            kv_stored_ = true;
        } else {
            conv2_kernel<<<div_up(2 * n_key_, kThreads), kThreads, 0, cs(st_)>>>(ck, cv, n_key_, n, rec_n_);
            head_norms_kernel<<<dim3(2 * n_heads_ + 2 * n_kv_, n), kThreads, 0, cs(st_)>>>(h, rms_eps_, pos_, alpha_);
        }
#if defined(STRATA_HIP_GFX906)
    } else if (oop_conv) {
        k_cur = att_;
        v_cur = att_ + (size_t) n * n_key_;
        float* hist_next = state_next_ + conv_at(visit_);
        const dim3 gkv(div_up(n_key_, kThreads), div_up(n, kConvSpan));
        if (!swa) lin(b.kproj, xk, taps_, n);
        conv_fused_prefill_kernel<<<gkv, kThreads, 0, cs(st_)>>>(
            k_, hist, fp(b.kconv), swa ? nullptr : taps_, k_cur, hist_next, n_key_, n);
        if (!swa) lin(b.vproj, xv, taps_, n);
        conv_fused_prefill_kernel<<<gkv, kThreads, 0, cs(st_)>>>(
            v_, hist + 3 * n_key_, fp(b.vconv), swa ? nullptr : taps_, v_cur, hist_next + 3 * n_key_, n_key_, n);
#endif
    } else {
        kv_conv(k_, hist, b.kconv, b.kproj, xk);
        kv_conv(v_, hist + 3 * n_key_, b.vconv, b.vproj, xv);
    }
    const auto heads = [&](float* x, const float* w, int width, int count, float clip, int log_floor) {
        head_norm_kernel<<<dim3(count, n), kThreads, 0, cs(st_)>>>(x, w, width, count, clip, rms_eps_, pos_, log_floor,
                                                                  alpha_);
    };
    if (!fused(n)) {
        heads(q_, fp(b.qclip), kHead, n_heads_, qk_clip_, floor);
        heads(k_cur, nullptr, kHead, n_kv_, qk_clip_, 0);
        heads(v_cur, nullptr, kHead, n_kv_, 0.f, 0);
        heads(rel_, fp(b.rclip), kRelative, n_heads_, r_clip_, floor);
    }
    dbg("Qcur", visit_, q_, n_query_);
    dbg("Kcur", visit_, k_cur, n_key_);
    attention(b, n, k_cur, v_cur);
#if defined(STRATA_HIP_GFX906)
    bool o_done = false;
    if (q4_0_dense_cols_ && fused(n) && n >= 2 && n <= 8 && attq_ && b.o.type == GGML_TYPE_Q4_0 && b.o.scale == 1.f) {
        kernels::MmvqSegments os;
        os.w[0] = b.o.p;
        os.x[0] = attq_;
        os.y[0] = ao_;
        os.rows[0] = b.o.n_out;
        os.count = 1;
        o_done = kernels::native_inkling_q4_0_segments_cols(os, b.o.n_in, n, st_);
    }
    if (!o_done)
#endif
    lin(b.o, att_, ao_, n, fused(n) ? attq_ : nullptr);   // attend_decode() quantized it
    if (tp2_) tp2_->all_reduce(tp_rank_, ao_, n * kWidth, st_);
    if (fused(n))
        conv_add_kernel<<<div_up(kWidth, kThreads), kThreads, 0, cs(st_)>>>(
            ao_, nullptr, hist + 6 * n_key_, fp(b.aconv), r_, il % kBlock == 0 ? depth_[depth_n_++] : nullptr, kWidth,
            n, false, TailSum{}, snap_of(hist + 6 * n_key_), rec_n_);
#if defined(STRATA_HIP_GFX906)
    else if (oop_conv) {
        const dim3 gw(div_up(kWidth, kThreads), div_up(n, kConvSpan));
        conv_add_prefill_kernel<<<gw, kThreads, 0, cs(st_)>>>(
            ao_, hist + 6 * n_key_, fp(b.aconv), r_, il % kBlock == 0 ? depth_[depth_n_++] : nullptr,
            state_next_ + conv_at(visit_) + 6 * n_key_, kWidth, n);
    }
#endif
    else conv(ao_, hist + 6 * n_key_, fp(b.aconv), nullptr, kWidth, n);
    if (fused(n) || oop_conv) {   // conv_add_kernel / conv_add_prefill_kernel did the rest
    } else if (il % kBlock == 0) {
        axpy(r_, 1.f, depth_[depth_n_++], n * kWidth, false, st_);
        axpy(ao_, 1.f, r_, n * kWidth, false, st_);
    } else axpy(ao_, 1.f, r_, n * kWidth, true, st_);

    bool ffn_normed = false;
    if (loop) {
#if defined(STRATA_HIP_GFX906)
        const Control& fc = b.control[3];
        if (mix_ctrl_fuse_ && fused(n) && depth_n_ + 1 <= 5 && fc.gain.type == kTypeF32 && fc.forget.type == kTypeF32 &&
            fc.gain.scale == 1.f && fc.forget.scale == 1.f) {
            MixInputs fm{};
            for (int i = 0; i < depth_n_; ++i) fm.x[i] = depth_[i];
            fm.x[depth_n_] = r_;
            const size_t row = 2 * (size_t) (il - kStart) + 1;
            const float* cquery = b.msg_query.p ? fp(b.msg_query) + 3 * (size_t) kWidth
                                                : fp(loop_query_) + row * kWidth;
            mix_control_in_kernel<<<n, kMixThreads, 0, cs(st_)>>>(fm, depth_n_ + 1, fp(b.fquery), cur_,
                                                                  bank0_, bank1_, cquery, msg_, pair_);
            lin(fc.down, pair_, rank_, n);
            control_out_kernel<<<dim3(kWidth / 32, n), 256, 0, cs(st_)>>>(fp(fc.gain), fp(fc.forget), rank_, msg_, cur_,
                                                                          kRank);
        } else
#endif
        {
            depth_mix(fp(b.fquery), cur_, 0, n);
            control(b, il, 3, cur_, n);
        }
#if defined(STRATA_HIP_GFX906)
    } else if (mix_rms_fuse_ && n <= 8 && depth_n_ + 1 <= 5) {
        depth_mix_norm<true>(fp(b.fquery), fp(L.ffn_norm), xn_, xq_, n);
        ffn_normed = true;
    } else if (prefill_fuse_ && n >= kStageRows && depth_n_ + 1 <= 5 && L.gate_inp.type == kTypeF32) {
        MixInputs fm{};
        for (int j = 0; j < depth_n_; ++j) fm.x[j] = depth_[j];
        fm.x[depth_n_] = r_;
        fused_mix_rms_ffn_prefill<<<n, kMixThreads, 0, cs(st_)>>>(
            fm, depth_n_ + 1, fp(b.fquery), fp(L.ffn_norm), reinterpret_cast<float4*>(xn_), rms_eps_);
        ffn_normed = true;
#endif
    } else {
        depth_mix(fp(b.fquery), cur_, 0, n);
    }
    if (!ffn_normed)
        kernels::native_add_rms_norm_q8_1(cur_, nullptr, fp(L.ffn_norm), xn_, xq_, kWidth, n, rms_eps_, 128, st_);
    // with STRATA_INKLING_ROUTE_DUAL and one token, the router's gemv below also writes the host h_rl_ and h_x_,
    // and moe_router() launches no copy2 for them
    // The dual-store variant uses the same GEMV dot product, but its host copies are only needed by stock moe().
    if (gpu_route) lin(L.gate_inp, xn_, rl_, n, xq_);
    else if (!batched(L, n)) moe_router(L, n);
}

bool Inkling2::fused(int n) const { return decode_ && n < kStageRows; }

// A window's states after its tokens (Engine::snap_at_, a token's rec_n_ floats after the one before), at the place
// `state` has in state_; null outside a window.  The window takes every token's (snap_tokens(): a place for each).
float* Inkling2::snap_of(float* state) const {
    if (!windowed_ || !snap_bits_) return nullptr;
    return snap_at_ + (state - state_);
}

// A token's q, gate and relative terms of input_[0] (xq_, quantized), and its K and V of the same or of their own
// inputs: one launch a weight format (kernels::native_mmvq_segments(), the rows bit for bit lin()'s), lin() for what
// that does not take.
// A window's five projections (2..8 rows): the Q8_0 ones that project() groups for one token sum in its order
// (native_q8_0_segments_cols), the others as lin() does, so row t equals the token's own projection bit for bit.
void Inkling2::project_cols(const Block& b, const float* xq, const float* xk, const float* xv, bool shared_kv,
                            const void* q8, const void* k8, const void* v8, int n) {
    struct Part {
        const Dev* w;
        const float* x;
        const void* xq;
        float* y;
    } parts[5] = {{&b.q, xq, q8, q_},
                  {&b.gate, xq, q8, gate_},
                  {&b.rel, xq, q8, rel_},
                  {&b.k, xk, shared_kv ? q8 : k8, k_},
                  {&b.v, xv, shared_kv ? q8 : v8, v_}};
    // the Q8_0 parts whose Q8_1 input is at hand in one launch (as project() groups them); the rest on their own
    kernels::MmvqSegments group;
    for (const Part& p : parts) {
        const Dev& w = *p.w;
        if (w.type == GGML_TYPE_Q8_0 && w.n_in == kWidth && w.scale == 1.f && p.xq) {
            group.w[group.count] = w.p;
            group.x[group.count] = p.xq;
            group.y[group.count] = p.y;
            group.rows[group.count++] = w.n_out;
        }
    }
    if (group.count && !kernels::native_q8_0_segments_cols_grouped(group, kWidth, n, st_))
        throw std::runtime_error("STRATA_INKLING_Q8_COLS: unsupported projection group");
#if defined(STRATA_HIP_GFX906)
    bool q4_done[5] = {};
    if (q4_0_dense_cols_ && n >= 2 && n <= 8 && q8) {
        kernels::MmvqSegments q4g;
        int q4_idx[3] = {};
        for (int i = 0; i < 3; ++i) {
            const Dev& w = *parts[i].w;
            if (w.type == GGML_TYPE_Q4_0 && w.n_in == kWidth && w.scale == 1.f && parts[i].xq == q8) {
                q4g.w[q4g.count] = w.p;
                q4g.x[q4g.count] = q8;
                q4g.y[q4g.count] = parts[i].y;
                q4g.rows[q4g.count] = w.n_out;
                q4_idx[q4g.count++] = i;
            }
        }
        if (q4g.count && kernels::native_inkling_q4_0_segments_cols(q4g, kWidth, n, st_)) {
            for (int k = 0; k < q4g.count; ++k) q4_done[q4_idx[k]] = true;
        }
    }
#endif
    for (int i = 0; i < 5; ++i) {
#if defined(STRATA_HIP_GFX906)
        if (q4_done[i]) continue;
#endif
        const Part& p = parts[i];
        const Dev& w = *p.w;
        if (w.type == GGML_TYPE_Q8_0 && w.n_in == kWidth && w.scale == 1.f) {
            if (p.xq) continue;   // in the group above
            // as lin() would quantize it: the rows' blocks back to back, one long vector
            kernels::native_quantize_q8_1(p.x, q8_, kWidth * n, 1, st_);
            if (!kernels::native_q8_0_segments_cols(w.p, q8_, p.y, kWidth, w.n_out, n, st_))
                throw std::runtime_error("STRATA_INKLING_Q8_COLS: unsupported projection shape");
        } else lin(w, p.x, p.y, n, p.xq);
    }
}

void Inkling2::project(const Block& b, const float* xq, const float* xk, const float* xv, bool shared_kv,
                       const void* q8, const void* k8, const void* v8) {
    if (!shared_kv && !k8) {
        kernels::native_quantize_q8_1(xk, kq8_, kWidth, 1, st_);
        kernels::native_quantize_q8_1(xv, vq8_, kWidth, 1, st_);
        k8 = kq8_;
        v8 = vq8_;
    }
    struct Part {
        const Dev* w;
        const float* x;
        const void* xq;
        float* y;
        bool done;
    } parts[5] = {{&b.q, xq, q8, q_, false},
                  {&b.gate, xq, q8, gate_, false},
                  {&b.rel, xq, q8, rel_, false},
                  {&b.k, xk, shared_kv ? q8 : k8, k_, false},
                  {&b.v, xv, shared_kv ? q8 : v8, v_, false}};
    const auto fits = [](const Dev& w) { return w.n_in == kWidth && w.scale == 1.f; };
#if defined(STRATA_HIP_GFX906)
    static const bool group_q4 = [] {
        const char* e = std::getenv("STRATA_INKLING_Q4_0_DENSE");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    const auto q4 = [&](int i) { return parts[i].w->type == GGML_TYPE_Q4_0 && fits(*parts[i].w); };
    if (group_q4 && tp_size_ == 2 && !block_pass_ && q8 && q4(1) && q4(2)) {
        // Cache once per worker/device, not once per visit. The gfx906 build may
        // also contain code objects for other architectures.
        static thread_local int arch_device = -1;
        static thread_local bool arch_ok = false;
        if (arch_device != device_) {
            cudaDeviceProp p{};
            arch_ok = cudaGetDeviceProperties(&p, device_) == cudaSuccess && p.warpSize == 64 &&
                      kernels::gfx_arch_is(p.gcnArchName, "gfx906");
            arch_device = device_;
        }
        if (arch_ok) {
            kernels::MmvqSegments s;
            const int first = q4(0) ? 0 : 1;
            for (int i = first; i < 3; ++i) {
                s.w[s.count] = parts[i].w->p;
                s.x[s.count] = parts[i].xq;
                s.y[s.count] = parts[i].y;
                s.rows[s.count++] = parts[i].w->n_out;
            }
            if (kernels::native_inkling_q4_0_segments(s, kWidth, st_))
                for (int i = first; i < 3; ++i) parts[i].done = true;
        }
    }
#endif
    for (int i = 0; i < 5; ++i) {
        if (parts[i].done || !fits(*parts[i].w)) continue;
        kernels::MmvqSegments s;
        const uint32_t type = parts[i].w->type;
        for (int j = i; j < 5; ++j)
            if (!parts[j].done && parts[j].w->type == type && fits(*parts[j].w)) {
                s.w[s.count] = parts[j].w->p;
                s.x[s.count] = parts[j].xq;
                s.y[s.count] = parts[j].y;
                s.rows[s.count++] = parts[j].w->n_out;
            }
        if (!kernels::native_mmvq_segments((int) type, s, kWidth, st_)) continue;
        for (int j = i; j < 5; ++j)
            if (parts[j].w->type == type && fits(*parts[j].w)) parts[j].done = true;
    }
    for (const Part& p : parts)
        if (!p.done) lin(*p.w, p.x, p.y, 1, p.xq);
}

void Inkling2::moe_add(int n) {
    if (fused(n)) cpu_sum_ = true;   // layer_tail() adds it
    else Engine::moe_add(n);
}

void Inkling2::layer_tail(Layer&, int il, int n) {
    if (tp2_) tp2_->all_reduce(tp_rank_, fo_, n * kWidth, st_);
    float* hist = state_ + conv_at(visit_) + 6 * n_key_ + 3 * kWidth;
    if (fused(n)) {
        // with the record's experts summed here (STRATA_INKLING_TAIL_MOE), moe() launched nothing for them
        conv_add_kernel<<<div_up(kWidth, kThreads), kThreads, 0, cs(st_)>>>(
            fo_, cpu_sum_ ? h_cpu_ : nullptr, hist, fp(blocks_[il].fconv), r_, nullptr, kWidth, n, true,
            tail_record_ ? tail_sum_ : TailSum{}, snap_of(hist), rec_n_);
        cpu_sum_ = false;
        fo_clean_ = true;
        tail_record_ = false;   // one record's life: layer_head() fills the next visit's
        dbg("l_out", visit_, r_, kWidth);
        return;
    }
#if defined(STRATA_HIP_GFX906)
    if (prefill_fuse_ && state_next_ && n > kConvSpan) {
        const dim3 gw(div_up(kWidth, kThreads), div_up(n, kConvSpan));
        conv_add_prefill_kernel<<<gw, kThreads, 0, cs(st_)>>>(
            fo_, hist, fp(blocks_[il].fconv), r_, nullptr,
            state_next_ + conv_at(visit_) + 6 * n_key_ + 3 * kWidth, kWidth, n);
        dbg("l_out", visit_, r_, kWidth);
        return;
    }
#endif
    conv(fo_, hist, fp(blocks_[il].fconv), nullptr, kWidth, n);
    axpy(fo_, 1.f, r_, n * kWidth, true, st_);
    dbg("l_out", visit_, r_, kWidth);
}

void Inkling2::route(int il, const float* logits, int32_t, int* ids, float* w, float* tmp) const {
    float* pick = tmp + n_expert_;
    for (int i = 0; i < n_expert_; ++i) {
        tmp[i] = 1.f / (1.f + std::exp(-logits[i]));
        pick[i] = tmp[i] + blocks_[il].bias[i];
    }
    float sum = 0.f;
    for (int i = 0; i < n_used_; ++i) {
        const int id = (int) (std::max_element(pick, pick + n_expert_) - pick);
        pick[id] = -INFINITY;
        ids[i] = id;
        sum += (w[i] = tmp[id]);
    }
    for (int i = 0; i < n_used_; ++i) w[i] *= expert_scale_ / (sum + 1.e-20f);
}

void Inkling2::moe_shared(const Layer&, int, int n) {
    // the previous layer's fused tail left fo_ 0 for the same rows
    if (!(fused(n) && fo_clean_)) zero_vram(fo_, (size_t) n * kWidth * sizeof(float));
    fo_clean_ = false;
}

// The kept tokens' rows into the drafter's K/V, at pos() on: each layer's mix of the four taps, normed, through its
// context projections (keys normed and turned, as graph_inkling2's injection does), as Q8_0; then the recurrent
// head's mix of the last token's taps into hn_'s first row.  The call's scratch (input_, q_, k_, v_) is free by now.
void Inkling2::drafter_kept(int n) {
    if (n <= 0) return;
    if (df2_) return df_kept(n);
    const size_t plane = (size_t) rows_ * kWidth;
    dspark::Taps t{{ds_taps_, ds_taps_ + plane, ds_taps_ + 2 * plane, hn_ + n_embd_}, {}};
    float* x = input_[0];
    for (int l = 0; l < kDsLayers; ++l) {
        const DsLayer& L = ds_[(size_t) l];
        std::copy(ds_mix_[l], ds_mix_[l] + 4, t.w);
        dspark::mix_taps(t, fp(L.ctx_norm), x, 0, n, ds_eps_, cs(st_));
        lin(L.k_ctx, x, k_, n);
        lin(L.v_ctx, x, v_, n);
        head_norm_rope(k_, kHead, fp(L.k_ctx_norm), q_, kHead, kKv, kHead, ds_theta_, pos_, n, ds_eps_, st_);
        Cache& c = ds_cache_[(size_t) l];
        const bool ring = ds_swa(l);
        int64_t at = ring ? pos_ % kWindow : pos_;
        for (int done = 0; done < n;) {
            const int m = ring ? (int) std::min<int64_t>(n - done, kWindow - at) : n - done;
            store_q8(q_ + (size_t) done * kKey, (int8_t*) c.planes[0].data, (uint16_t*) c.planes[1].data, kKey, at, m,
                     st_);
            store_q8(v_ + (size_t) done * kKey, (int8_t*) c.planes[2].data, (uint16_t*) c.planes[3].data, kKey, at, m,
                     st_);
            done += m;
            at = ring ? 0 : at + m;
        }
    }
    std::copy(ds_mix_[kDsLayers], ds_mix_[kDsLayers] + 4, t.w);
    dspark::mix_taps(t, fp(ds_rnn_norm_), hn_, n - 1, 1, ds_eps_, cs(st_));
}

// W' = W_head W_hidden_out and b' = W_head b_hidden_out, so that a step scores the vocabulary from its 256 latents
// and not from a 2304-wide correction (ninfer-3090 does the same).  Once, at the first guess: the head's rows a few
// columns of hidden_out at a time, through wlogits_.
void Inkling2::prepare_head() {
    if (ds_w2_host_) {
        std::vector<uint16_t> w2_t((size_t) kDsRank * (size_t) n_vocab_);
        const auto* src = (const uint16_t*) ds_w2_host_;
        for (int v = 0; v < n_vocab_; ++v) {
            const uint16_t* row = src + (size_t) v * kDsRank;
            for (int r = 0; r < kDsRank; ++r) {
                float f = 0.f;
                if (ds_w2_bf16_) {
                    const uint32_t bits = (uint32_t) row[r] << 16;
                    std::memcpy(&f, &bits, 4);
                } else {
                    f = ggml_fp16_to_fp32(row[r]);
                }
                w2_t[(size_t) r * (size_t) n_vocab_ + (size_t) v] = ggml_fp32_to_fp16(f);
            }
        }
        ck(cudaMemcpyAsync(ds_w2_, w2_t.data(), w2_t.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, cs(st_)),
           "DSpark markov_w2");
        ck(cudaMemcpyAsync(ds_b2_, ds_b2_host_, (size_t) n_vocab_ * sizeof(float), cudaMemcpyHostToDevice, cs(st_)),
           "DSpark markov_w2.bias");
        ck(cudaStreamSynchronize(cs(st_)), "DSpark head");
        ds_head_ready_ = true;
        return;
    }
    const int rows = std::min(kDsBlock, opt_.window);
    for (int j0 = 0; j0 < kDsRank; j0 += rows) {
        const int m = std::min(rows, kDsRank - j0);
        for (int r = 0; r < m; ++r)
            for (int i = 0; i < kWidth; ++i)
                ds_host_[(size_t) r * kWidth + i] = ds_hidden_[(size_t) i * kDsRank + (size_t) (j0 + r)];
        ck(cudaMemcpyAsync(ds_xa_, ds_host_, (size_t) m * kWidth * sizeof(float), cudaMemcpyHostToDevice, cs(st_)),
           "DSpark head");
        lin(output_, ds_xa_, wlogits_, m);
        dspark::to_half(wlogits_, ds_w2_ + (size_t) j0 * n_vocab_, n_vocab_, m, n_vocab_, cs(st_));
        ck(cudaStreamSynchronize(cs(st_)), "DSpark head");
    }
    std::copy(ds_hidden_b_.begin(), ds_hidden_b_.end(), ds_host_);
    ck(cudaMemcpyAsync(ds_xa_, ds_host_, kWidth * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "DSpark head");
    lin(output_, ds_xa_, ds_b2_, 1);
    ck(cudaStreamSynchronize(cs(st_)), "DSpark head");
    ds_head_ready_ = true;
}

// The block at the anchor `token`, the token at pos() that is picked and not fed: its embedding as the model takes it
// and seven masks, through the three layers (each query sees the drafter's rows before pos() and the block), the
// output norm, and the head's logits of the rows that guess.  The recurrent state starts at zero.
void Inkling2::ds_block(int32_t token) {
    const int64_t a = pos_;
    input(token, a, ds_host_);
    ck(cudaMemcpyAsync(ds_out_, ds_host_, kWidth * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "DSpark");
    rms_norm(ds_out_, fp(token_norm_), ds_x_, kWidth, 1, rms_eps_, st_);
    if (ds_mask_rows_ == kDsBlock - 1)
        ck(cudaMemcpyAsync(ds_x_ + kWidth, fp(ds_mask_), (size_t) (kDsBlock - 1) * kWidth * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(st_)),
           "DSpark mask_embd");
    else
        dspark::fill_rows(fp(ds_mask_), ds_x_ + kWidth, kWidth, kDsBlock - 1, cs(st_));
    for (int l = 0; l < kDsLayers; ++l) {
        const DsLayer& L = ds_[(size_t) l];
        rms_norm(ds_x_, fp(L.attn_norm), ds_xa_, kWidth, kDsBlock, ds_eps_, st_);
        lin(L.q, ds_xa_, ds_q_, kDsBlock);
        lin(L.k, ds_xa_, ds_k_, kDsBlock);
        lin(L.v, ds_xa_, ds_v_, kDsBlock);
        head_norm_rope(ds_q_, kHead, fp(L.q_norm), ds_qr_, kHead, kHeads, kHead, ds_theta_, a, kDsBlock, ds_eps_, st_);
        head_norm_rope(ds_k_, kHead, fp(L.k_norm), ds_kr_, kHead, kKv, kHead, ds_theta_, a, kDsBlock, ds_eps_, st_);
        const Cache& c = ds_cache_[(size_t) l];
        const bool ring = ds_swa(l);
        dspark::attend(ds_qr_, ds_kr_, ds_v_, {(int8_t*) c.planes[0].data, (uint16_t*) c.planes[1].data},
                       {(int8_t*) c.planes[2].data, (uint16_t*) c.planes[3].data},
                       ring ? std::max<int64_t>(0, a - kWindow) : 0, a, ring ? kWindow : 0, ds_att_, ds_ao_, cs(st_));
        lin(L.o, ds_ao_, ds_out_, kDsBlock);
        axpy(ds_out_, 1.f, ds_x_, kDsBlock * kWidth, true, st_);
        rms_norm(ds_x_, fp(L.ffn_norm), ds_xa_, kWidth, kDsBlock, ds_eps_, st_);
        lin(L.gate, ds_xa_, ds_g_, kDsBlock);
        lin(L.up, ds_xa_, ds_u_, kDsBlock);
        swiglu_clamp(ds_g_, ds_u_, ds_h_, kDsBlock * kDsFfn, INFINITY, st_);
        lin(L.down, ds_h_, ds_out_, kDsBlock);
        axpy(ds_out_, 1.f, ds_x_, kDsBlock * kWidth, true, st_);
    }
    rms_norm(ds_x_, fp(ds_out_norm_), ds_cur_, kWidth, kDsBlock, ds_eps_, st_);
    ck(cudaMemsetAsync(ds_state_, 0, kDsRank * sizeof(float), cs(st_)), "DSpark");
    ds_based_ = 0;
}

// The head's logits of the block's rows 1 .. to (before the recurrent bias): the first three at once (a round
// mostly asks that many, --draft-max), the rest together when a round asks for a fourth.
void Inkling2::ds_base(int to) {
    if (to <= ds_based_) return;
    const int first = ds_based_, n = (first ? ds_rows_ : std::max(to, std::min(ds_rows_, 3))) - first;
    project_logits(ds_cur_ + (size_t) (first + 1) * kWidth, ds_base_ + (size_t) first * n_vocab_, n);
    ds_based_ = first + n;
}

// Guess i + 1 of the block (Engine::draft()): the recurrent head's step from the token before it, the anchor for the
// first guess and guess i after it; its logits into logits_.
void Inkling2::draft_step(int32_t token, int i) {
    if (df2_) throw std::logic_error("draft: the DFlash2 drafter guesses its block at once");
    if (!ds_head_ready_) prepare_head();
    if (i < 0 || i >= ds_rows_) throw std::logic_error("draft: the DSpark block has no more guesses");
    if (i == 0) ds_block(token);
    float* row = ds_host_ + kWidth;   // past the anchor's embedding, which may still be on its way
    const uint16_t* w1 = (const uint16_t*) ds_w1_ + (size_t) token * kDsRank;
    if (ds_w1_bf16_) {
        for (int j = 0; j < kDsRank; ++j) {
            const uint32_t bits = (uint32_t) w1[j] << 16;
            std::memcpy(&row[j], &bits, 4);
        }
    } else {
        for (int j = 0; j < kDsRank; ++j) row[j] = ggml_fp16_to_fp32(w1[j]);
    }
    ck(cudaMemcpyAsync(ds_w1row_, row, kDsRank * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "DSpark");
    dspark::joint(ds_state_, ds_w1row_, ds_cur_ + (size_t) (i + 1) * kWidth, hn_, ds_jx_, cs(st_));
    lin(ds_joint_, ds_jx_, ds_jo_, 1);
    dspark::recur(ds_jo_, fp(ds_joint_b_), ds_state_, ds_lat_, cs(st_));
    ds_base(i + 1);
    dspark::head(ds_base_ + (size_t) i * n_vocab_, ds_b2_, ds_w2_, ds_lat_, logits_, n_vocab_, ds_cap_, cs(st_));
}

// ---- DFlash2 ----------------------------------------------------------------------------------------------------
namespace {
const TensorInfo* df_enc_norm(const Shards& g) {
    if (const auto* t = g.find("enc.output_norm.weight")) return t;
    return g.find("enc_output_norm.weight");
}
}  // namespace

void Inkling2::df_shape(const Shards& g) {
    namespace d2 = dflash2;
    const std::string file = opt_.draft_model;
    const auto need = [&](bool ok, const std::string& what) {
        if (!ok) throw std::runtime_error(file + " is not the DFlash2 drafter of Inkling-2-Mini: " + what);
    };
    const MetaValue* arch = g.get("general.architecture");
    need(arch && arch->s == "dflash", "its architecture is not 'dflash'");
    const auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = meta_in(g, key);
        need(v != nullptr, std::string("no dflash.") + key);
        return *v;
    };
    const auto is = [&](const char* key, uint64_t value) { need(must(key).u == value, std::string("dflash.") + key); };
    is("block_count", kDfLayers);
    is("embedding_length", kWidth);
    is("feed_forward_length", kDfFfn);
    is("attention.head_count", d2::kHeads);
    is("attention.head_count_kv", d2::kKv);
    is("attention.key_length", kHead);
    is("attention.value_length", kHead);
    is("block_size", d2::kBlock);
    is("attention.sliding_window", d2::kWindow);
    is("selector_top_k", d2::kTop);
    is("selector_rank", d2::kRank);
    is("conv_kernel_size", 2);
    is("conv_group_size", 16);
    const MetaValue* mask_v = meta_in(g, "mask_token_id");
    if (!mask_v) mask_v = g.get("tokenizer.ggml.mask_token_id");
    need(mask_v && mask_v->u == kDfMask, "dflash.mask_token_id / tokenizer.ggml.mask_token_id");
    need(must("selector_softcap_after_correction").u != 0, "its softcap is not after the correction");
    if (const MetaValue* ca = meta_in(g, "attention.causal")) need(ca->u == 0, "it is causal");
    const MetaValue& taps = must("target_visits");
    need(taps.items.size() == 4 && taps.items[0].u == (uint64_t) kTap[0] && taps.items[1].u == (uint64_t) kTap[1] &&
             taps.items[2].u == (uint64_t) kTap[2] && taps.items[3].u == (uint64_t) kVisits,
         "its target_visits are not [5, 23, 41, 48]");
    need(n_vocab_ >= kDfVocab, "the target's vocabulary is smaller than its 200,058 rows");
    df_eps_ = (float) must("attention.layer_norm_rms_epsilon").f;
    df_cap_ = (float) must("final_logit_softcapping").f;
    df_theta_ = std::pow((float) must("rope.freq_base").f, -2.f / kHead);
    need(std::isfinite(df_eps_) && df_eps_ > 0.f && std::isfinite(df_cap_) && df_cap_ > 0.f && df_theta_ > 0.f,
         "its epsilon, logit cap or rope base");
    const TensorInfo* enc = df_enc_norm(g);
    need(enc && !enc->shape.empty() && enc->shape[0] == kWidth && enc->elements() == kWidth,
         "missing or invalid enc.output_norm.weight / enc_output_norm.weight");
    for (const char* name : {"selector_predecessor.weight", "selector_successor.weight"}) {
        const TensorInfo* t = g.find(name);
        need(t && (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0) &&
                 t->shape.size() == 2 && t->shape[0] == d2::kRank && t->shape[1] == kDfVocab,
             std::string(name) + " is not F16, Q8_0 or Q4_0 [256, 200058]");
    }
    df2_ = true;
}

void Inkling2::df_gpu23_init() {
    const auto on = [](const char* k) { const char* v = std::getenv(k); return v && v[0] == '1' && !v[1]; };
    const bool async_ctx = on("STRATA_DFLASH2_ASYNC_CTX");
    if (!on("STRATA_DFLASH2_GPU23")) {
        if (async_ctx) std::fputs("strata-poly: DFlash2 ASYNC_CTX requires GPU23; using ordinary context\n", stderr);
        return;
    }
#if defined(STRATA_HIP_GFX906)
    try {
        if (tp_size_ != 2 || tp_rank_ != 0 || device_ != 0 || opt_.batch != 0)
            throw std::runtime_error("requires one sequence on TP2 owners 0/1");
        // Do not silently requantize the full target head: that would change the draft proposals.
        if (!on("STRATA_DFLASH2_HEAD8") && !on("STRATA_DFLASH2_HEAD_Q4") && !on("STRATA_DFLASH2_HEAD_FUSE"))
            throw std::runtime_error("requires the existing HEAD8, HEAD_Q4 or HEAD_FUSE option");
        int count = 0;
        ck(hipGetDeviceCount(&count), "DFlash2 visible devices");
        if (count != 4) throw std::runtime_error("requires four visible gfx906 GPUs");
        for (int device = 0; device < 4; ++device) {
            hipDeviceProp_t props{};
            ck(hipGetDeviceProperties(&props, device), "DFlash2 GPU properties");
            if (props.warpSize != 64 || !kernels::gfx_arch_is(props.gcnArchName, "gfx906"))
                throw std::runtime_error("requires gfx906 wave64 on all four devices");
        }
        for (auto [from, to] : {std::pair{0, 2}, std::pair{2, 0}, std::pair{2, 3}, std::pair{3, 2}}) {
            DfOnDevice here(from);
            int peer = 0;
            ck(hipDeviceCanAccessPeer(&peer, from, to), "DFlash2 query peer access");
            if (!peer) throw std::runtime_error("missing P2P link " + std::to_string(from) + " -> " + std::to_string(to));
            const auto rc = hipDeviceEnablePeerAccess(to, 0);
            // Mappings may already belong to KV helpers or another engine; never disable them here.
            if (rc == hipErrorPeerAccessAlreadyEnabled) (void) hipGetLastError();
            else ck(rc, "DFlash2 enable peer access");
        }
        auto gpu = std::make_unique<DfGpu23>();
        const auto stream = [](cudaStream_t* s) { ck(hipStreamCreateWithFlags(s, hipStreamNonBlocking), "DFlash2 stream"); };
        const auto event = [](cudaEvent_t* e) { ck(hipEventCreateWithFlags(e, hipEventDisableTiming), "DFlash2 event"); };
        stream(&gpu->tx0);
        for (auto& t : gpu->taps) {
            event(&t.ready0);
            event(&t.pushed);
        }
        {
            DfOnDevice on2(2);
            stream(&gpu->s2);
            for (auto& t : gpu->taps) {
                event(&t.read2);
                event(&t.done2);
            }
            event(&gpu->h2_ready);
        }
        {
            DfOnDevice on3(3);
            stream(&gpu->s3);
            event(&gpu->h3_done);
        }
        gpu->async_ctx = async_ctx;
        df_gpu23_ = std::move(gpu);
        std::fputs("strata-poly: DFlash2 GPU23: backbone/context on GPU2, private head on GPU2/3\n", stderr);
    } catch (const std::runtime_error& e) {
        std::fprintf(stderr, "strata-poly: DFlash2 GPU23 unavailable: %s; using ordinary DFlash2\n", e.what());
    }
#else
    std::fputs("strata-poly: DFlash2 GPU23 unavailable without gfx906; using ordinary DFlash2\n", stderr);
#endif
}

void Inkling2::df_gpu23_scratch() {
    namespace d2 = dflash2;
    auto& gpu = *df_gpu23_;
    const int rows = std::min(rows_, kDfSub);
    gpu.tap_stage0 = (float*) gpu.alloc((size_t) rows * 4 * kWidth * sizeof(float), "DFlash2 tap staging");
    DfOnDevice on2(2);
    df_scratch();
    const size_t ring = (size_t) kDfLayers * d2::kWindow * d2::kKey * sizeof(__half);
    df_kc_ = (__half*) gpu.alloc(ring, "DFlash2 K rings");
    df_vc_ = (__half*) gpu.alloc(ring, "DFlash2 V rings");
    gpu.lin.stream = gpu.s2;
    gpu.lin.q8 = gpu.alloc(std::max(rows, d2::kBlock) * kernels::native_q8_1_bytes(4 * kWidth), "DFlash2 Q8_1 rows");
    if (rows >= kStageRows) {
        // Keep the same prefill lin() dispatch as V6-R1; all its working memory is local to GPU2.
        const auto floats = [](const Dev& w) { return w.type == kTypeF32 || w.type == kTypeBF16; };
        bool blas = floats(df_fc_), mmq = false;
        const auto test_mmq = [&](const Dev& w) {
            return !floats(w) && mmq_ && Mmq::fits((int) w.type, w.n_out, w.n_in);
        };
        mmq |= test_mmq(df_fc_);
        size_t count = df_fc_.type == kTypeBF16 ? (size_t) df_fc_.n_in * df_fc_.n_out : 0;
        for (const auto& layer : df_) for (const Dev& w : {layer.k, layer.v}) {
            blas |= floats(w);
            mmq |= test_mmq(w);
            if (w.type == kTypeBF16) count = std::max(count, (size_t) w.n_in * w.n_out);
        }
        if (blas) gpu.lin.blas = blas_create(gpu.s2);
        if (count) {
            gpu.lin.bf16_max = count;
            gpu.lin.wf = (float*) gpu.alloc(count * sizeof(float), "DFlash2 BF16 conversion scratch");
        }
        if (mmq) {
            gpu.mmq = std::make_unique<Mmq>();
            gpu.lin.mmq = gpu.mmq.get();
            gpu.lin.mqx = gpu.alloc(Mmq::q8_bytes(rows, 4 * kWidth), "DFlash2 MMQ scratch");
            gpu.lin.hf = (float*) gpu.alloc(rows * sizeof(float), "DFlash2 MMQ row scales");
        }
    }
    // These are the width-invariant short paths: FC/K/V bypass GEMM/MMQ, have at most 2304 outputs, and
    // use rowwise F32/BF16, native Q4_0/Q8_0 or V6-R1's planar K|V. Unknown formats still use df_kept().
    const auto prefix_exact = [](const Dev& w) {
        return w.type == kTypeF32 || w.type == kTypeBF16 || w.type == GGML_TYPE_Q4_0 || w.type == GGML_TYPE_Q8_0;
    };
    if (gpu.async_ctx) {
        const char* waves = std::getenv("STRATA_MMVQ_WAVE");
        gpu.async_ctx = !(waves && !std::strcmp(waves, "0")) && prefix_exact(df_fc_);
        for (const auto& layer : df_) gpu.async_ctx &= prefix_exact(layer.k) && prefix_exact(layer.v);
        if (!gpu.async_ctx) {
            std::fputs("strata-poly: DFlash2 ASYNC_CTX needs width-invariant FC/K/V; using kept-only context\n", stderr);
            return;
        }
        {
            DfOnDevice on0(0);
            gpu.prov_taps0 = (float*) gpu.alloc(d2::kBlock * 4 * kWidth * sizeof(float), "DFlash2 provisional source taps");
        }
        gpu.prov_taps2 = (float*) gpu.alloc(d2::kBlock * 4 * kWidth * sizeof(float), "DFlash2 provisional destination taps");
        const size_t bytes = (size_t) kDfLayers * d2::kBlock * d2::kKey * sizeof(__half);
        gpu.prov_k2 = (__half*) gpu.alloc(bytes, "DFlash2 provisional K");
        gpu.prov_v2 = (__half*) gpu.alloc(bytes, "DFlash2 provisional V");
        std::fputs("strata-poly: DFlash2 ASYNC_CTX: prepare decode K/V behind the target head\n", stderr);
    }
}

void Inkling2::df_linear(const Dev& w, const float* x, float* y, int n) {
    if (df_gpu23_) lin_on(df_gpu23_->lin, w, x, y, n);
    else lin(w, x, y, n);
}

void Inkling2::df_scratch() {
    namespace d2 = dflash2;
    const auto f = [&](size_t n) {
        return df_gpu23_ ? (float*) df_gpu23_->alloc(n * sizeof(float), "DFlash2 scratch") : fbuf(n);
    };
    const size_t b = d2::kBlock, c = kDfSub;
    df_t_ = f(2 * kWidth);
    df_x_ = f(b * kWidth);
    df_h_ = f(b * kWidth);
    df_hp_ = f(b * kWidth);
    df_dyn_ = f(b * d2::kDyn);
    df_q_ = f(b * d2::kQuery);
    df_qr_ = f(b * d2::kQuery);
    df_k_ = f(b * d2::kKey);
    df_kr_ = f(b * d2::kKey);
    df_v_ = f(b * d2::kKey);
    df_ao_ = f(b * d2::kQuery);
    df_o_ = f(b * kWidth);
    df_g_ = f(b * kDfFfn);
    df_u_ = f(b * kDfFfn);
    df_m_ = f(b * kDfFfn);
    df_fin_ = f(b * kWidth);
    if (!df_head_fuse_ && !df_gpu23_) df_logits_ = f((b - 1) * (size_t) n_vocab_);
    df_selh_ = f((b - 1) * d2::kRank);
    df_vals_ = f((b - 1) * d2::kTop);
    df_ids_ = (int32_t*) f((b - 1) * d2::kTop);
    df_edges_ = f((b - 1) * d2::kTop * d2::kTop);
    df_att_ = f(d2::attend_floats());
    if (!df_head_fuse_ && !df_gpu23_) df_top_ = f(d2::topk_floats((int) b - 1));
    df_ctx_ = f(c * 4 * kWidth);
    df_c_ = f(c * kWidth);
    df_cn_ = f(c * kWidth);
    df_ck_ = f(c * d2::kKey);
    df_ckr_ = f(c * d2::kKey);
    df_cv_ = f(c * d2::kKey);
    df_xq_ = (int8_t*) f(b * 4 * kWidth / 4);
    df_xd_ = f(b * 4 * kWidth / 32);
}

void Inkling2::df_load_weights() {
    namespace d2 = dflash2;
    const Shards& g = *dg_;
    const auto up = [&](const std::string& name, int64_t n_in, int64_t n_out, Want want) {
        const TensorInfo* t = g.find(name);
        if (!t) throw std::runtime_error(opt_.draft_model + " has no " + name);
        return upload(t, name, n_in, n_out, want);
    };
    for (int l = 0; l < kDfLayers; ++l) {
        DfLayer& L = df_[(size_t) l];
        const auto name = [&](const char* n) { return "blk." + std::to_string(l) + "." + n; };
        L.attn_norm = up(name("attn_norm.weight"), kWidth, 1, kF32);
        L.ffn_norm = up(name("ffn_norm.weight"), kWidth, 1, kF32);
        L.q = up(name("attn_q.weight"), kWidth, d2::kQuery, kAnyMat);
        L.k = up(name("attn_k.weight"), kWidth, d2::kKey, kAnyMat);
        L.v = up(name("attn_v.weight"), kWidth, d2::kKey, kAnyMat);
        L.o = up(name("attn_output.weight"), d2::kQuery, kWidth, kAnyMat);
        L.q_norm = up(name("attn_q_norm.weight"), kHead, 1, kF32);
        L.k_norm = up(name("attn_k_norm.weight"), kHead, 1, kF32);
        L.gate = up(name("ffn_gate.weight"), kWidth, kDfFfn, kAnyMat);
        L.up = up(name("ffn_up.weight"), kWidth, kDfFfn, kAnyMat);
        L.down = up(name("ffn_down.weight"), kDfFfn, kWidth, kAnyMat);
        L.abase = up(name("attn_conv_base"), kWidth, 4, kF32);
        L.aproj = up(name("attn_conv_proj.weight"), kWidth, d2::kDyn, kAnyMat);
        L.fbase = up(name("ffn_conv_base"), kWidth, 4, kF32);
        L.fproj = up(name("ffn_conv_proj.weight"), kWidth, d2::kDyn, kAnyMat);
    }
    df_fc_ = up("fc.weight", 4 * kWidth, kWidth, kAnyMat);
    const TensorInfo* enc = df_enc_norm(g);
    df_hnorm_ = upload(enc, enc->name, kWidth, 1, kF32);
    df_onorm_ = up("output_norm.weight", kWidth, 1, kF32);
    df_selw_ = up("selector_hidden.weight", kWidth, d2::kRank, kAnyMat);
    if (df_gpu23_) {
        const char* name = "token_embd_norm.weight";
        df_token_norm_ = upload(gg_->find(name), name, kWidth, 1, kF32);
    }
}

void Inkling2::df_load() {
    namespace d2 = dflash2;
    const auto on = [](const char* k) { const char* v = std::getenv(k); return v && v[0] == '1' && !v[1]; };
    const bool fill = dense_ != nullptr;
    if (!fill) df_gpu23_init();
    // The stock path keeps using the Engine arena. GPU23 sizes and fills a separate GPU2 arena with the same uploader.
    if (auto* gpu = df_gpu23_.get()) {
        DfOnDevice on2(2);
        DfWeightArena& w = gpu->weights;
        const auto swap = [&] {
            std::swap(w.data, dense_);
            std::swap(w.bytes, dense_bytes_);
            std::swap(w.at, dense_at_);
            std::swap(w.bf16_max, bf16_max_);
        };
        swap();
        void* model_stream = std::exchange(st_, gpu->s2);
        try {
            dense_at_ = 0;
            df_load_weights();
            if (!fill) {
                dense_bytes_ = dense_at_;
                dense_ = (uint8_t*) gpu->alloc(dense_bytes_ + 256, "DFlash2 dense weights");
                zero_vram(dense_, dense_bytes_ + 256);
            }
        } catch (...) {
            swap();
            st_ = model_stream;
            throw;
        }
        swap();
        st_ = model_stream;
    } else {
        df_load_weights();
    }
    if (!fill) return;
    // Continue packing on the same device and stream as the raw uploads.
    const DfOnDevice draft_device(df_gpu23_ ? 2 : device_);
    const auto alloc = [&](size_t bytes, const char* what) {
        return df_gpu23_ ? df_gpu23_->alloc(bytes, what) : dmalloc(bytes, what);
    };
    const Shards& g = *dg_;
    const auto book = [&](const char* name) {
        const TensorInfo& t = *g.find(name);   // validated by df_shape()
        const size_t bytes = g.payload(t);
        void* d = alloc(bytes, "DFlash2 selector codebook");
        ck(cudaMemcpy(d, g.data(t), bytes, cudaMemcpyHostToDevice), "DFlash2 selector codebook");
        const auto kind = t.type == GGML_TYPE_F16 ? d2::BookKind::kF16 :
                          t.type == GGML_TYPE_Q8_0 ? d2::BookKind::kQ8 : d2::BookKind::kQ4;
        return d2::Book{d, kind};
    };
    df_pred_ = book("selector_predecessor.weight");
    df_succ_ = book("selector_successor.weight");
    const bool mixed = on("STRATA_DFLASH2_MIXED");
    df_planar_ = mixed || on("STRATA_DFLASH2_GEMV8");
    df_kept_opt_ = on("STRATA_DFLASH2_KEPT_OPT");
    df_q4_f32_ = on("STRATA_DFLASH2_Q4_F32");
    df_attend3_ = on("STRATA_DFLASH2_ATTEND3");
    bool need_xsum = false;
    if (df_planar_ || df_kept_opt_) {
        for (auto& L : df_) {
            L.qkv = df_pack_rows({L.q, L.k, L.v});
            need_xsum |= L.qkv.q4();
            if (!df_planar_) continue;
            L.gu = df_pack_rows({L.gate, L.up});
            L.wo = df_pack_rows({L.o});
            L.wdown = df_pack_rows({L.down});
            L.waproj = df_pack_rows({L.aproj});
            L.wfproj = df_pack_rows({L.fproj});
            need_xsum |= L.gu.q4() || L.wo.q4() || L.wdown.q4() || L.waproj.q4() || L.wfproj.q4();
        }
        if (df_planar_) {
            df_sel_ = df_pack_rows({df_selw_});
            need_xsum |= df_sel_.q4();
        }
        ck(cudaStreamSynchronize(df_stream()), "DFlash2 planar pack");
    }
    if (need_xsum)
        df_xsum_ = (int32_t*) alloc(d2::kBlock * 4 * kWidth / 32 * sizeof(int32_t), "DFlash2 activation sums");
    if (df_gpu23_ && !df_planar_ && !df_kept_opt_)
        ck(cudaStreamSynchronize(df_stream()), "DFlash2 dense upload");
    df_head_q4_ = tp_size_ == 2 && on("STRATA_DFLASH2_HEAD_Q4");
    df_head_fuse_ = tp_size_ == 2 && on("STRATA_DFLASH2_HEAD_FUSE");
    df_private_head_ = tp_size_ == 2 && (on("STRATA_DFLASH2_HEAD8") || df_head_q4_ || df_head_fuse_);
}

// Homogeneous groups pack from the resident GGML matrices. Mixed/other types retain their ordinary lin() path.
dflash2::Mat Inkling2::df_pack_rows(std::initializer_list<Dev> weights) {
    namespace d2 = dflash2;
    const auto& first = *weights.begin();
    const bool q4 = first.type == GGML_TYPE_Q4_0;
    if (!q4 && first.type != GGML_TYPE_Q8_0) return {};
    int n_out = 0;
    for (const Dev& w : weights) {
        if (w.type != first.type || w.n_in != first.n_in) return {};
        n_out += w.n_out;
    }
    const int n_in = first.n_in;
    const size_t blocks = (size_t) n_out * (n_in / 32);
    d2::Mat mat{nullptr, nullptr, n_in, n_out, q4 ? d2::MatKind::kQ4 : d2::MatKind::kQ8, !q4 || df_q4_f32_};
    const auto alloc = [&](size_t bytes, const char* what) {
        return df_gpu23_ ? df_gpu23_->alloc(bytes, what) : dmalloc(bytes, what);
    };
    auto* q = (uint8_t*) alloc(blocks * mat.code_bytes(), "DFlash2 planar codes");
    auto* d = (uint8_t*) alloc(blocks * mat.scale_bytes(), "DFlash2 planar scales");
    mat.q = q;
    mat.d = d;
    size_t at = 0;
    for (const Dev& w : weights) {
        if (q4)
            d2::pack_q4_0(w.p, n_in, w.n_out, q + at * mat.code_bytes(), d + at * mat.scale_bytes(),
                           mat.scale_f32, df_stream());
        else
            d2::pack_q8_0(w.p, n_in, w.n_out, q + at * mat.code_bytes(), reinterpret_cast<float*>(d) + at, df_stream());
        at += (size_t) w.n_out * (n_in / 32);
    }
    return mat;
}

void Inkling2::df_lin(const Dev& w, const dflash2::Mat& packed, const float* x, float* y, int rows) {
    if (!packed || rows > dflash2::kBlock) return df_linear(w, x, y, rows);
    dflash2::quantize(x, packed.n_in, rows, df_xq_, df_xd_, packed.q4() ? df_xsum_ : nullptr, df_stream());
    dflash2::gemv(packed, df_xq_, df_xd_, df_xsum_, y, rows, df_stream());
}

// Requantize this rank's target head rows on the host, one row at a time. These weights affect guesses only.
void Inkling2::df_head_load(int lo, int n, Inkling2* gpu23_owner) {
    namespace d2 = dflash2;
    const TensorInfo* t = gg_->find("output.weight");
    if (!t) throw std::runtime_error("DFlash2 private head: the target has no output.weight");
    const size_t rb = ggml_row_size((ggml_type) t->type, kWidth);
    gg_->payload(*t);
    const uint8_t* src = gg_->data(*t) + (size_t) lo * rb;
    d2::Mat mat{nullptr, nullptr, kWidth, n, df_head_q4_ ? d2::MatKind::kQ4 : d2::MatKind::kQ8, !df_head_q4_};
    constexpr int blocks_per_row = kWidth / 32;
    const size_t blocks = (size_t) n * blocks_per_row;
    std::vector<uint8_t> q(blocks * mat.code_bytes()), d(blocks * mat.scale_bytes());
    const auto* traits = ggml_get_type_traits((ggml_type) t->type);
    if (!traits || !traits->to_float) throw std::runtime_error("DFlash2 private head: no dequantizer for the head");
    const auto q4_ref = ggml_get_type_traits(GGML_TYPE_Q4_0)->from_float_ref;
    const int workers = 8;
    std::vector<std::thread> pool;
    for (int w = 0; w < workers; ++w)
        pool.emplace_back([&, w] {
            std::vector<float> row(kWidth);
            std::array<uint16_t, blocks_per_row * 9> q4_row;   // aligned GGML Q4_0 blocks: scale then 16 code bytes
            for (int r = w; r < n; r += workers) {
                traits->to_float(src + (size_t) r * rb, row.data(), kWidth);
                if (mat.q4()) {
                    q4_ref(row.data(), q4_row.data(), kWidth);
                    const auto* bytes = reinterpret_cast<const uint8_t*>(q4_row.data());
                    for (int b = 0; b < blocks_per_row; ++b) {
                        const size_t at = (size_t) r * blocks_per_row + b;
                        std::memcpy(q.data() + at * 16, bytes + b * 18 + 2, 16);
                        std::memcpy(d.data() + at * 2, bytes + b * 18, 2);
                    }
                } else {
                    for (int b = 0; b < blocks_per_row; ++b) {
                        float a = 0.f;
                        for (int i = 0; i < 32; ++i) a = std::max(a, std::fabs(row[b * 32 + i]));
                        const float s = a / 127.f;
                        std::memcpy(d.data() + ((size_t) r * blocks_per_row + b) * sizeof(float), &s, sizeof(float));
                        for (int i = 0; i < 32; ++i)
                            q[(size_t) r * kWidth + b * 32 + i] =
                                (int8_t) (s > 0.f ? std::lrint(row[b * 32 + i] / s) : 0);
                    }
                }
            }
        });
    for (auto& th : pool) th.join();
    const auto alloc = [&](size_t bytes, const char* what) {
        return gpu23_owner ? gpu23_owner->df_gpu23_->alloc(bytes, what) : dmalloc(bytes, what);
    };
    auto* dq = (uint8_t*) alloc(q.size(), "DFlash2 private head codes");
    void* dd = alloc(d.size(), "DFlash2 private head scales");
    ck(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice), "DFlash2 private head codes");
    ck(cudaMemcpy(dd, d.data(), d.size(), cudaMemcpyHostToDevice), "DFlash2 private head scales");
    mat.q = dq;
    mat.d = dd;
    df_head_ = mat;
    df_head_lo_ = lo;
    const auto f = [&](size_t count, const char* what) { return (float*) alloc(count * sizeof(float), what); };
    if (df_head_fuse_) df_head_parts_ = alloc(d2::head_top_bytes(n), "DFlash2 fused head top");
    else {
        df_hlog_ = f((size_t) d2::kGuess * n, "DFlash2 head logits");
        df_htop_ = f(d2::topk_floats(d2::kGuess), "DFlash2 head top");
    }
    df_hids_ = (int32_t*) f(d2::kGuess * d2::kTop, "DFlash2 head ids");
    df_hvals_ = f(d2::kGuess * d2::kTop, "DFlash2 head values");
    df_hids1_ = (int32_t*) f(d2::kGuess * d2::kTop, "DFlash2 head ids");
    df_hvals1_ = f(d2::kGuess * d2::kTop, "DFlash2 head values");
    df_hpack_ = f(d2::kGuess * 2 * d2::kTop, "DFlash2 head pack");
    df_hx_ = f(d2::kGuess * kWidth, "DFlash2 head rows");
    df_hxq_ = (int8_t*) f(d2::kGuess * kWidth / 4, "DFlash2 head codes");
    df_hxd_ = f(d2::kGuess * kWidth / 32, "DFlash2 head scales");
    if (df_head_q4_ || df_sel_.q4())
        df_hxsum_ = (int32_t*) alloc(d2::kGuess * kWidth / 32 * sizeof(int32_t), "DFlash2 head activation sums");
}

void Inkling2::df_head_run(const float* h, cudaStream_t stream) {
    namespace d2 = dflash2;
    d2::quantize(h, kWidth, d2::kGuess, df_hxq_, df_hxd_, df_hxsum_, stream);
    if (df_head_fuse_) {
        d2::head_top(df_head_, df_hxq_, df_hxd_, df_hxsum_, df_head_parts_, df_hids_, df_hvals_, stream);
    } else {
        d2::gemv(df_head_, df_hxq_, df_hxd_, df_hxsum_, df_hlog_, d2::kGuess, stream);
        d2::topk(df_hlog_, d2::kGuess, df_head_.n_out, df_head_.n_out, df_htop_, df_hids_, df_hvals_, stream);
    }
}

void Inkling2::df_taps_push(int first, int n, bool provisional) {
    namespace d2 = dflash2;
    auto& gpu = *df_gpu23_;
    const size_t plane = (size_t) rows_ * kWidth;
    const d2::Taps taps{{ds_taps_, ds_taps_ + plane, ds_taps_ + 2 * plane, hn_ + n_embd_}};
    float* src = provisional ? gpu.prov_taps0 : gpu.tap_stage0;
    float* dst = provisional ? gpu.prov_taps2 : df_ctx_;
    auto& ev = gpu.taps[gpu.next_taps];
    gpu.next_taps = (gpu.next_taps + 1) % gpu.taps.size();
    if (ev.used) {
        // Normal decode already synchronized GPU2 for its guesses. Backpressure only if an unusually long series of
        // calls without an ask fills the event ring; wait for this old transfer, not the new context.
        DfOnDevice on2(2);
        const auto status = cudaEventQuery(ev.done2);
        if (status == cudaErrorNotReady) ck(cudaEventSynchronize(ev.done2), "DFlash2 recycle transfer events");
        else ck(status, "DFlash2 check transfer events");
    }
    auto& prev = gpu.last_taps[provisional];
    // The model can reuse the tap planes after gather. Same-buffer reuse waits only for the previous DMA.
    if (prev && prev != &ev) ck(cudaStreamWaitEvent(cs(st_), prev->pushed, 0), "DFlash2 reuse GPU0 staging");
    d2::gather_taps(taps, first, n, src, cs(st_));
    ck(cudaEventRecord(ev.ready0, cs(st_)), "DFlash2 taps ready");
    ck(cudaStreamWaitEvent(gpu.tx0, ev.ready0, 0), "DFlash2 wait for taps");
    if (prev && prev != &ev) ck(cudaStreamWaitEvent(gpu.tx0, prev->read2, 0), "DFlash2 reuse GPU2 staging");
#if defined(STRATA_HIP_GFX906)
    ck(hipMemcpyPeerAsync(dst, 2, src, 0, (size_t) n * 4 * kWidth * sizeof(float), gpu.tx0), "DFlash2 push taps 0->2");
#endif
    ck(cudaEventRecord(ev.pushed, gpu.tx0), "DFlash2 taps pushed");
    DfOnDevice on2(2);
    ck(cudaStreamWaitEvent(gpu.s2, ev.pushed, 0), "DFlash2 wait for pushed taps");
    df_context(dst, pos_ + first, n, provisional, ev.read2);
    ck(cudaEventRecord(ev.done2, gpu.s2), "DFlash2 context done");
    // The other staging pair's last use has completed if we just recycled its event slot.
    if (gpu.last_taps[!provisional] == &ev) gpu.last_taps[!provisional] = nullptr;
    prev = &ev;
    ev.used = true;
}

void Inkling2::on_verify_rows(int n) {
    auto* gpu = df_gpu23_.get();
    if (!gpu || !gpu->async_ctx) return;
    gpu->prov_valid = false;
    if (n < 1 || n > dflash2::kBlock || pos_ < df_skip_) return;
    df_taps_push(0, n, true);
    gpu->prov_pos = pos_;
    gpu->prov_n = n;
    gpu->prov_valid = true;
}

// The kept rows' context: [tap 5, tap 23, tap 41, final norm] -> fc -> hidden_norm, and each layer's K (k_norm, RoPE at
// the row's position) and V into its ring. A prompt's rows that no block will see (df_skip_) are left out.
void Inkling2::df_kept(int n) {
    namespace d2 = dflash2;
    auto* gpu = df_gpu23_.get();
    const bool commit = gpu && std::exchange(gpu->prov_valid, false) && gpu->prov_pos == pos_ && n <= gpu->prov_n;
    int first = (int) std::clamp<int64_t>(df_skip_ - pos_, 0, n);
    first = std::max(first, n - d2::kWindow);
    if (first >= n) return;
    const int64_t at = pos_ + first;
    if (at > df_hi_ || at < df_lo_) df_lo_ = at;   // a gap: the older rows are no longer this sequence's neighbours
    if (commit && first == 0) {
        DfOnDevice on2(2);
        // The same stream orders provisional computation, commit, and the next draft ask. Never wait on the host.
        d2::commit_prov_kv(gpu->prov_k2, gpu->prov_v2, df_kc_, df_vc_, pos_, n, kDfLayers, gpu->s2);
    } else if (gpu) {
        for (int done = first; done < n;) {
            const int m = std::min(kDfSub, n - done);
            df_taps_push(done, m, false);
            done += m;
        }
    } else {
        const size_t plane = (size_t) rows_ * kWidth;
        const d2::Taps taps{{ds_taps_, ds_taps_ + plane, ds_taps_ + 2 * plane, hn_ + n_embd_}};
        for (int done = first; done < n;) {
            const int m = std::min(kDfSub, n - done);
            d2::gather_taps(taps, done, m, df_ctx_, cs(st_));
            df_context(df_ctx_, pos_ + done, m, false);
            done += m;
        }
    }
    df_hi_ = pos_ + n;
}

void Inkling2::df_context(const float* taps, int64_t pos, int m, bool provisional, cudaEvent_t taps_read) {
    namespace d2 = dflash2;
    const auto stream = df_stream();
    const size_t ring = (size_t) d2::kWindow * d2::kKey;
    df_linear(df_fc_, taps, df_c_, m);
    // The packed input is dead after FC (including a possible half-sum rescale).
    if (taps_read) ck(cudaEventRecord(taps_read, stream), "DFlash2 taps consumed");
    rms_norm(df_c_, fp(df_hnorm_), df_cn_, kWidth, m, df_eps_, stream);
    // Use the same planar activation format as the draft block (not lin()'s half-scale Q8_1).
    const bool planar = df_kept_opt_ && m <= d2::kBlock;
    bool quantized = false;
    for (int l = 0; l < kDfLayers; ++l) {
        const DfLayer& L = df_[(size_t) l];
        if (planar && L.qkv) {
            if (!quantized) {
                d2::quantize(df_cn_, kWidth, m, df_xq_, df_xd_, df_xsum_, stream);
                quantized = true;
            }
            const auto kv = L.qkv.slice(d2::kQuery, 2 * d2::kKey);
            d2::gemv_split(kv, df_xq_, df_xd_, df_xsum_, df_ck_, d2::kKey, df_cv_, d2::kKey, nullptr, 0, m, stream);
        } else {
            df_linear(L.k, df_cn_, df_ck_, m);
            df_linear(L.v, df_cn_, df_cv_, m);
        }
        head_norm_rope(df_ck_, kHead, fp(L.k_norm), df_ckr_, kHead, d2::kKv, kHead, df_theta_, pos, m,
                       df_eps_, stream);
        if (provisional) {
            const size_t off = (size_t) l * d2::kBlock * d2::kKey;
            d2::store(df_ckr_, df_cv_, df_gpu23_->prov_k2 + off, df_gpu23_->prov_v2 + off, 0, m, stream);
        } else {
            d2::store(df_ckr_, df_cv_, df_kc_ + l * ring, df_vc_ + l * ring, pos, m, stream);
        }
    }
}

void Inkling2::df_gpu23_head(bool reuse_selector_quant) {
    namespace d2 = dflash2;
    auto& gpu = *df_gpu23_;
    auto& other = *peer_;
    const auto h = df_fin_ + kWidth;
#if defined(STRATA_HIP_GFX906)
    // Use GPU3's stream for both DMAs so both can overlap GPU2's private head, which reads the same immutable rows.
    ck(hipEventRecord(gpu.h2_ready, gpu.s2), "DFlash2 head inputs ready");
    {
        DfOnDevice on3(3);
        ck(hipStreamWaitEvent(gpu.s3, gpu.h2_ready, 0), "DFlash2 head 3 waits for GPU2");
        ck(hipMemcpyPeerAsync(other.df_hx_, 3, h, 2, d2::kGuess * kWidth * sizeof(float), gpu.s3),
           "DFlash2 hidden rows 2->3");
        other.df_head_run(other.df_hx_, gpu.s3);
        d2::pack_top(other.df_hids_, other.df_hvals_, d2::kGuess, other.df_head_lo_, other.df_hpack_, gpu.s3);
        ck(hipMemcpyPeerAsync(df_hpack_, 2, other.df_hpack_, 3, d2::kGuess * 2 * d2::kTop * sizeof(float), gpu.s3),
           "DFlash2 top-16 3->2");
        ck(hipEventRecord(gpu.h3_done, gpu.s3), "DFlash2 head 3 done");
    }
    df_head_run(h, gpu.s2);
    if (reuse_selector_quant)
        d2::gemv(df_sel_, df_hxq_, df_hxd_, df_hxsum_, df_selh_, d2::kGuess, gpu.s2);
    ck(hipStreamWaitEvent(gpu.s2, gpu.h3_done, 0), "DFlash2 wait for GPU3 top-16");
    d2::unpack_top(df_hpack_, d2::kGuess, df_hids1_, df_hvals1_, gpu.s2);
    d2::merge2(df_hids_, df_hvals_, df_hids1_, df_hvals1_, d2::kGuess, df_ids_, df_vals_, gpu.s2);
#else
    (void) gpu; (void) other; (void) h; (void) reuse_selector_quant;
#endif
}

// The block at the anchor `token` (picked at pos(), not fed yet): its embedding and seven masks, each through the
// target's token norm, five layers, the output norm, the target's head on rows 1..7, the top 16 of each, and the
// selector's walk into h_block_.
void Inkling2::drafter_guesses(int32_t token) {
    if (df_gpu23_) {
        DfOnDevice on2(2);
        return df_ask(token);
    }
    df_ask(token);
}

void Inkling2::df_ask(int32_t token) {
    namespace d2 = dflash2;
    const auto stream = df_stream();
    const int64_t a = pos_;
    input(token, a, ds_host_);
    input(kDfMask, a + 1, ds_host_ + kWidth);
    ck(cudaMemcpyAsync(df_t_, ds_host_, 2 * kWidth * sizeof(float), cudaMemcpyHostToDevice, stream), "DFlash2");
    rms_norm(df_t_, fp(df_gpu23_ ? df_token_norm_ : token_norm_), df_x_, kWidth, 2, rms_eps_, stream);
    dspark::fill_rows(df_x_ + kWidth, df_x_ + 2 * kWidth, kWidth, d2::kBlock - 2, stream);
    const int64_t lo = df_hi_ >= a ? df_lo_ : a;
    const size_t ring = (size_t) d2::kWindow * d2::kKey;
    constexpr int b = d2::kBlock;
    for (int l = 0; l < kDfLayers; ++l) {
        const DfLayer& L = df_[(size_t) l];
        if (df_fast_ && df_planar_ && L.waproj && L.qkv) {
            d2::rms_norm_quantize(df_x_, fp(L.attn_norm), df_h_, df_xq_, df_xd_, L.waproj.q4() ? df_xsum_ : nullptr,
                                  kWidth, b, df_eps_, stream);
            d2::gemv(L.waproj, df_xq_, df_xd_, df_xsum_, df_dyn_, b, stream);
            d2::conv_quantize(df_h_, df_dyn_, fp(L.abase), 0, b, df_xq_, df_xd_, L.qkv.q4() ? df_xsum_ : nullptr, stream);
            d2::gemv_split(L.qkv, df_xq_, df_xd_, df_xsum_, df_q_, d2::kQuery, df_k_, d2::kKey, df_v_, d2::kKey,
                           b, stream);
        } else {
            rms_norm(df_x_, fp(L.attn_norm), df_h_, kWidth, b, df_eps_, stream);
            df_lin(L.aproj, L.waproj, df_h_, df_dyn_, b);
            d2::conv(df_h_, df_dyn_, fp(L.abase), 0, df_hp_, b, false, stream);
            if (df_planar_ && L.qkv) {   // one launch for q, k and v
                d2::quantize(df_hp_, kWidth, b, df_xq_, df_xd_, L.qkv.q4() ? df_xsum_ : nullptr, stream);
                d2::gemv_split(L.qkv, df_xq_, df_xd_, df_xsum_, df_q_, d2::kQuery, df_k_, d2::kKey, df_v_, d2::kKey,
                               b, stream);
            } else {
                df_linear(L.q, df_hp_, df_q_, b);
                df_linear(L.k, df_hp_, df_k_, b);
                df_linear(L.v, df_hp_, df_v_, b);
            }
        }
        head_norm_rope(df_q_, kHead, fp(L.q_norm), df_qr_, kHead, d2::kHeads, kHead, df_theta_, a, b, df_eps_, stream);
        head_norm_rope(df_k_, kHead, fp(L.k_norm), df_kr_, kHead, d2::kKv, kHead, df_theta_, a, b, df_eps_, stream);
        (df_attend3_ ? d2::attend3 : d2::attend)(
            df_qr_, df_kr_, df_v_, df_kc_ + l * ring, df_vc_ + l * ring, lo, a, df_att_, df_ao_, stream);
        df_lin(L.o, L.wo, df_ao_, df_o_, b);
        d2::conv(df_o_, df_dyn_, fp(L.abase), 1, df_x_, b, true, stream);
        if (df_fast_ && df_planar_ && L.wfproj && L.gu && L.wdown) {
            d2::rms_norm_quantize(df_x_, fp(L.ffn_norm), df_h_, df_xq_, df_xd_, L.wfproj.q4() ? df_xsum_ : nullptr,
                                  kWidth, b, df_eps_, stream);
            d2::gemv(L.wfproj, df_xq_, df_xd_, df_xsum_, df_dyn_, b, stream);
            d2::conv_quantize(df_h_, df_dyn_, fp(L.fbase), 0, b, df_xq_, df_xd_, L.gu.q4() ? df_xsum_ : nullptr, stream);
            d2::gemv_split(L.gu, df_xq_, df_xd_, df_xsum_, df_g_, kDfFfn, df_u_, kDfFfn, nullptr, 0, b, stream);
            d2::swiglu_quantize(df_g_, df_u_, kDfFfn, b, df_xq_, df_xd_, L.wdown.q4() ? df_xsum_ : nullptr, stream);
            d2::gemv(L.wdown, df_xq_, df_xd_, df_xsum_, df_o_, b, stream);
        } else {
            rms_norm(df_x_, fp(L.ffn_norm), df_h_, kWidth, b, df_eps_, stream);
            df_lin(L.fproj, L.wfproj, df_h_, df_dyn_, b);
            d2::conv(df_h_, df_dyn_, fp(L.fbase), 0, df_hp_, b, false, stream);
            if (df_planar_ && L.gu) {
                d2::quantize(df_hp_, kWidth, b, df_xq_, df_xd_, L.gu.q4() ? df_xsum_ : nullptr, stream);
                d2::gemv_split(L.gu, df_xq_, df_xd_, df_xsum_, df_g_, kDfFfn, df_u_, kDfFfn, nullptr, 0, b, stream);
            } else {
                df_linear(L.gate, df_hp_, df_g_, b);
                df_linear(L.up, df_hp_, df_u_, b);
            }
            swiglu_clamp(df_g_, df_u_, df_m_, b * kDfFfn, INFINITY, stream);
            df_lin(L.down, L.wdown, df_m_, df_o_, b);
        }
        d2::conv(df_o_, df_dyn_, fp(L.fbase), 1, df_x_, b, true, stream);
    }
    rms_norm(df_x_, fp(df_onorm_), df_fin_, kWidth, b, df_eps_, stream);
    const int guesses = std::min(d2::kGuess, opt_.window - 1);
    const bool reuse_selector_quant = df_fast_ && df_private_head_ && df_sel_;
    if (df_gpu23_) {
        df_gpu23_head(reuse_selector_quant);
    } else if (df_private_head_) {   // both owners' private halves run at once; rank 1 returns its top 16 only
        peer_dispatch(PeerCmd::kDraftHead, d2::kGuess);
        try {
            tp2_->broadcast_from_0(0, df_fin_ + kWidth, d2::kGuess * kWidth, stream);
            df_head_run(df_fin_ + kWidth, stream);
            if (reuse_selector_quant)
                d2::gemv(df_sel_, df_hxq_, df_hxd_, df_hxsum_, df_selh_, d2::kGuess, stream);
            tp2_->gather_from_1(0, df_hpack_, nullptr, d2::kGuess * 2 * d2::kTop, stream);
            d2::unpack_top(df_hpack_, d2::kGuess, df_hids1_, df_hvals1_, stream);
            d2::merge2(df_hids_, df_hvals_, df_hids1_, df_hvals1_, d2::kGuess, df_ids_, df_vals_, stream);
        } catch (...) {
            tp2_->abort_barrier();
            peer_wait();
            throw;
        }
        peer_wait();
    } else {
        project_logits(df_fin_ + kWidth, df_logits_, d2::kGuess);
        d2::topk(df_logits_, d2::kGuess, n_vocab_, kDfVocab, df_top_, df_ids_, df_vals_, stream);
    }
    if (!reuse_selector_quant)
        df_lin(df_selw_, df_sel_, df_fin_ + kWidth, df_selh_, d2::kGuess);
    static_assert(sizeof(d2::Guess) == sizeof(Likeliest));
    d2::select(df_ids_, df_vals_, df_selh_, df_pred_, df_succ_, token, df_cap_, df_edges_,
               reinterpret_cast<d2::Guess*>(h_block_), stream);
    ck(cudaStreamSynchronize(stream), "DFlash2 guesses");
    block_n_ = guesses;
}

void Inkling2::output_rows(int first, int rows, float* h) {
    depth_mix(fp(output_query_), cur_, first, rows);
    rms_norm(cur_, fp(out_norm_), h, kWidth, rows, rms_eps_, st_);
}

void Inkling2::finish_logits(float* logits, int rows) { softcap(logits, (size_t) rows * n_vocab_, cap_, st_); }

#if defined(STRATA_HIP_GFX906)
namespace {
struct StageBehind {
    std::thread t;
    std::exception_ptr err;
    void wait() {
        if (t.joinable()) t.join();
        if (err) {
            std::exception_ptr e = err;
            err = nullptr;
            std::rethrow_exception(e);
        }
    }
    template<class F>
    void go(F&& fn) {
        wait();
        t = std::thread([this, f = std::forward<F>(fn)]() mutable {
            try {
                f();
            } catch (...) {
                err = std::current_exception();
            }
        });
    }
    ~StageBehind() {
        if (t.joinable()) t.join();
    }
};
}  // namespace

void Inkling2::destroy_phase_dualtp_handles() {
    int prev = 0;
    const bool ok = (cudaGetDevice(&prev) == cudaSuccess);
    if (ok && cudaSetDevice(device_) == cudaSuccess) {
        if (cut_st_) { (void) cudaStreamDestroy(cut_st_); cut_st_ = nullptr; }
        if (cut_ev_) { (void) cudaEventDestroy(cut_ev_); cut_ev_ = nullptr; }
        if (kv_ret_st_) { (void) cudaStreamDestroy(kv_ret_st_); kv_ret_st_ = nullptr; }
        if (kv_ret_ev_) { (void) cudaEventDestroy(kv_ret_ev_); kv_ret_ev_ = nullptr; }
        if (inbox_free_ev_) { (void) cudaEventDestroy(inbox_free_ev_); inbox_free_ev_ = nullptr; }
        (void) cudaSetDevice(prev);
    }
}

void Inkling2::phase_dualtp_init(const Options& o) {
    const char* env = std::getenv("STRATA_INKLING_PHASE_DUALTP");
    if (!env || !*env || std::strcmp(env, "0") == 0) return;
    char* end = nullptr;
    const long cut = std::strtol(env, &end, 10);
    if (*end || cut != 24)
        throw std::runtime_error("STRATA_INKLING_PHASE_DUALTP supports only cut 24");
    if (const char* mc = std::getenv("STRATA_INKLING_PHASE_DUALTP_MIN_CALLS"); mc && *mc) {
        const int v = std::atoi(mc);
        if (v >= 1) phase_dualtp_min_calls_ = v;
    }
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "PhaseDualTP device count");
    if (devices < 4 || o.kv_grow || o.dump || o.batch || !o.layer_split.empty() || o.kv_format != KvFmt::kQ8_0)
        throw std::runtime_error("STRATA_INKLING_PHASE_DUALTP=24 requires 4 GPUs, Q8_0 --kv-whole, single sequence");
    for (int d = 0; d < 4; ++d) {
        cudaDeviceProp p{};
        ck(cudaGetDeviceProperties(&p, d), "PhaseDualTP device props");
        if (p.warpSize != 64 || !kernels::gfx_arch_is(p.gcnArchName, "gfx906"))
            throw std::runtime_error("STRATA_INKLING_PHASE_DUALTP=24 requires 4 gfx906 wave64 GPUs");
    }
    if (!all_cached() || !peer_ || !peer_->all_cached())
        throw std::runtime_error("STRATA_INKLING_PHASE_DUALTP=24 requires fully resident canonical TP2");

    struct DevGuard { int prev = 0; ~DevGuard() { (void) cudaSetDevice(prev); } } guard;
    ck(cudaGetDevice(&guard.prev), "PhaseDualTP getDevice");
    const auto enable_p2p = [](int src, int dst) {
        ck(cudaSetDevice(src), "PhaseDualTP setDevice P2P");
        int can = 0;
        ck(cudaDeviceCanAccessPeer(&can, src, dst), "PhaseDualTP canAccessPeer");
        if (!can) throw std::runtime_error("PhaseDualTP requires bidirectional P2P between 0<->2 and 1<->3");
        cudaError_t st = cudaDeviceEnablePeerAccess(dst, 0);
        if (st != cudaErrorPeerAccessAlreadyEnabled) ck(st, "PhaseDualTP enablePeerAccess");
        (void) cudaGetLastError();
    };
    enable_p2p(0, 2); enable_p2p(2, 0);
    enable_p2p(1, 3); enable_p2p(3, 1);

    const int max_floats = std::max(o.prompt_batch, 8) * kWidth;
    auto comm_b = Tp2Comm::create(2, 3, max_floats);
    auto b0 = std::make_unique<Inkling2>();
    auto b1 = std::make_unique<Inkling2>();
    for (int r = 0; r < 2; ++r) {
        Inkling2* b = r == 0 ? b0.get() : b1.get();
        b->device_ = 2 + r;
        b->split_ = true;
        b->tp_size_ = 2;
        b->tp_rank_ = r;
        b->tp2_ = comm_b;
        b->first_visit_ = (int) cut;
        b->last_visit_ = kVisits;
        b->run_last_visit_ = kVisits;
        b->lo_ = kStart;
        b->end_ = -1;
        b->prefill_only_ = true;
        b->swa_ring_win_ = opt_.window;
        b->compact_swa_asked_ = compact_swa_asked_;
        b->prefill_depth_share_ = prefill_depth_share_;
    }
    b0->stage_b_want_df2_ = (df2_ && df_gpu23_ != nullptr);

    Options ob = o;
    ob.tp = 2;
    ob.draft = false;
    ob.draft_model.clear();
    ob.window = 1;
    ob.vram_reserve_mib = std::min(o.vram_reserve_mib, 512);

    ck(cudaSetDevice(2), "PhaseDualTP setDevice 2");
    b0->load(ob);
    ck(cudaSetDevice(3), "PhaseDualTP setDevice 3");
    b1->load(ob);

    if (gpu_prefill_route_) {
        if (!b0->gpu_route_supported() || !b1->gpu_route_supported())
            throw std::runtime_error("PhaseDualTP Stage B failed gpu_route_supported()");
        ck(cudaSetDevice(3), "PhaseDualTP route dev 3");
        b1->init_gpu_route(false, true);
        ck(cudaSetDevice(2), "PhaseDualTP route dev 2");
        b0->init_gpu_route(false, true);
    }
    if (!b0->all_cached() || !b1->all_cached() || b0->ring_ != ring_ || b1->ring_ != peer_->ring_ ||
        b0->rows_ != rows_ || b1->rows_ != peer_->rows_)
        throw std::runtime_error("PhaseDualTP Stage B geometry or residency mismatch");
    // Verify identical contiguous SWA ring layout for visits [24, 48) so sync is one P2P copy per rank.
    for (int r = 0; r < 2; ++r) {
        const Inkling2* a = r == 0 ? this : peer_.get();
        const Inkling2* b = r == 0 ? b0.get() : b1.get();
        const uint8_t* a_base = a->caches_[24].planes[0].data;
        const uint8_t* b_base = b->caches_[24].planes[0].data;
        for (int v = 24; v < kVisits; ++v) if (local(v)) {
            for (int j = 0; j < 4; ++j) if (a->caches_[v].planes[j].stride) {
                if ((a->caches_[v].planes[j].data - a_base) != (b->caches_[v].planes[j].data - b_base))
                    throw std::runtime_error("PhaseDualTP Stage B SWA ring layout mismatch");
            }
        }
    }
    // Create dedicated async P2P streams and events on all 4 cards.
    for (int r = 0; r < 2; ++r) {
        Inkling2* a = r == 0 ? this : peer_.get();
        Inkling2* b = r == 0 ? b0.get() : b1.get();
        ck(cudaSetDevice(a->device_), "PhaseDualTP stream dev A");
        ck(cudaStreamCreateWithFlags(&a->cut_st_, cudaStreamNonBlocking), "PhaseDualTP cut_st");
        ck(cudaEventCreateWithFlags(&a->cut_ev_, cudaEventDisableTiming), "PhaseDualTP cut_ev");
        ck(cudaSetDevice(b->device_), "PhaseDualTP stream dev B");
        ck(cudaStreamCreateWithFlags(&b->kv_ret_st_, cudaStreamNonBlocking), "PhaseDualTP kv_ret_st");
        ck(cudaEventCreateWithFlags(&b->kv_ret_ev_, cudaEventDisableTiming), "PhaseDualTP kv_ret_ev");
        ck(cudaEventCreateWithFlags(&b->inbox_free_ev_, cudaEventDisableTiming), "PhaseDualTP inbox_free_ev");
    }

    ck(cudaSetDevice(2), "PhaseDualTP peer thread dev 2");
    b0->peer_ = std::move(b1);
    b0->start_peer_thread();
    size_t f2 = 0, t2 = 0, f3 = 0, t3 = 0;
    ck(cudaSetDevice(2), "PhaseDualTP meminfo 2"); ck(cudaMemGetInfo(&f2, &t2), "PhaseDualTP meminfo 2");
    ck(cudaSetDevice(3), "PhaseDualTP meminfo 3"); ck(cudaMemGetInfo(&f3, &t3), "PhaseDualTP meminfo 3");
    std::fprintf(stderr,
                 "strata-poly: STRATA_INKLING_PHASE_DUALTP=24: co-resident Stage B on GPUs 2/3 "
                 "(blocks 6..23, visits [24,48), %d experts/rank, staggered cut + overlapped KV return; "
                 "GPU2 free=%zu MiB, GPU3 free=%zu MiB)\n",
                 b0->cached_, f2 >> 20, f3 >> 20);
    stage_b_ = std::move(b0);
    stage_b_kv_valid_ = 0;
}

void Inkling2::phase_dualtp_sync_to_b(int64_t pos) {
    struct DevGuard { int prev = 0; ~DevGuard() { (void) cudaSetDevice(prev); } } guard;
    ck(cudaGetDevice(&guard.prev), "PhaseDualTP sync_to_b getDevice");
    Inkling2* src[2] = {this, peer_.get()};
    Inkling2* dst[2] = {stage_b_.get(), stage_b_->peer_.get()};
    if (pos == 0) {
        const size_t conv_bytes = (size_t) (kVisits - 24) * conv_floats_ * sizeof(float);
        for (int r = 0; r < 2; ++r) {
            ck(cudaSetDevice(dst[r]->device_), "PhaseDualTP zero B state");
            ck(cudaMemsetAsync(dst[r]->state_, 0, conv_bytes, cs(dst[r]->st_)), "PhaseDualTP zero B ShortConv");
        }
        stage_b_->pos_ = stage_b_->peer_->pos_ = 0;
        stage_b_kv_valid_ = 0;
        return;
    }
    const size_t swa_slab_bytes = (size_t) stage_b_->owned_swa() * ring_ * kv_bytes();
    const size_t conv_bytes = (size_t) (kVisits - 24) * conv_floats_ * sizeof(float);
    for (int r = 0; r < 2; ++r) {
        Inkling2& a = *src[r];
        Inkling2& b = *dst[r];
        ck(cudaSetDevice(a.device_), "PhaseDualTP sync_to_b dev");
        ck(cudaStreamSynchronize(cs(a.st_)), "PhaseDualTP sync_to_b drain A");
        // Sync missing full KV rows [stage_b_kv_valid_, pos) for Stage B's 4 full visits.
        if (stage_b_kv_valid_ < pos) {
            const int64_t rows = pos - stage_b_kv_valid_;
            for (int v = 24; v < kVisits; ++v) if (!local(v)) {
                for (int j = 0; j < 4; ++j) if (a.caches_[v].planes[j].stride) {
                    const size_t stride = a.caches_[v].planes[j].stride;
                    ck(cudaMemcpyPeerAsync(b.caches_[v].planes[j].data + stage_b_kv_valid_ * stride, b.device_,
                                           a.caches_[v].planes[j].data + stage_b_kv_valid_ * stride, a.device_,
                                           (size_t) rows * stride, a.cut_st_),
                       "PhaseDualTP sync KV to B");
                }
            }
        }
        // Sync all 20 SWA rings in one contiguous P2P copy + 24 ShortConv states.
        ck(cudaMemcpyPeerAsync(b.caches_[24].planes[0].data, b.device_,
                               a.caches_[24].planes[0].data, a.device_, swa_slab_bytes, a.cut_st_),
           "PhaseDualTP sync SWA to B");
        ck(cudaMemcpyPeerAsync(b.state_, b.device_,
                               a.state_ + (size_t) 24 * conv_floats_, a.device_, conv_bytes, a.cut_st_),
           "PhaseDualTP sync ShortConv to B");
    }
    for (int r = 0; r < 2; ++r) {
        ck(cudaSetDevice(src[r]->device_), "PhaseDualTP sync_to_b wait");
        ck(cudaStreamSynchronize(src[r]->cut_st_), "PhaseDualTP sync_to_b wait");
    }
    stage_b_->pos_ = stage_b_->peer_->pos_ = pos;
    stage_b_kv_valid_ = pos;
}

void Inkling2::phase_dualtp_handoff_cut(int n, bool send_df_taps, int df_first) {
    struct DevGuard { int prev = 0; ~DevGuard() { (void) cudaSetDevice(prev); } } guard;
    ck(cudaGetDevice(&guard.prev), "PhaseDualTP handoff getDevice");
    Inkling2* src[2] = {this, peer_.get()};
    Inkling2* dst[2] = {stage_b_.get(), stage_b_->peer_.get()};
    const size_t bytes = (size_t) n * kWidth * sizeof(float);

    // Ensure Stage B's previous chunk finished reading its live buffers.
    for (int r = 0; r < 2; ++r) {
        ck(cudaSetDevice(dst[r]->device_), "PhaseDualTP handoff drain B");
        ck(cudaStreamSynchronize(cs(dst[r]->st_)), "PhaseDualTP handoff drain B");
    }
    // Wait for Stage A's staggered cut_st_ (already done behind visits 1..23) and send only r_ (18 MiB) + optional taps.
    for (int r = 0; r < 2; ++r) {
        Inkling2& a = *src[r];
        Inkling2& b = *dst[r];
        ck(cudaSetDevice(a.device_), "PhaseDualTP handoff send r_");
        ck(cudaMemcpyPeerAsync(b.r_, b.device_, a.r_, a.device_, bytes, cs(a.st_)), "PhaseDualTP send r_");
        if (r == 0 && send_df_taps && b.sb_tap0_ && b.sb_tap1_) {
            const size_t off = (size_t) df_first * kWidth;
            const size_t tbytes = (size_t) (n - df_first) * kWidth * sizeof(float);
            const size_t plane = (size_t) rows_ * kWidth;
            ck(cudaMemcpyPeerAsync(b.sb_tap0_ + off, b.device_, a.ds_taps_ + off, a.device_, tbytes, cs(a.st_)),
               "PhaseDualTP send tap0");
            ck(cudaMemcpyPeerAsync(b.sb_tap1_ + off, b.device_, a.ds_taps_ + plane + off, a.device_, tbytes, cs(a.st_)),
               "PhaseDualTP send tap1");
        }
    }
    for (int r = 0; r < 2; ++r) {
        ck(cudaSetDevice(src[r]->device_), "PhaseDualTP handoff sync A");
        ck(cudaStreamSynchronize(src[r]->cut_st_), "PhaseDualTP handoff sync cut_st");
        ck(cudaStreamSynchronize(cs(src[r]->st_)), "PhaseDualTP handoff sync A st");
    }
    // Materialize Stage B's live buffers from its inbox (~0.08 ms D2D on HBM2); depth_[1] == prelude_ (0 PCIe bytes!).
    for (int r = 0; r < 2; ++r) {
        Inkling2& a = *src[r];
        Inkling2& b = *dst[r];
        ck(cudaSetDevice(b.device_), "PhaseDualTP handoff unpack B");
        ck(cudaMemcpyAsync(b.depth_[0], b.cut_depth0_, bytes, cudaMemcpyDeviceToDevice, cs(b.st_)), "PhaseDualTP d0");
        ck(cudaMemcpyAsync(b.prelude_, b.cut_prelude_, bytes, cudaMemcpyDeviceToDevice, cs(b.st_)), "PhaseDualTP pre");
        ck(cudaMemcpyAsync(b.depth_[1], b.cut_prelude_, bytes, cudaMemcpyDeviceToDevice, cs(b.st_)), "PhaseDualTP d1");
        ck(cudaMemcpyAsync(b.bank0_, b.cut_bank0_, bytes, cudaMemcpyDeviceToDevice, cs(b.st_)), "PhaseDualTP b0");
        ck(cudaMemcpyAsync(b.bank1_, b.cut_bank1_, bytes, cudaMemcpyDeviceToDevice, cs(b.st_)), "PhaseDualTP b1");
        b.depth_n_ = 2;
        ck(cudaEventRecord(b.inbox_free_ev_, cs(b.st_)), "PhaseDualTP inbox_free");
        ck(cudaSetDevice(a.device_), "PhaseDualTP handoff wait inbox_free");
        ck(cudaStreamWaitEvent(a.cut_st_, b.inbox_free_ev_, 0), "PhaseDualTP cut_st wait inbox_free");
    }
}

void Inkling2::phase_dualtp_save_range(int first_v, int last_v, int64_t pos, uint8_t* dst,
                                       size_t r0_state_n, size_t r1_state_n) {
    struct DevGuard { int prev = 0; ~DevGuard() { (void) cudaSetDevice(prev); } } guard;
    ck(cudaGetDevice(&guard.prev), "PhaseDualTP save_range getDevice");
    const size_t one = (size_t) std::min(opt_.max_context, kWindow - 1) * kv_bytes();
    const size_t r0_swa_total = (kVisits - kVisits / kBlock) * one;
    uint8_t* dst_rank[2] = {dst, dst + r0_state_n * sizeof(float) + r0_swa_total};
    const size_t state_n_rank[2] = {r0_state_n, r1_state_n};
    Inkling2* ranks[2] = {this, peer_.get()};
    for (int r = 0; r < 2; ++r) {
        Inkling2& eng = *ranks[r];
        ck(cudaSetDevice(eng.device_), "PhaseDualTP save_range dev");
        ck(cudaStreamSynchronize(cs(eng.st_)), "PhaseDualTP save_range drain");
        for (int v = first_v; v < last_v; ++v) {
            const size_t dst_conv = (size_t) v * conv_floats_;
            const size_t src_conv = eng.conv_at(v);
            ck(cudaMemcpy(dst_rank[r] + dst_conv * sizeof(float), eng.state_ + src_conv,
                          conv_floats_ * sizeof(float), cudaMemcpyDeviceToHost),
               "PhaseDualTP save ShortConv");
            if (local(v)) {
                const size_t off = (size_t) (v - v / kBlock) * one;
                eng.ring_state(v, pos, nullptr, dst_rank[r] + state_n_rank[r] * sizeof(float) + off);
            }
        }
        if (r == 0 && first_v == 0 && draft_) save_last(dst);
    }
}

void Inkling2::phase_dualtp_df_kept(int64_t chunk_pos, int m, int first) {
    namespace d2 = dflash2;
    auto* gpu = df_gpu23_.get();
    if (!gpu || !stage_b_ || !stage_b_->sb_tap0_) return;
    gpu->prov_valid = false;
    // Drain any prior decode-turn tap transfers before reusing df_ctx_.
    {
        DfOnDevice on2(2);
        for (auto& ev : gpu->taps) {
            if (ev.used) {
                ck(cudaEventSynchronize(ev.done2), "PhaseDualTP drain df_taps");
                ev.used = false;
            }
        }
        gpu->last_taps[0] = gpu->last_taps[1] = nullptr;
        const int64_t at = chunk_pos + first;
        if (at > df_hi_ || at < df_lo_) df_lo_ = at;
        ck(cudaStreamSynchronize(cs(stage_b_->st_)), "PhaseDualTP Stage B taps ready");
        const d2::Taps taps{{stage_b_->sb_tap0_, stage_b_->sb_tap1_, stage_b_->sb_tap2_, stage_b_->sb_hn_}};
        for (int done = first; done < m;) {
            const int sub = std::min(kDfSub, m - done);
            d2::gather_taps(taps, done, sub, df_ctx_, gpu->s2);
            df_context(df_ctx_, chunk_pos + done, sub, false);
            done += sub;
        }
        ck(cudaStreamSynchronize(gpu->s2), "PhaseDualTP df_context done");
        df_hi_ = chunk_pos + m;
    }
}

void Inkling2::phase_dualtp_run_b(int64_t chunk_pos, int n, bool do_df_taps, int df_first, uint8_t* checkpoint,
                                  size_t r0_state_n, size_t r1_state_n) {
    Inkling2& b0 = *stage_b_;
    Inkling2& b1 = *b0.peer_;
    ck(cudaSetDevice(b0.device_), "PhaseDualTP run_b dev 2");
    b0.cover(chunk_pos + n);
    b0.pos_ = chunk_pos;
    b0.tok_ = nullptr;
    b0.windowed_ = false;
    b0.snap_bits_ = 0;
    b0.sb_capture_df2_ = do_df_taps;
    b0.run_visits(n);
    if (do_df_taps && b0.sb_hn_) b0.output_rows(0, n, b0.sb_hn_);
    // Async overlapped KV delta return: 4 full visits x n rows (<=8.5 MiB/rank) back to canonical GPU0/1!
    Inkling2* src[2] = {&b0, &b1};
    Inkling2* dst[2] = {this, peer_.get()};
    for (int r = 0; r < 2; ++r) {
        Inkling2& s = *src[r];
        Inkling2& d = *dst[r];
        ck(cudaSetDevice(s.device_), "PhaseDualTP kv_ret dev");
        ck(cudaEventRecord(s.kv_ret_ev_, cs(s.st_)), "PhaseDualTP kv_ret record");
        ck(cudaStreamWaitEvent(s.kv_ret_st_, s.kv_ret_ev_, 0), "PhaseDualTP kv_ret wait");
        for (int v = 24; v < kVisits; ++v) if (!local(v)) {
            for (int j = 0; j < 4; ++j) if (s.caches_[v].planes[j].stride) {
                const size_t stride = s.caches_[v].planes[j].stride;
                ck(cudaMemcpyPeerAsync(d.caches_[v].planes[j].data + chunk_pos * stride, d.device_,
                                       s.caches_[v].planes[j].data + chunk_pos * stride, s.device_,
                                       (size_t) n * stride, s.kv_ret_st_),
                   "PhaseDualTP kv_ret P2P");
            }
        }
    }
    b0.pos_ = b1.pos_ = chunk_pos + n;
    if (do_df_taps) phase_dualtp_df_kept(chunk_pos, n, df_first);
    if (checkpoint) b0.phase_dualtp_save_range(24, kVisits, chunk_pos + n, checkpoint, r0_state_n, r1_state_n);
}

void Inkling2::phase_dualtp_sync_from_b(int64_t final_pos) {
    struct DevGuard { int prev = 0; ~DevGuard() { (void) cudaSetDevice(prev); } } guard;
    ck(cudaGetDevice(&guard.prev), "PhaseDualTP sync_from_b getDevice");
    Inkling2* src[2] = {stage_b_.get(), stage_b_->peer_.get()};
    Inkling2* dst[2] = {this, peer_.get()};
    const size_t swa_slab_bytes = (size_t) stage_b_->owned_swa() * ring_ * kv_bytes();
    const size_t conv_bytes = (size_t) (kVisits - 24) * conv_floats_ * sizeof(float);
    for (int r = 0; r < 2; ++r) {
        Inkling2& s = *src[r];
        Inkling2& d = *dst[r];
        ck(cudaSetDevice(s.device_), "PhaseDualTP sync_from_b dev");
        ck(cudaStreamSynchronize(cs(s.st_)), "PhaseDualTP sync_from_b drain st");
        // Enqueue final SWA rings + ShortConv right behind the final chunk's KV delta on kv_ret_st_.
        ck(cudaMemcpyPeerAsync(d.caches_[24].planes[0].data, d.device_,
                               s.caches_[24].planes[0].data, s.device_, swa_slab_bytes, s.kv_ret_st_),
           "PhaseDualTP return SWA");
        ck(cudaMemcpyPeerAsync(d.state_ + (size_t) 24 * conv_floats_, d.device_,
                               s.state_, s.device_, conv_bytes, s.kv_ret_st_),
           "PhaseDualTP return ShortConv");
    }
    for (int r = 0; r < 2; ++r) {
        ck(cudaSetDevice(src[r]->device_), "PhaseDualTP sync_from_b wait");
        ck(cudaStreamSynchronize(src[r]->kv_ret_st_), "PhaseDualTP sync_from_b wait");
    }
    stage_b_kv_valid_ = final_pos;
}

bool Inkling2::read_custom(const int32_t* tokens, int n, const Mark* marks, int n_marks,
                           const std::function<bool(int, int)>* turn, bool& completed) {
    const int n_calls = (int) calls_.size();
    if (!stage_b_ || !peer_ || n_calls < phase_dualtp_min_calls_) return false;
    (void) n_marks;
    const int64_t pos0 = pos_;
    const size_t r0_state_n = state_n_, r1_state_n = peer_->state_n_;
    const int64_t sb_exp_before = stage_b_->stats_.gpu_experts + stage_b_->peer_->stats_.gpu_experts;

    phase_dualtp_sync_to_b(pos0);

    struct PhaseGuard {
        Inkling2* r0;
        Inkling2* r1;
        ~PhaseGuard() {
            r0->run_last_visit_ = kVisits;
            r1->run_last_visit_ = kVisits;
            r0->phase_dualtp_active_ = false;
            r1->phase_dualtp_active_ = false;
            r0->cut_dst_ = nullptr;
            r1->cut_dst_ = nullptr;
            (void) cudaSetDevice(r0->device_);
        }
    } guard{this, peer_.get()};

    run_last_visit_ = 24;
    peer_->run_last_visit_ = 24;
    phase_dualtp_active_ = true;
    peer_->phase_dualtp_active_ = true;
    cut_dst_ = stage_b_.get();
    peer_->cut_dst_ = stage_b_->peer_.get();

    StageBehind behind;
    bool cancelled = false;
    int64_t completed_pos = pos0;
    for (int c = 0; c < n_calls; ++c) {
        const Call& call = calls_[(size_t) c];
        const int64_t chunk_pos = pos0 + call.at;
        uint8_t* const checkpoint = call.mark < 0 ? nullptr : marks[call.mark].state;
        const int df_first = (draft_ && df2_ && df_gpu23_)
                                 ? (int) std::max<int64_t>(std::clamp<int64_t>(df_skip_ - chunk_pos, 0, call.m),
                                                           call.m - dflash2::kWindow)
                                 : call.m;
        const bool do_df = (draft_ && df2_ && df_gpu23_ && df_first < call.m);

        ck(cudaSetDevice(device_), "PhaseDualTP Stage A dev");
        run(tokens + call.at, call.m);
        behind.wait();
        phase_dualtp_handoff_cut(call.m, do_df, df_first);
        behind.go([this, chunk_pos, m = call.m, do_df, df_first, checkpoint, r0_state_n, r1_state_n] {
            phase_dualtp_run_b(chunk_pos, m, do_df, df_first, checkpoint, r0_state_n, r1_state_n);
        });
        pos_ = chunk_pos + call.m;
        peer_->pos_ = pos_;
        stats_.tokens += call.m;
        completed_pos = pos_;
        if (checkpoint) phase_dualtp_save_range(0, 24, pos_, checkpoint, r0_state_n, r1_state_n);
        if (turn && c + 1 < n_calls && !(*turn)(c + 1, n_calls)) {
            cancelled = true;
            break;
        }
    }
    behind.wait();
    phase_dualtp_sync_from_b(completed_pos);
    const int64_t sb_exp_after = stage_b_->stats_.gpu_experts + stage_b_->peer_->stats_.gpu_experts;
    stats_.gpu_experts += (sb_exp_after - sb_exp_before);
    completed = !cancelled;
    return true;
}
#endif

}  // namespace strata::poly
