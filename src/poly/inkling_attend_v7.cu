// src/poly/inkling_attend_v7.cu - Exact V7-R3 Flash attention variants (0-ULP bitwise identical to flash_lean_kernel).
// Kept in its own translation unit so compiler register allocation for flash_lean_kernel is never perturbed.
#include "inkling_attend_lean.hpp"
#include "inkling_flash_common.cuh"
#include "lean_lane_exchange_gfx906.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strata::poly::inkling {
namespace {

template <bool Q8, int X> __device__ __forceinline__ float v7_add(float v) {
#if defined(__gfx906__)
    if constexpr (Q8) return v + tp2_lean_xor<X>(v);
#endif
    return add_xor8<X>(v);
}

template <bool Q8, int X> __device__ __forceinline__ float v7_max(float v) {
#if defined(__gfx906__)
    if constexpr (Q8) return fmaxf(v, tp2_lean_xor<X>(v));
#endif
    return max_xor8<X>(v);
}

template <bool Q8> __device__ __forceinline__ float v7_prob(float v, int owner) {
#if defined(__gfx906__)
    if constexpr (Q8) return tp2_lean_broadcast(v, owner);
#endif
    return lean_broadcast8(v, owner);
}

// 1 query head per CTA (h1q16): 3x CTA parallelism and ~3x lower VGPR pressure for narrow T (32..256).
// Arithmetic, K-tile order, online softmax, and Kahan denominator are bit-for-bit identical to flash_lean_kernel.
template <bool Q8, int KvStride, int QStride>
__global__ void __launch_bounds__(kFlashThreads)
flash_lean_h1_kernel(const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ table,
                     const float* __restrict__ sink, const float* __restrict__ gate, KvView k, KvView v, int64_t pos,
                     int n, int window, int ring, float* __restrict__ out) {
    __shared__ float4 ks[kKeys][kHead / 4];
    __shared__ float4 vs[kKeys][kHead / 4];
    __shared__ float ts[kSlice * kRelative];
    const int tid = threadIdx.x, part = tid % kSplit, tl = tid / kSplit;
    const int h_out = blockIdx.y, g = h_out / kGroup;
    const int t0 = blockIdx.x * kTokens, last = min(t0 + kTokens, n) - 1, t = min(t0 + tl, last);
    const bool live = t0 + tl < n;
    const int64_t p = pos + t, lo = oldest(p + 1, window);
    const int64_t kmin = oldest(pos + t0 + 1, window), kmax = pos + last;
    const size_t qh = (size_t) t * QStride + h_out;

    float qr[4 * kVec], o[4 * kVec], rr[kRelPart];
    const float4* qv = reinterpret_cast<const float4*>(q + qh * kHead);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        const float4 x = qv[part + kSplit * i];
        qr[4 * i] = x.x;
        qr[4 * i + 1] = x.y;
        qr[4 * i + 2] = x.z;
        qr[4 * i + 3] = x.w;
        o[4 * i] = o[4 * i + 1] = o[4 * i + 2] = o[4 * i + 3] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < kRelPart; ++i) rr[i] = rel[qh * kRelative + part * kRelPart + i] * kLog2e;
    float m = sink2(sink[qh]), l = 0.f, lc = 0.f;

    Raw<Q8> raw;
    fetch<Q8, KvStride>(raw, k, v, kmin / kKeys * kKeys, kmin, kmax, ring, g, tid);
    const auto tile = [&]<bool Near, bool Mask>(int64_t kb) __attribute__((always_inline)) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int at_i = tid + i * kFlashThreads, c = at_i / (kHead / 4), w = at_i % (kHead / 4);
            ks[c][w] = unpack<Q8>(raw, i, false);
            vs[c][w] = unpack<Q8>(raw, i, true);
        }
        if (kb + kKeys <= kmax) fetch<Q8, KvStride>(raw, k, v, kb + kKeys, kmin, kmax, ring, g, tid);
        if constexpr (Near) {
            const int64_t dlo = pos + t0 - (kb + kKeys - 1);
            for (int i = tid; i < kSlice * kRelative; i += kFlashThreads) {
                const int64_t d = dlo + i / kRelative;
                ts[i] = d >= 0 && d < kExtent ? table[d * kRelative + i % kRelative] : 0.f;
            }
        }
        __syncthreads();

        float s[kKeys / kSplit] = {};
#pragma unroll
        for (int c = 0; c < kKeys; ++c) {
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const float4 x = ks[c][part + kSplit * i];
                dot = fmaf(qr[4 * i], x.x, dot);
                dot = fmaf(qr[4 * i + 1], x.y, dot);
                dot = fmaf(qr[4 * i + 2], x.z, dot);
                dot = fmaf(qr[4 * i + 3], x.w, dot);
            }
            const float* tr = ts + (tl + kKeys - 1 - c) * kRelative + part * kRelPart;
            float score = dot * (0.08838834764831845f * kLog2e);
            if constexpr (Near) {
#pragma unroll
                for (int i = 0; i < kRelPart; ++i) score = fmaf(rr[i], tr[i], score);
            }
            score = v7_add<Q8, 1>(score);
            score = v7_add<Q8, 2>(score);
            score = v7_add<Q8, 4>(score);
            const int64_t key = kb + c;
            if (part == c % kSplit) s[c / kSplit] = Mask && (key < lo || key > p) ? -INFINITY : score;
        }
        float top = fmaxf(s[0], s[1]);
        top = v7_max<Q8, 1>(top);
        top = v7_max<Q8, 2>(top);
        top = v7_max<Q8, 4>(top);
        const float mn = fmaxf(m, top), shrink = exp2f(m - mn);
        m = mn;
        l *= shrink;
        lc *= shrink;
#pragma unroll
        for (int i = 0; i < 4 * kVec; ++i) o[i] *= shrink;
        s[0] = exp2f(s[0] - mn);
        s[1] = exp2f(s[1] - mn);

        float sum = 0.f;
#pragma unroll
        for (int c = 0; c < kKeys; ++c) {
            const float prob = v7_prob<Q8>(s[c / kSplit], c % kSplit);
            sum += prob;
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const float4 x = vs[c][part + kSplit * i];
                o[4 * i] = fmaf(prob, x.x, o[4 * i]);
                o[4 * i + 1] = fmaf(prob, x.y, o[4 * i + 1]);
                o[4 * i + 2] = fmaf(prob, x.z, o[4 * i + 2]);
                o[4 * i + 3] = fmaf(prob, x.w, o[4 * i + 3]);
            }
        }
        const float y = sum - lc, next = l + y;
        lc = (next - l) - y;
        l = next;
    };

    int64_t kb = kmin / kKeys * kKeys;
    bool finite = true;
#pragma unroll
    for (int i = 0; i < kRelPart; ++i) finite = finite && isfinite(rr[i]);
    const bool far_ok = __syncthreads_and(finite) != 0 && window == 0;
    const int64_t far_end = far_ok ? std::min<int64_t>(kmax + 1, pos + t0 - kExtent - kKeys + 2) : kb;
    for (; kb < far_end; kb += kKeys) tile.template operator()<false, false>(kb);
    for (; kb <= kmax; kb += kKeys) tile.template operator()<true, true>(kb);
    if (!live) return;
    l += exp2f(sink2(sink[qh]) - m);
    const size_t at = qh * kHead;
#pragma unroll
    for (int i = 0; i < kVec; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t d = at + 4 * (part + kSplit * i) + j;
            out[d] = o[4 * i + j] / l / (1.f + expf(-gate[d]));
        }
}

// 3 query heads per CTA (h3q16) with parameterized (KvStride, QStride) for 2-head half-owner / half-helper execution.
template <bool Q8, int KvStride, int QStride>
__global__ void __launch_bounds__(kFlashThreads)
flash_lean_h3_half_kernel(const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ table,
                          const float* __restrict__ sink, const float* __restrict__ gate, KvView k, KvView v,
                          int64_t pos, int n, int window, int ring, float* __restrict__ out) {
    __shared__ float4 ks[kKeys][kHead / 4];
    __shared__ float4 vs[kKeys][kHead / 4];
    __shared__ float ts[kSlice * kRelative];
    const int tid = threadIdx.x, part = tid % kSplit, tl = tid / kSplit, g = blockIdx.y;
    const int t0 = blockIdx.x * kTokens, last = min(t0 + kTokens, n) - 1, t = min(t0 + tl, last);
    const bool live = t0 + tl < n;
    const int64_t p = pos + t, lo = oldest(p + 1, window);
    const int64_t kmin = oldest(pos + t0 + 1, window), kmax = pos + last;
    const size_t qh0 = (size_t) t * QStride + g * kGroup;

    float qr[kGroup][4 * kVec], o[kGroup][4 * kVec], rr[kGroup][kRelPart], m[kGroup], l[kGroup], lc[kGroup];
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        const float4* qv = reinterpret_cast<const float4*>(q + (qh0 + r) * kHead);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            const float4 x = qv[part + kSplit * i];
            qr[r][4 * i] = x.x;
            qr[r][4 * i + 1] = x.y;
            qr[r][4 * i + 2] = x.z;
            qr[r][4 * i + 3] = x.w;
            o[r][4 * i] = o[r][4 * i + 1] = o[r][4 * i + 2] = o[r][4 * i + 3] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < kRelPart; ++i) rr[r][i] = rel[(qh0 + r) * kRelative + part * kRelPart + i] * kLog2e;
        m[r] = sink2(sink[qh0 + r]);
        l[r] = lc[r] = 0.f;
    }

    Raw<Q8> raw;
    fetch<Q8, KvStride>(raw, k, v, kmin / kKeys * kKeys, kmin, kmax, ring, g, tid);
    const auto tile = [&]<bool Near, bool Mask>(int64_t kb) __attribute__((always_inline)) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int at_i = tid + i * kFlashThreads, c = at_i / (kHead / 4), w = at_i % (kHead / 4);
            ks[c][w] = unpack<Q8>(raw, i, false);
            vs[c][w] = unpack<Q8>(raw, i, true);
        }
        if (kb + kKeys <= kmax) fetch<Q8, KvStride>(raw, k, v, kb + kKeys, kmin, kmax, ring, g, tid);
        if constexpr (Near) {
            const int64_t dlo = pos + t0 - (kb + kKeys - 1);
            for (int i = tid; i < kSlice * kRelative; i += kFlashThreads) {
                const int64_t d = dlo + i / kRelative;
                ts[i] = d >= 0 && d < kExtent ? table[d * kRelative + i % kRelative] : 0.f;
            }
        }
        __syncthreads();

        float s[kGroup][kKeys / kSplit];
#pragma unroll
        for (int r = 0; r < kGroup; ++r) s[r][0] = s[r][1] = 0.f;
#pragma unroll
        for (int c = 0; c < kKeys; ++c) {
            float dot[kGroup] = {};
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const float4 x = ks[c][part + kSplit * i];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    dot[r] = fmaf(qr[r][4 * i], x.x, dot[r]);
                    dot[r] = fmaf(qr[r][4 * i + 1], x.y, dot[r]);
                    dot[r] = fmaf(qr[r][4 * i + 2], x.z, dot[r]);
                    dot[r] = fmaf(qr[r][4 * i + 3], x.w, dot[r]);
                }
            }
            const float* tr = ts + (tl + kKeys - 1 - c) * kRelative + part * kRelPart;
#pragma unroll
            for (int r = 0; r < kGroup; ++r) {
                float score = dot[r] * (0.08838834764831845f * kLog2e);
                if constexpr (Near) {
#pragma unroll
                    for (int i = 0; i < kRelPart; ++i) score = fmaf(rr[r][i], tr[i], score);
                }
                score = v7_add<Q8, 1>(score);
                score = v7_add<Q8, 2>(score);
                score = v7_add<Q8, 4>(score);
                const int64_t key = kb + c;
                if (part == c % kSplit) s[r][c / kSplit] = Mask && (key < lo || key > p) ? -INFINITY : score;
            }
        }
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            float top = fmaxf(s[r][0], s[r][1]);
            top = v7_max<Q8, 1>(top);
            top = v7_max<Q8, 2>(top);
            top = v7_max<Q8, 4>(top);
            const float mn = fmaxf(m[r], top), shrink = exp2f(m[r] - mn);
            m[r] = mn;
            l[r] *= shrink;
            lc[r] *= shrink;
#pragma unroll
            for (int i = 0; i < 4 * kVec; ++i) o[r][i] *= shrink;
            s[r][0] = exp2f(s[r][0] - mn);
            s[r][1] = exp2f(s[r][1] - mn);
        }
        float sum[kGroup] = {};
#pragma unroll
        for (int c = 0; c < kKeys; ++c) {
            float prob[kGroup];
#pragma unroll
            for (int r = 0; r < kGroup; ++r) {
                prob[r] = v7_prob<Q8>(s[r][c / kSplit], c % kSplit);
                sum[r] += prob[r];
            }
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const float4 x = vs[c][part + kSplit * i];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    o[r][4 * i] = fmaf(prob[r], x.x, o[r][4 * i]);
                    o[r][4 * i + 1] = fmaf(prob[r], x.y, o[r][4 * i + 1]);
                    o[r][4 * i + 2] = fmaf(prob[r], x.z, o[r][4 * i + 2]);
                    o[r][4 * i + 3] = fmaf(prob[r], x.w, o[r][4 * i + 3]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            const float y = sum[r] - lc[r], next = l[r] + y;
            lc[r] = (next - l[r]) - y;
            l[r] = next;
        }
    };

    int64_t kb = kmin / kKeys * kKeys;
    bool finite = true;
#pragma unroll
    for (int r = 0; r < kGroup; ++r)
#pragma unroll
        for (int i = 0; i < kRelPart; ++i) finite = finite && isfinite(rr[r][i]);
    const bool far_ok = __syncthreads_and(finite) != 0 && window == 0;
    const int64_t far_end = far_ok ? std::min<int64_t>(kmax + 1, pos + t0 - kExtent - kKeys + 2) : kb;
    for (; kb < far_end; kb += kKeys) tile.template operator()<false, false>(kb);
    for (; kb <= kmax; kb += kKeys) tile.template operator()<true, true>(kb);
    if (!live) return;
#pragma unroll
    for (int r = 0; r < kGroup; ++r) {
        l[r] += exp2f(sink2(sink[qh0 + r]) - m[r]);
        const size_t at = (qh0 + r) * kHead;
#pragma unroll
        for (int i = 0; i < kVec; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const size_t d = at + 4 * (part + kSplit * i) + j;
                out[d] = o[r][4 * i + j] / l[r] / (1.f + expf(-gate[d]));
            }
    }
}

}  // namespace

void attend_flash_lean_h1(const Attend& a, cudaStream_t st) {
    if (a.n <= 0) return;
    if (a.kv_heads != 4 || !a.k.scales) {
        attend_flash_lean(a, st);
        return;
    }
    const dim3 grid(div_up(a.n, kTokens), 4 * kGroup);
    flash_lean_h1_kernel<true, 4, 4 * kGroup><<<grid, kFlashThreads, 0, st>>>(
        a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n, a.window, a.ring, a.out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("Inkling lean h1 flash attention: ") + cudaGetErrorString(e));
}

void attend_flash_lean_half(const Attend& a, bool h1, int kv_stride, cudaStream_t st) {
    if (a.n <= 0) return;
    if (!a.k.scales || (kv_stride != 4 && kv_stride != 2))
        throw std::invalid_argument("attend_flash_lean_half requires Q8_0 KV and kv_stride 2 or 4");
    const int q_tiles = div_up(a.n, kTokens);
    if (kv_stride == 4) {
        if (h1)
            flash_lean_h1_kernel<true, 4, 12><<<dim3(q_tiles, 6), kFlashThreads, 0, st>>>(
                a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n, a.window, a.ring, a.out);
        else
            flash_lean_h3_half_kernel<true, 4, 12><<<dim3(q_tiles, 2), kFlashThreads, 0, st>>>(
                a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n, a.window, a.ring, a.out);
    } else {
        if (h1)
            flash_lean_h1_kernel<true, 2, 6><<<dim3(q_tiles, 6), kFlashThreads, 0, st>>>(
                a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n, a.window, a.ring, a.out);
        else
            flash_lean_h3_half_kernel<true, 2, 6><<<dim3(q_tiles, 2), kFlashThreads, 0, st>>>(
                a.q, a.rel, a.table, a.sink, a.gate, a.k, a.v, a.pos, a.n, a.window, a.ring, a.out);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Inkling lean half flash attention: ") + cudaGetErrorString(e));
}

}  // namespace strata::poly::inkling
