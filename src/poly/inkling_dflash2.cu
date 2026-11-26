// The DFlash2 drafter's own launches (inkling_dflash2.hpp).
#include "inkling_dflash2.hpp"

#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::poly::dflash2 {
namespace {

__host__ __device__ int div_up(int64_t a, int b) { return (int) ((a + b - 1) / b); }
constexpr float kScale = 0.08838834764831845f;   // 1 / sqrt(128)

__global__ void gather_kernel(Taps t, int first, float* out) {
    const int r = blockIdx.x, j = blockIdx.y;
    const float4* s = reinterpret_cast<const float4*>(t.p[j] + (size_t) (first + r) * kWidth);
    float4* d = reinterpret_cast<float4*>(out + (size_t) r * 4 * kWidth + (size_t) j * kWidth);
    for (int i = threadIdx.x; i < kWidth / 4; i += 256) d[i] = s[i];
}

__global__ void conv_kernel(const float* __restrict__ x, const float* __restrict__ dyn, const float* __restrict__ base,
                            int stage, float* out, bool add) {
    const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (c >= kWidth) return;
    const int g = c / 16;
    const float* d = dyn + (size_t) r * kDyn + stage * 2 * kGroups;
    float y = (base[(stage * 2 + 0) * kWidth + c] + d[g]) * x[(size_t) r * kWidth + c];
    if (r) y += (base[(stage * 2 + 1) * kWidth + c] + d[kGroups + g]) * x[(size_t) (r - 1) * kWidth + c];
    float* o = out + (size_t) r * kWidth + c;
    *o = add ? *o + y : y;
}

__global__ void store_kernel(const float* k, const float* v, __half* kc, __half* vc, int64_t pos) {
    const int r = blockIdx.x;
    const size_t slot = (size_t) ((pos + r) & (kWindow - 1));
    const float* src = blockIdx.y ? v : k;
    __half* dst = (blockIdx.y ? vc : kc) + slot * kKey;
    for (int i = threadIdx.x; i < kKey; i += 256) dst[i] = __float2half(src[(size_t) r * kKey + i]);
}

// Attention: a CTA takes one K/V group's 32 queries (8 rows x 4 heads) over kChunk keys; partial max, sum and output
// a query; a second launch adds the chunks.  Keys: ring positions lo0 .. pos - 1 (n_ring), then the block's 8.
constexpr int kQ = kBlock * (kHeads / kKv);   // 32
__global__ void __launch_bounds__(256) attend_part_kernel(const float* __restrict__ q, const float* __restrict__ kb,
                                                           const float* __restrict__ vb, const __half* __restrict__ kc,
                                                           const __half* __restrict__ vc, int64_t lo0, int n_ring,
                                                           int64_t pos, int chunks, float* part) {
    __shared__ float qs[kQ][kHead + 1];
    __shared__ __half ks[kChunk][kHead + 2];
    __shared__ __half vs[kChunk][kHead + 2];
    __shared__ float p[kQ][kChunk + 1];
    __shared__ int64_t kpos[kChunk];
    const int g = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
    const int n_keys = n_ring + kBlock, k0 = c * kChunk;
    // queries: qi = row * 4 + hh -> head g * 4 + hh of row
    for (int i = tid; i < kQ * kHead; i += 256) {
        const int qi = i / kHead, d = i % kHead, row = qi / 4, hh = qi % 4;
        qs[qi][d] = q[((size_t) row * kHeads + g * 4 + hh) * kHead + d];
    }
    for (int i = tid; i < kChunk * kHead; i += 256) {
        const int j = i / kHead, d = i % kHead, kk = k0 + j;
        __half kv = __float2half(0.f), vv = __float2half(0.f);
        if (kk < n_ring) {
            const size_t slot = (size_t) ((lo0 + kk) & (kWindow - 1));
            kv = kc[slot * kKey + g * kHead + d];
            vv = vc[slot * kKey + g * kHead + d];
        } else if (kk < n_keys) {
            const int b = kk - n_ring;
            kv = __float2half(kb[((size_t) b * kKv + g) * kHead + d]);
            vv = __float2half(vb[((size_t) b * kKv + g) * kHead + d]);
        }
        ks[j][d] = kv;
        vs[j][d] = vv;
        if (d == 0) kpos[j] = kk < n_ring ? lo0 + kk : kk < n_keys ? pos + (kk - n_ring) : INT64_MIN;
    }
    __syncthreads();
    // scores: thread -> query tid / 8, keys (tid % 8) + 8 j
    const int qi = tid / 8, sub = tid % 8, row = qi / 4;
    const int64_t qpos = pos + row;
    float m = -INFINITY;
    float sc[kChunk / 8];
#pragma unroll
    for (int j = 0; j < kChunk / 8; ++j) {
        const int kj = sub + 8 * j;
        const int64_t kp = kpos[kj];
        // block keys are always visible; ring keys within kWindow - 1 of the query
        const bool ok = kp != INT64_MIN && (kp >= pos || qpos - kp < kWindow);
        float s = -INFINITY;
        if (ok) {
            s = 0.f;
#pragma unroll 8
            for (int d = 0; d < kHead; ++d) s += qs[qi][d] * __half2float(ks[kj][d]);
            s *= kScale;
        }
        sc[j] = s;
        m = fmaxf(m, s);
    }
    for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 8));
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < kChunk / 8; ++j) {
        const float e = sc[j] == -INFINITY ? 0.f : __expf(sc[j] - m);
        p[qi][sub + 8 * j] = e;
        l += e;
    }
    for (int o = 4; o > 0; o >>= 1) l += __shfl_xor(l, o, 8);
    __syncthreads();
    // output: thread -> query tid / 8, dims (tid % 8) * 16 .. + 16
    float acc[16] = {};
    for (int j = 0; j < kChunk; ++j) {
        const float pj = p[qi][j];
#pragma unroll
        for (int d = 0; d < 16; ++d) acc[d] += pj * __half2float(vs[j][sub * 16 + d]);
    }
    float* o = part + ((size_t) (g * chunks + c) * kQ + qi) * (kHead + 2);
#pragma unroll
    for (int d = 0; d < 16; ++d) o[sub * 16 + d] = acc[d];
    if (sub == 0) {
        o[kHead] = m;
        o[kHead + 1] = l;
    }
}

__global__ void __launch_bounds__(kHead) attend_add_kernel(const float* __restrict__ part, int chunks, float* out) {
    __shared__ float w[kMaxChunks], red[2];
    const int g = blockIdx.x / kQ, qi = blockIdx.x % kQ, d = threadIdx.x;
    const float* base = part + ((size_t) g * chunks * kQ + qi) * (kHead + 2);
    const size_t step = (size_t) kQ * (kHead + 2);
    // a thread a chunk: the chunks' maxima, the largest, each chunk's weight and the total
    const float mc = d < chunks ? base[d * step + kHead] : -INFINITY;
    const float lc = d < chunks ? base[d * step + kHead + 1] : 0.f;
    float big = mc;
    for (int o = 32; o > 0; o >>= 1) big = fmaxf(big, __shfl_xor(big, o, 64));
    if (d % 64 == 0) red[d / 64] = big;
    __syncthreads();
    big = fmaxf(red[0], red[1]);
    const float wc = mc == -INFINITY ? 0.f : __expf(mc - big);
    if (d < chunks) w[d] = wc;
    float l = wc * lc;
    for (int o = 32; o > 0; o >>= 1) l += __shfl_xor(l, o, 64);
    __syncthreads();
    if (d % 64 == 0) red[d / 64] = l;
    __syncthreads();
    l = red[0] + red[1];
    float acc = 0.f;
#pragma unroll 8
    for (int c = 0; c < chunks; ++c) acc += w[c] * base[c * step + d];
    const int row = qi / 4, hh = qi % 4;
    out[((size_t) row * kHeads + g * 4 + hh) * kHead + d] = l > 0.f ? acc / l : 0.f;
}

// ---- top-16: per row, 128 chunks each keep their 16 best, then one block keeps the best 16 of the 2048 ----
constexpr int kTopParts = 512;
__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }
__device__ __forceinline__ void block_best(float& v, int& i, float* sv, int* si) {
    for (int o = 32; o > 0; o >>= 1) {
        const float v2 = __shfl_xor(v, o, 64);
        const int i2 = __shfl_xor(i, o, 64);
        if (better(v2, i2, v, i)) { v = v2; i = i2; }
    }
    const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
    if (lane == 0) { sv[wave] = v; si[wave] = i; }
    __syncthreads();
    v = sv[0];
    i = si[0];
    for (int w = 1; w < (int) blockDim.x / 64; ++w)
        if (better(sv[w], si[w], v, i)) { v = sv[w]; i = si[w]; }
    __syncthreads();
}
// a part: one wave over valid / kTopParts values (<= 392), each lane holding <= 7 of them in registers
constexpr int kTopLane = 7;
__global__ void __launch_bounds__(64) topk_part_kernel(const float* __restrict__ logits, int64_t ld, int valid,
                                                        float* part) {
    const int row = blockIdx.y, lane = threadIdx.x, span = div_up(valid, kTopParts), first = blockIdx.x * span;
    const int n = max(0, min(span, valid - first));
    float v[kTopLane];
#pragma unroll
    for (int u = 0; u < kTopLane; ++u) {
        const int j = lane + 64 * u;
        const float x = j < n ? logits[(size_t) row * ld + first + j] : -INFINITY;
        v[u] = x != x ? -INFINITY : x;
    }
    float* o = part + ((size_t) row * kTopParts + blockIdx.x) * 2 * kTop;
    for (int r = 0; r < kTop; ++r) {
        float bv = -INFINITY;
        int bi = INT32_MAX, bu = -1;
#pragma unroll
        for (int u = 0; u < kTopLane; ++u) {
            const int id = first + lane + 64 * u;
            if (better(v[u], id, bv, bi)) { bv = v[u]; bi = id; bu = u; }
        }
        float wv = bv;
        int wi = bi;
        for (int s = 32; s > 0; s >>= 1) {
            const float v2 = __shfl_xor(wv, s, 64);
            const int i2 = __shfl_xor(wi, s, 64);
            if (better(v2, i2, wv, wi)) { wv = v2; wi = i2; }
        }
        if (bi == wi && bu >= 0) {   // the lane that holds the winner takes it out
#pragma unroll
            for (int u = 0; u < kTopLane; ++u) if (u == bu) v[u] = -INFINITY;
        }
        if (lane == 0) { o[r] = wv; o[kTop + r] = __int_as_float(wi); }
    }
}
// the 16 best of the parts' sorted lists: each thread the heads of two parts' lists, 16 block rounds
__global__ void __launch_bounds__(256) topk_merge_kernel(const float* __restrict__ part, int32_t* ids_out,
                                                          float* vals_out) {
    static_assert(kTopParts == 2 * 256);
    __shared__ float sv[4];
    __shared__ int si[4];
    const int row = blockIdx.x, t = threadIdx.x;
    const float* l0 = part + ((size_t) row * kTopParts + 2 * t) * 2 * kTop;
    const float* l1 = l0 + 2 * kTop;
    int p0 = 0, p1 = 0;
    for (int r = 0; r < kTop; ++r) {
        const float v0 = p0 < kTop ? l0[p0] : -INFINITY, v1 = p1 < kTop ? l1[p1] : -INFINITY;
        const int i0 = p0 < kTop ? __float_as_int(l0[kTop + p0]) : INT32_MAX;
        const int i1 = p1 < kTop ? __float_as_int(l1[kTop + p1]) : INT32_MAX;
        const bool first = better(v0, i0, v1, i1);
        float bv = first ? v0 : v1;
        int bi = first ? i0 : i1;
        const float mine = bv;
        const int mine_i = bi;
        block_best(bv, bi, sv, si);
        if (mine_i == bi && mine == bv && bi != INT32_MAX) (first ? p0 : p1)++;
        if (t == 0) {
            vals_out[row * kTop + r] = bv;
            ids_out[row * kTop + r] = bi;
        }
    }
}

// ---- selector: edges a position, then the walk ----
__device__ __forceinline__ float book_at(Book book, size_t i) {
    if (book.kind == BookKind::kF16) return __half2float(static_cast<const __half*>(book.data)[i]);
    const int lane = i % 32;
    const bool q4 = book.kind == BookKind::kQ4;
    const auto* blk = static_cast<const uint8_t*>(book.data) + i / 32 * (q4 ? 18 : 34);
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const int q = q4 ? ((blk[2 + lane % 16] >> (lane / 16 * 4)) & 15) - 8 : (int8_t) blk[2 + lane];
    return d * q;
}

template <bool HalfBooks>
__global__ void __launch_bounds__(256) edges_kernel(const int32_t* __restrict__ ids, const float* __restrict__ vals,
                                                     const float* __restrict__ h, Book pred, Book succ, int32_t anchor, float cap,
                                                     float* edges) {
    __shared__ float ps[kTop][kRank + 1];
    __shared__ float qs[kTop][kRank + 1];
    const int t = blockIdx.x, tid = threadIdx.x;
    const int n_pred = t == 0 ? 1 : kTop;
    const float* ht = h + (size_t) t * kRank;
    for (int i = tid; i < kTop * kRank; i += 256) {
        const int a = i / kRank, r = i % kRank;
        if (a < n_pred) {
            const int32_t tok = t == 0 ? anchor : ids[(t - 1) * kTop + a];
            const size_t i = (size_t) tok * kRank + r;
            const float p = HalfBooks ? __half2float(static_cast<const __half*>(pred.data)[i]) : book_at(pred, i);
            ps[a][r] = p * ht[r];
        }
        const size_t at = (size_t) ids[t * kTop + a] * kRank + r;
        qs[a][r] = HalfBooks ? __half2float(static_cast<const __half*>(succ.data)[at]) : book_at(succ, at);
    }
    __syncthreads();
    const int a = tid / kTop, b = tid % kTop;
    if (a >= n_pred) return;
    float pair = 0.f;
    for (int r = 0; r < kRank; ++r) pair += ps[a][r] * qs[b][r];
    edges[(size_t) t * kTop * kTop + a * kTop + b] = cap * tanhf((vals[t * kTop + b] + pair) / cap);
}
__global__ void walk_kernel(const int32_t* __restrict__ ids, const float* __restrict__ edges, float inv_temp,
                            Guess* out) {
    if (threadIdx.x != 0) return;
    int prev = 0;
    for (int t = 0; t < kGuess; ++t) {
        const float* e = edges + (size_t) t * kTop * kTop + (t == 0 ? 0 : prev) * kTop;
        int best = 0;
        for (int b = 1; b < kTop; ++b)
            if (e[b] > e[best]) best = b;
        float sum = 0.f;
        for (int b = 0; b < kTop; ++b) sum += __expf((e[b] - e[best]) * inv_temp);
        out[t].token = ids[t * kTop + best];
        out[t].prob = 1.f / sum;
        prev = best;
    }
}


// ---- the drafter's dense products: aligned int8 codes, 16 lanes a row, 8 activation rows from LDS ----
__global__ void pack_kernel(const uint8_t* __restrict__ src, uint8_t* q, float* d, size_t total) {
    const size_t b = (size_t) blockIdx.x * 8 + threadIdx.x / 32;   // a block of 32 values a 32-thread group
    if (b >= total) return;
    const int j = threadIdx.x % 32;
    const uint8_t* blk = src + b * 34;
    q[b * 32 + j] = blk[2 + j];
    if (j == 0) d[b] = __half2float(*reinterpret_cast<const __half*>(blk));
}
template <typename Scale>
__global__ void pack4_kernel(const uint8_t* __restrict__ src, uint8_t* q, Scale* d, size_t total) {
    const size_t b = (size_t) blockIdx.x * 16 + threadIdx.x / 16;
    if (b >= total) return;
    const int j = threadIdx.x % 16;
    const uint8_t* blk = src + b * 18;
    q[b * 16 + j] = blk[2 + j];
    if (j == 0) {
        const __half scale = *reinterpret_cast<const __half*>(blk);
        if constexpr (sizeof(Scale) == 2) d[b] = scale;
        else d[b] = __half2float(scale);
    }
}

template <bool Sum>
__device__ __forceinline__ void quant_block(float v, int lane, int8_t* q, float* d, int32_t* xsum, size_t b) {
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
    const float s = a / 127.f;
    const auto code = (int8_t) (s > 0.f ? __float2int_rn(v / s) : 0);
    *q = code;
    if (lane == 0) d[b] = s;
    if constexpr (Sum) {
        int sum = code;
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
        if (lane == 0) xsum[b] = sum;
    }
}

template <bool Sum>
__global__ void quantize_kernel(const float* __restrict__ x, int n_in, int8_t* q, float* d, int32_t* xsum) {
    // a 32-value block a 32-lane group: d = amax / 127, q = round(x / d)
    const int r = blockIdx.y, b = blockIdx.x * 8 + threadIdx.x / 32, j = threadIdx.x % 32;
    if (b * 32 >= n_in) return;
    const size_t at = (size_t) r * n_in + b * 32 + j;
    quant_block<Sum>(x[at], j, q + at, d, xsum, (size_t) r * (n_in / 32) + b);
}
struct Outs {
    float* y[3];
    int n[3];
};
constexpr int kTileK = 2304;   // x staged in K tiles: 8 rows of 2304 codes = 18 KB, 2 KB of scales
template <int NC>
__global__ void __launch_bounds__(256) gemv8_kernel(const int8_t* __restrict__ wq, const float* __restrict__ wd,
                                                     const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                     Outs out, int n_in, int n_out) {
    constexpr int U = 3;   // a lane's chunks loaded a step ahead
    __shared__ int4 xs[NC * (kTileK / 16)];
    __shared__ float xds[NC * (kTileK / 32)];
    const int j = threadIdx.x % 16, row = blockIdx.x * 16 + threadIdx.x / 16, blocks = n_in / 32;
    float acc[NC] = {};
    for (int k0 = 0; k0 < n_in; k0 += kTileK) {
        const int chunks = min(kTileK, n_in - k0) / 16;
        __syncthreads();
        for (int i = threadIdx.x; i < NC * chunks; i += 256) {
            const int k = i / chunks, c = i % chunks;
            xs[k * (kTileK / 16) + c] = reinterpret_cast<const int4*>(xq + (size_t) k * n_in + k0)[c];
        }
        for (int i = threadIdx.x; i < NC * (chunks / 2); i += 256) {
            const int k = i / (chunks / 2), c = i % (chunks / 2);
            xds[k * (kTileK / 32) + c] = xd[(size_t) k * blocks + k0 / 32 + c];
        }
        __syncthreads();
        if (row >= n_out) continue;
        const int4* w = reinterpret_cast<const int4*>(wq + (size_t) row * n_in + k0);
        const float* ws = wd + (size_t) row * blocks + k0 / 32;
        for (int c0 = j; c0 < chunks; c0 += 16 * U) {
            int4 a[U];
            float s[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int c = c0 + 16 * u;
                a[u] = c < chunks ? w[c] : make_int4(0, 0, 0, 0);
                s[u] = c < chunks ? ws[c / 2] : 0.f;
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int c = c0 + 16 * u;
                if (c >= chunks) break;
#pragma unroll
                for (int k = 0; k < NC; ++k) {
                    const int4 b = xs[k * (kTileK / 16) + c];
                    int t = __builtin_amdgcn_sdot4(a[u].x, b.x, 0, false);
                    t = __builtin_amdgcn_sdot4(a[u].y, b.y, t, false);
                    t = __builtin_amdgcn_sdot4(a[u].z, b.z, t, false);
                    t = __builtin_amdgcn_sdot4(a[u].w, b.w, t, false);
                    acc[k] += (float) t * s[u] * xds[k * (kTileK / 32) + c / 2];
                }
            }
        }
    }
    if (row >= n_out) return;
    int seg = 0, at = row;   // the output segment the row is in (a fused q|k|v or gate|up)
    while (seg < 2 && at >= out.n[seg]) at -= out.n[seg++];
#pragma unroll
    for (int k = 0; k < NC; ++k) {
        float v = acc[k];
        for (int o = 8; o > 0; o >>= 1) v += __shfl_xor(v, o, 16);
        if (j == 0) out.y[seg][(size_t) k * out.n[seg] + at] = v;
    }
}

// STRATA_DFLASH2_FAST=1: 8 lanes/row, 2 rows/subwarp (64 rows/CTA), full 32B block (8x sdot4 per scale).
// Used when n_out >= 4096 (qkv8, gu8, head8) where 64 rows/CTA saturates all 60 CUs and halves LDS & FP32 work.
template <int NC>
__global__ void __launch_bounds__(256) gemv8_l8_r2_kernel(const int8_t* __restrict__ wq, const float* __restrict__ wd,
                                                          const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                          Outs out, int n_in, int n_out) {
    __shared__ int4 xs[NC * (kTileK / 16)];
    __shared__ float xds[NC * (kTileK / 32)];
    const int j = int(threadIdx.x) & 7;
    const int row0 = (int(blockIdx.x) * 32 + (int(threadIdx.x) >> 3)) * 2;
    const int blocks = n_in / 32;
    float acc[2][NC] = {};
    for (int k0 = 0; k0 < n_in; k0 += kTileK) {
        const int blks = min(kTileK, n_in - k0) / 32;
        const int chunks = blks * 2;
        __syncthreads();
        for (int i = int(threadIdx.x); i < NC * chunks; i += 256) {
            const int k = i / chunks, c = i % chunks;
            xs[k * (kTileK / 16) + c] = reinterpret_cast<const int4*>(xq + (size_t) k * n_in + k0)[c];
        }
        for (int i = int(threadIdx.x); i < NC * blks; i += 256) {
            const int k = i / blks, b = i % blks;
            xds[k * (kTileK / 32) + b] = xd[(size_t) k * blocks + k0 / 32 + b];
        }
        __syncthreads();
        if (row0 >= n_out) continue;
        for (int blk = j; blk < blks; blk += 8) {
            int4 a0[2], a1[2];
            float ws[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int rr = row0 + r;
                if (rr < n_out) {
                    const int4* w = reinterpret_cast<const int4*>(wq + (size_t) rr * n_in + k0);
                    a0[r] = w[2 * blk];
                    a1[r] = w[2 * blk + 1];
                    ws[r] = wd[(size_t) rr * blocks + k0 / 32 + blk];
                } else {
                    a0[r] = make_int4(0, 0, 0, 0);
                    a1[r] = make_int4(0, 0, 0, 0);
                    ws[r] = 0.f;
                }
            }
#pragma unroll
            for (int k = 0; k < NC; ++k) {
                const int4 b0 = xs[k * (kTileK / 16) + 2 * blk];
                const int4 b1 = xs[k * (kTileK / 16) + 2 * blk + 1];
                const float xscale = xds[k * (kTileK / 32) + blk];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    int t = __builtin_amdgcn_sdot4(a0[r].x, b0.x, 0, false);
                    t = __builtin_amdgcn_sdot4(a0[r].y, b0.y, t, false);
                    t = __builtin_amdgcn_sdot4(a0[r].z, b0.z, t, false);
                    t = __builtin_amdgcn_sdot4(a0[r].w, b0.w, t, false);
                    t = __builtin_amdgcn_sdot4(a1[r].x, b1.x, t, false);
                    t = __builtin_amdgcn_sdot4(a1[r].y, b1.y, t, false);
                    t = __builtin_amdgcn_sdot4(a1[r].z, b1.z, t, false);
                    t = __builtin_amdgcn_sdot4(a1[r].w, b1.w, t, false);
                    acc[r][k] += float(t) * (ws[r] * xscale);
                }
            }
        }
    }
    if (row0 >= n_out) return;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int rr = row0 + r;
        if (rr >= n_out) break;
        int seg = 0, at = rr;
        while (seg < 2 && at >= out.n[seg]) at -= out.n[seg++];
#pragma unroll
        for (int k = 0; k < NC; ++k) {
            float v = acc[r][k];
            for (int o = 4; o > 0; o >>= 1) v += __shfl_xor(v, o, 8);
            if (j == 0) out.y[seg][(size_t) k * out.n[seg] + at] = v;
        }
    }
}

// GGML split-half nibbles: byte j holds w[j] and w[16+j], each biased by 8.
// Code plane: one aligned int4/block. Separate d plane: FP32 (20 B/block total)
// or FP16 (18 B/block, no padding). Activation codes stay in V5-R3 order.
struct Pair { float v; int32_t id; };
static_assert(sizeof(Pair) == 8);
constexpr Pair invalid{-INFINITY, INT32_MAX};

// Run the bitonic network on one full wave per position, and keep its first 16.
// Four waves per CTA produce seven lists in two passes.
template <int Rows>
__device__ __forceinline__ void local_top16(const float* stage, Pair* parts, int n) {
    static_assert(Rows <= 64);
    const int t = threadIdx.x, wave = t / 64, lane = t % 64;
    const int out = int(blockIdx.x) * Rows + lane;
    for (int m = wave; m < kGuess; m += 4) {
        Pair a = invalid;
        if (lane < Rows && out < n) {
            a = {stage[m * Rows + lane], out};
            if (isnan(a.v)) a.v = -INFINITY;   // keep a valid id even if all scores are NaN
        }
#pragma unroll
        for (int len = 2; len <= Rows; len *= 2) {
#pragma unroll
            for (int hop = len / 2; hop > 0; hop /= 2) {
                const Pair b{__shfl_xor(a.v, hop, 64), __shfl_xor(a.id, hop, 64)};
                const bool want_better = ((lane & len) == 0) == ((lane & hop) == 0);
                if (want_better ? better(b.v, b.id, a.v, a.id) : better(a.v, a.id, b.v, b.id)) a = b;
            }
        }
        if (lane < kTop) {
            // Make rank 0 contiguous across CTAs for the merge; the remaining
            // 15 are contiguous within a CTA's list. Exactly 16 pairs/CTA/pos.
            const size_t i = lane == 0 ? blockIdx.x : gridDim.x + size_t(blockIdx.x) * (kTop - 1) + lane - 1;
            parts[size_t(m) * gridDim.x * kTop + i] = a;
        }
    }
}

template <int M, int L, int R, bool Q4, typename Scale, bool Fused = false>
__global__ void __launch_bounds__(256) planar_kernel(
        const int4* __restrict__ wq, const Scale* __restrict__ wd,
        const int8_t* __restrict__ xq, const float* __restrict__ xd, const int32_t* __restrict__ xsum,
        Outs out, Pair* __restrict__ parts, int k, int n) {
    constexpr int rows = (256 / L) * R, kb_tile = kTileK / 32;
    static_assert(!Fused || (M == kGuess && rows <= 64));
    static_assert(Q4 || (L == 8 && R == 2));
    __shared__ int4 xs[M * (kTileK / 16)];
    __shared__ float xds[M * kb_tile];
    __shared__ int xcs[Q4 ? M * kb_tile : 1];
    static_assert(!Fused || rows <= kb_tile);
    const int t = threadIdx.x, lane = t % L, local = (t / L) * R;
    const int first = int(blockIdx.x) * rows + local, kb = k / 32;
    float acc[R][M] = {};
    for (int k0 = 0; k0 < kb; k0 += kb_tile) {
        const int blks = min(kb_tile, kb - k0), chunks = 2 * blks;
        __syncthreads();
        for (int i = t; i < M * chunks; i += 256) {
            const int m = i / chunks, c = i % chunks;
            xs[m * (kTileK / 16) + c] = reinterpret_cast<const int4*>(xq)[size_t(m) * 2 * kb + 2 * k0 + c];
        }
        for (int i = t; i < M * blks; i += 256) {
            const int m = i / blks, b = i % blks;
            const size_t xi = size_t(m) * kb + k0 + b;
            xds[m * kb_tile + b] = xd[xi];
            if constexpr (Q4) xcs[m * kb_tile + b] = -8 * xsum[xi];
        }
        __syncthreads();
        if (first >= n) continue;
        for (int b = lane; b < blks; b += L) {
            int4 a0[R], a1[R];
            float ws[R];
            constexpr int mask = 0x0f0f0f0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                a0[r] = a1[r] = make_int4(0, 0, 0, 0);
                ws[r] = 0.f;
                if (first + r < n) {
                    const size_t wi = size_t(first + r) * kb + k0 + b;
                    if constexpr (Q4) {
                        const int4 a = wq[wi];
                        a0[r] = make_int4(a.x & mask, a.y & mask, a.z & mask, a.w & mask);
                        a1[r] = make_int4((a.x >> 4) & mask, (a.y >> 4) & mask, (a.z >> 4) & mask, (a.w >> 4) & mask);
                    } else {
                        a0[r] = wq[2 * wi];
                        a1[r] = wq[2 * wi + 1];
                    }
                    if constexpr (sizeof(Scale) == 2) ws[r] = __half2float(wd[wi]);
                    else ws[r] = wd[wi];
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const int4 x0 = xs[m * (kTileK / 16) + 2 * b], x1 = xs[m * (kTileK / 16) + 2 * b + 1];
                const float xd0 = xds[m * kb_tile + b];
                const int init = Q4 ? xcs[m * kb_tile + b] : 0;   // -8 * sum(xq) for biased nibbles
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int dot = __builtin_amdgcn_sdot4(a0[r].x, x0.x, init, false);
                    dot = __builtin_amdgcn_sdot4(a0[r].y, x0.y, dot, false);
                    dot = __builtin_amdgcn_sdot4(a0[r].z, x0.z, dot, false);
                    dot = __builtin_amdgcn_sdot4(a0[r].w, x0.w, dot, false);
                    dot = __builtin_amdgcn_sdot4(a1[r].x, x1.x, dot, false);
                    dot = __builtin_amdgcn_sdot4(a1[r].y, x1.y, dot, false);
                    dot = __builtin_amdgcn_sdot4(a1[r].z, x1.z, dot, false);
                    dot = __builtin_amdgcn_sdot4(a1[r].w, x1.w, dot, false);
                    acc[r][m] += float(dot) * (ws[r] * xd0);
                }
            }
        }
    }
    // Reuse the activation-scale tile after all its readers are done. A
    // separate M*64-float scratch tile would reduce Q4 head CTA residency
    // from three to two on gfx906 (64 KiB LDS/CU).
    if constexpr (Fused) __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (first + r >= n) break;
        int seg = 0, at = first + r;
        if constexpr (!Fused) {
            while (seg < 2 && at >= out.n[seg]) at -= out.n[seg++];
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = acc[r][m];
            for (int o = L / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, L);
            if (lane == 0) {
                if constexpr (Fused) xds[m * rows + local + r] = v;
                else out.y[seg][size_t(m) * out.n[seg] + at] = v;
            }
        }
    }
    if constexpr (Fused) {
        __syncthreads();
        local_top16<rows>(xds, parts, n);
    }
}

// One CTA per position. Hold up to four sorted-list heads per thread, loading
// only rank zero initially and advancing only the list that wins each round.
constexpr int kMergeThreads = 1024, kListsPerThread = 4;
__global__ void __launch_bounds__(kMergeThreads) fused_merge_kernel(
        const Pair* __restrict__ parts, int lists, int32_t* __restrict__ ids, float* __restrict__ vals) {
    __shared__ float sv[kMergeThreads / 64];
    __shared__ int si[kMergeThreads / 64];
    const int m = blockIdx.x, t = threadIdx.x;
    const Pair* p = parts + size_t(m) * lists * kTop;
    Pair heads[kListsPerThread];
    int next[kListsPerThread];
#pragma unroll
    for (int j = 0; j < kListsPerThread; ++j) {
        const int list = t + j * kMergeThreads;
        heads[j] = list < lists ? p[list] : invalid;
        next[j] = 1;
    }
    for (int rank = 0; rank < kTop; ++rank) {
        Pair best = invalid;
#pragma unroll
        for (int j = 0; j < kListsPerThread; ++j)
            if (better(heads[j].v, heads[j].id, best.v, best.id)) best = heads[j];
        block_best(best.v, best.id, sv, si);
        if (t == 0) {
            ids[m * kTop + rank] = best.id;
            vals[m * kTop + rank] = best.v;
        }
#pragma unroll
        for (int j = 0; j < kListsPerThread; ++j) {
            if (best.id != INT32_MAX && heads[j].id == best.id) {
                const int list = t + j * kMergeThreads;
                heads[j] = next[j] < kTop ? p[lists + size_t(list) * (kTop - 1) + next[j] - 1] : invalid;
                ++next[j];
            }
        }
    }
}

template <bool Sum>
__global__ void __launch_bounds__(256) rms_norm_quantize_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                                float* __restrict__ y_out, int8_t* __restrict__ q,
                                                                float* __restrict__ d, int32_t* xsum, int n_in,
                                                                float eps) {
    __shared__ float sw[4];
    const int r = int(blockIdx.x), tid = int(threadIdx.x);
    const float* xr = x + (size_t) r * n_in;
    float ss = 0.f;
    for (int i = tid; i < n_in; i += 256) {
        const float v = xr[i];
        ss += v * v;
    }
    for (int o = 32; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 64);
    if ((tid & 63) == 0) sw[tid >> 6] = ss;
    __syncthreads();
    const float total = sw[0] + sw[1] + sw[2] + sw[3];
    const float scale = rsqrtf(total / float(n_in) + eps);
    const int sub = tid >> 5, lane = tid & 31, blocks = n_in / 32;
    float* yr = y_out ? y_out + (size_t) r * n_in : nullptr;
    int8_t* qr = q + (size_t) r * n_in;
    for (int b = sub; b < blocks; b += 8) {
        const int c = b * 32 + lane;
        const float val = xr[c] * scale * w[c];
        if (yr) yr[c] = val;
        quant_block<Sum>(val, lane, qr + c, d, xsum, (size_t) r * blocks + b);
    }
}

template <bool Sum>
__global__ void __launch_bounds__(256) conv_quantize_kernel(const float* __restrict__ x, const float* __restrict__ dyn,
                                                            const float* __restrict__ base, int stage,
                                                            int8_t* __restrict__ q, float* __restrict__ d, int32_t* xsum) {
    const int r = int(blockIdx.y), b = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5), j = int(threadIdx.x) & 31;
    const int c = b * 32 + j;
    if (c >= kWidth) return;
    const int g = c / 16;
    const float* dn = dyn + (size_t) r * kDyn + stage * 2 * kGroups;
    float val = (base[(stage * 2 + 0) * kWidth + c] + dn[g]) * x[(size_t) r * kWidth + c];
    if (r) val += (base[(stage * 2 + 1) * kWidth + c] + dn[kGroups + g]) * x[(size_t) (r - 1) * kWidth + c];
    quant_block<Sum>(val, j, q + (size_t) r * kWidth + c, d, xsum, (size_t) r * (kWidth / 32) + b);
}

template <bool Sum>
__global__ void __launch_bounds__(256) swiglu_quantize_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                                              int n_in, int8_t* __restrict__ q, float* __restrict__ d,
                                                              int32_t* xsum) {
    const int r = int(blockIdx.y), b = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5), j = int(threadIdx.x) & 31;
    const int c = b * 32 + j;
    if (c >= n_in) return;
    const size_t at = (size_t) r * n_in + c;
    const float gv = g[at];
    const float val = (gv / (1.0f + expf(-gv))) * u[at];
    quant_block<Sum>(val, j, q + at, d, xsum, (size_t) r * (n_in / 32) + b);
}

// ---- attention v3: 32 keys a CTA, halves in LDS (~29 KB: two CTAs a CU), dot2, four independent sums a thread ----
constexpr int kChunk3 = 32;
__global__ void __launch_bounds__(256) attend3_part_kernel(const float* __restrict__ q, const float* __restrict__ kb,
                                                            const float* __restrict__ vb, const __half* __restrict__ kc,
                                                            const __half* __restrict__ vc, int64_t lo0, int n_ring,
                                                            int64_t pos, int chunks, float* part) {
    __shared__ __half2 qs[kQ][kHead / 2 + 1];
    __shared__ __half2 ks[kChunk3][kHead / 2 + 1];
    __shared__ __half2 vs[kChunk3][kHead / 2 + 1];
    __shared__ float p[kQ][kChunk3 + 1];
    __shared__ int kpos_ok[kChunk3];   // 1: a ring key, 2: a block key, 0: none
    __shared__ int kpos_lo[kChunk3];   // ring keys: position - pos (negative)
    const int g = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
    const int n_keys = n_ring + kBlock, k0 = c * kChunk3;
    for (int i = tid; i < kQ * (kHead / 2); i += 256) {
        const int qi = i / (kHead / 2), d2 = i % (kHead / 2), row = qi / 4, hh = qi % 4;
        const float2 f = reinterpret_cast<const float2*>(q + ((size_t) row * kHeads + g * 4 + hh) * kHead)[d2];
        qs[qi][d2] = __floats2half2_rn(f.x * kScale, f.y * kScale);   // the score scale folded into the query
    }
    for (int i = tid; i < kChunk3 * (kHead / 2); i += 256) {
        const int j = i / (kHead / 2), d2 = i % (kHead / 2), kk = k0 + j;
        __half2 kv = __floats2half2_rn(0.f, 0.f), vv = kv;
        if (kk < n_ring) {
            const size_t slot = (size_t) ((lo0 + kk) & (kWindow - 1));
            kv = reinterpret_cast<const __half2*>(kc + slot * kKey + g * kHead)[d2];
            vv = reinterpret_cast<const __half2*>(vc + slot * kKey + g * kHead)[d2];
        } else if (kk < n_keys) {
            const size_t at = ((size_t) (kk - n_ring) * kKv + g) * kHead;
            const float2 a = reinterpret_cast<const float2*>(kb + at)[d2], b = reinterpret_cast<const float2*>(vb + at)[d2];
            kv = __floats2half2_rn(a.x, a.y);
            vv = __floats2half2_rn(b.x, b.y);
        }
        ks[j][d2] = kv;
        vs[j][d2] = vv;
        if (d2 == 0) {
            kpos_ok[j] = kk < n_ring ? 1 : kk < n_keys ? 2 : 0;
            kpos_lo[j] = kk < n_ring ? (int) (lo0 + kk - pos) : 0;
        }
    }
    __syncthreads();
    const int qi = tid / 8, sub = tid % 8, row = qi / 4;
    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (int d2 = 0; d2 < kHead / 2; ++d2) {
        const __half2 a = qs[qi][d2];
#pragma unroll
        for (int j = 0; j < 4; ++j) s[j] = __builtin_amdgcn_fdot2(a, ks[sub + 8 * j][d2], s[j], false);
    }
    float m = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int kj = sub + 8 * j;
        // a ring key at position pos + lo (lo < 0) is within the window of row `row` when row - lo < kWindow
        const bool ok = kpos_ok[kj] == 2 || (kpos_ok[kj] == 1 && row - kpos_lo[kj] < kWindow);
        s[j] = ok ? s[j] : -INFINITY;
        m = fmaxf(m, s[j]);
    }
    for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 8));
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float e = s[j] == -INFINITY ? 0.f : __expf(s[j] - m);
        p[qi][sub + 8 * j] = e;
        l += e;
    }
    for (int o = 4; o > 0; o >>= 1) l += __shfl_xor(l, o, 8);
    __syncthreads();
    float acc[16] = {};
    for (int j = 0; j < kChunk3; ++j) {
        const float pj = p[qi][j];
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            const float2 v = __half22float2(vs[j][sub * 8 + d]);
            acc[2 * d] += pj * v.x;
            acc[2 * d + 1] += pj * v.y;
        }
    }
    float* o = part + ((size_t) (g * chunks + c) * kQ + qi) * (kHead + 2);
#pragma unroll
    for (int d = 0; d < 16; ++d) o[sub * 16 + d] = acc[d];
    if (sub == 0) {
        o[kHead] = m;
        o[kHead + 1] = l;
    }
}

__global__ void merge2_kernel(const int32_t* __restrict__ ids0, const float* __restrict__ vals0,
                              const int32_t* __restrict__ ids1, const float* __restrict__ vals1, int32_t* ids, float* vals) {
    if (threadIdx.x != 0) return;
    const int r = blockIdx.x;
    int a = 0, b = 0;
    for (int k = 0; k < kTop; ++k) {
        const bool first = b >= kTop || (a < kTop && better(vals0[r * kTop + a], ids0[r * kTop + a], vals1[r * kTop + b],
                                                             ids1[r * kTop + b]));
        ids[r * kTop + k] = first ? ids0[r * kTop + a] : ids1[r * kTop + b];
        vals[r * kTop + k] = first ? vals0[r * kTop + a++] : vals1[r * kTop + b++];
    }
}
__global__ void pack_top_kernel(const int32_t* ids, const float* vals, int offset, float* out) {
    const int r = blockIdx.x, k = threadIdx.x;
    if (k >= kTop) return;
    out[r * 2 * kTop + k] = vals[r * kTop + k];
    out[r * 2 * kTop + kTop + k] = __int_as_float(ids[r * kTop + k] + offset);
}
__global__ void unpack_top_kernel(const float* in, int32_t* ids, float* vals) {
    const int r = blockIdx.x, k = threadIdx.x;
    if (k >= kTop) return;
    vals[r * kTop + k] = in[r * 2 * kTop + k];
    ids[r * kTop + k] = __float_as_int(in[r * 2 * kTop + kTop + k]);
}
}  // namespace

void gather_taps(const Taps& taps, int first, int n, float* out, cudaStream_t st) {
    if (n <= 0) return;
    gather_kernel<<<dim3(n, 4), 256, 0, st>>>(taps, first, out);
}

void conv(const float* x, const float* dyn, const float* base, int stage, float* out, int rows, bool add,
          cudaStream_t st) {
    conv_kernel<<<dim3(div_up(kWidth, 256), rows), 256, 0, st>>>(x, dyn, base, stage, out, add);
}

void store(const float* k, const float* v, __half* kc, __half* vc, int64_t pos, int n, cudaStream_t st) {
    if (n <= 0) return;
    store_kernel<<<dim3(n, 2), 256, 0, st>>>(k, v, kc, vc, pos);
}

size_t attend_floats() { return (size_t) kKv * kMaxChunks * kQ * (kHead + 2); }

void attend(const float* q, const float* kb, const float* vb, const __half* kc, const __half* vc, int64_t lo,
            int64_t pos, float* scratch, float* out, cudaStream_t st) {
    const int64_t lo0 = lo > pos - (kWindow - 1) ? lo : pos - (kWindow - 1);
    const int n_ring = (int) (pos - (lo0 < pos ? lo0 : pos));
    const int chunks = div_up(n_ring + kBlock, kChunk);
    attend_part_kernel<<<dim3(kKv, chunks), 256, 0, st>>>(q, kb, vb, kc, vc, lo0, n_ring, pos, chunks, scratch);
    attend_add_kernel<<<kKv * kQ, kHead, 0, st>>>(scratch, chunks, out);
}

size_t topk_floats(int rows) { return (size_t) rows * kTopParts * 2 * kTop; }

void topk(const float* logits, int rows, int64_t ld, int valid, float* scratch, int32_t* ids, float* vals,
          cudaStream_t st) {
    topk_part_kernel<<<dim3(kTopParts, rows), 64, 0, st>>>(logits, ld, valid, scratch);
    topk_merge_kernel<<<rows, 256, 0, st>>>(scratch, ids, vals);
}

Mat Mat::slice(int first, int rows) const {
    const size_t at = (size_t) first * (n_in / 32);
    return {q + at * code_bytes(), static_cast<const uint8_t*>(d) + at * scale_bytes(), n_in, rows, kind, scale_f32};
}

void pack_q8_0(const void* src, int n_in, int n_out, uint8_t* q, float* d, cudaStream_t st) {
    const size_t total = (size_t) n_out * (n_in / 32);
    pack_kernel<<<div_up(total, 8), 256, 0, st>>>(static_cast<const uint8_t*>(src), q, d, total);
}

void pack_q4_0(const void* src, int n_in, int n_out, uint8_t* q, void* d, bool scale_f32, cudaStream_t st) {
    const size_t total = (size_t) n_out * (n_in / 32);
    const auto* bytes = static_cast<const uint8_t*>(src);
    if (scale_f32) pack4_kernel<<<div_up(total, 16), 256, 0, st>>>(bytes, q, static_cast<float*>(d), total);
    else pack4_kernel<<<div_up(total, 16), 256, 0, st>>>(bytes, q, static_cast<__half*>(d), total);
}

void quantize(const float* x, int n_in, int rows, int8_t* q, float* d, int32_t* xsum, cudaStream_t st) {
    const dim3 grid(div_up(n_in / 32, 8), rows);
    if (xsum) quantize_kernel<true><<<grid, 256, 0, st>>>(x, n_in, q, d, xsum);
    else quantize_kernel<false><<<grid, 256, 0, st>>>(x, n_in, q, d, nullptr);
}

static void gemv8_split(const Mat& w, const int8_t* xq, const float* xd, Outs out, int rows, cudaStream_t st) {
    static const bool fast = [] {
        const char* e = std::getenv("STRATA_DFLASH2_FAST");
        const bool on = e && e[0] == '1' && !e[1];
        if (on) std::fputs("strata-poly: STRATA_DFLASH2_FAST=1: 32B-block L8/R2 gemv8 and fused drafter quantize\n", stderr);
        return on;
    }();
    const auto* q = reinterpret_cast<const int8_t*>(w.q);
    const auto* d = static_cast<const float*>(w.d);
    if (fast && w.n_out >= 4096 && w.n_in % 32 == 0) {
        const dim3 grid(div_up(w.n_out, 64));
        switch (rows) {
#define DF2_G2(N) case N: gemv8_l8_r2_kernel<N><<<grid, 256, 0, st>>>(q, d, xq, xd, out, w.n_in, w.n_out); break;
            DF2_G2(1) DF2_G2(2) DF2_G2(3) DF2_G2(4) DF2_G2(5) DF2_G2(6) DF2_G2(7) DF2_G2(8)
#undef DF2_G2
            default: break;
        }
        return;
    }
    const dim3 grid(div_up(w.n_out, 16));
    switch (rows) {
#define DF2_G(N) case N: gemv8_kernel<N><<<grid, 256, 0, st>>>(q, d, xq, xd, out, w.n_in, w.n_out); break;
        DF2_G(1) DF2_G(2) DF2_G(3) DF2_G(4) DF2_G(5) DF2_G(6) DF2_G(7) DF2_G(8)
#undef DF2_G
        default: break;
    }
}

namespace {
constexpr int kHeadRowsPerCta = 64;   // L8/R2

template <int M, int L, int R, bool Fused = false>
void launch_q4(const Mat& w, const int8_t* xq, const float* xd, const int32_t* xsum, Outs out, Pair* parts,
               cudaStream_t st) {
    const dim3 grid(div_up(w.n_out, 256 / L * R));
    const auto* q = reinterpret_cast<const int4*>(w.q);
    if (w.scale_f32)
        planar_kernel<M, L, R, true, float, Fused><<<grid, 256, 0, st>>>(
            q, static_cast<const float*>(w.d), xq, xd, xsum, out, parts, w.n_in, w.n_out);
    else
        planar_kernel<M, L, R, true, __half, Fused><<<grid, 256, 0, st>>>(
            q, static_cast<const __half*>(w.d), xq, xd, xsum, out, parts, w.n_in, w.n_out);
}

template <int M>
void gemv4_split(const Mat& w, const int8_t* xq, const float* xd, const int32_t* xsum, Outs out, cudaStream_t st) {
    if (w.n_in == kWidth && w.n_out == 2 * 6912)             // gate|up
        launch_q4<M, 8, 4>(w, xq, xd, xsum, out, nullptr, st);
    else if (w.n_in == kQuery && w.n_out == kWidth)          // O
        launch_q4<M, 16, 2>(w, xq, xd, xsum, out, nullptr, st);
    else if (w.n_in == kWidth && w.n_out >= kQuery + 2 * kKey)  // QKV or private head
        launch_q4<M, 8, 2>(w, xq, xd, xsum, out, nullptr, st);
    else                                                    // Down, conv, selector, K|V
        launch_q4<M, 16, 1>(w, xq, xd, xsum, out, nullptr, st);
}
}  // namespace

#ifdef STRATA_DFLASH2_TEST
// Test control: same L16 floating-point order as Down, with two outputs per lane group instead of one.
void down_r2_for_test(const Mat& w, const int8_t* xq, const float* xd, const int32_t* xsum, float* y, cudaStream_t st) {
    const Outs out{{y, nullptr, nullptr}, {w.n_out, 0, 0}};
    launch_q4<kBlock, 16, 2>(w, xq, xd, xsum, out, nullptr, st);
}
#endif

void gemv(const Mat& w, const int8_t* xq, const float* xd, const int32_t* xsum, float* y, int rows, cudaStream_t st) {
    gemv_split(w, xq, xd, xsum, y, w.n_out, nullptr, 0, nullptr, 0, rows, st);
}

void gemv_split(const Mat& w, const int8_t* xq, const float* xd, const int32_t* xsum, float* y0, int n0, float* y1,
                int n1, float* y2, int n2, int rows, cudaStream_t st) {
    const Outs out{{y0, y1, y2}, {n0, n1 ? n1 : 1 << 30, n2 ? n2 : 1 << 30}};
    if (!w.q4()) return gemv8_split(w, xq, xd, out, rows, st);
    switch (rows) {
#define DF2_G4(N) case N: gemv4_split<N>(w, xq, xd, xsum, out, st); break;
        DF2_G4(1) DF2_G4(2) DF2_G4(3) DF2_G4(4) DF2_G4(5) DF2_G4(6) DF2_G4(7) DF2_G4(8)
#undef DF2_G4
        default: break;
    }
}

size_t head_top_bytes(int n_out) { return (size_t) div_up(n_out, kHeadRowsPerCta) * kGuess * kTop * sizeof(Pair); }

void head_top(const Mat& w, const int8_t* xq, const float* xd, const int32_t* xsum, void* scratch, int32_t* ids,
              float* vals, cudaStream_t st) {
    const int lists = div_up(w.n_out, kHeadRowsPerCta);
    auto* parts = static_cast<Pair*>(scratch);
    if (w.q4()) launch_q4<kGuess, 8, 2, true>(w, xq, xd, xsum, {}, parts, st);
    else planar_kernel<kGuess, 8, 2, false, float, true><<<lists, 256, 0, st>>>(
        reinterpret_cast<const int4*>(w.q), static_cast<const float*>(w.d), xq, xd, nullptr, {}, parts, w.n_in, w.n_out);
    fused_merge_kernel<<<kGuess, kMergeThreads, 0, st>>>(parts, lists, ids, vals);
}

void rms_norm_quantize(const float* x, const float* w, float* y_out, int8_t* q, float* d, int32_t* xsum, int n_in,
                       int rows, float eps, cudaStream_t st) {
    if (xsum) rms_norm_quantize_kernel<true><<<rows, 256, 0, st>>>(x, w, y_out, q, d, xsum, n_in, eps);
    else rms_norm_quantize_kernel<false><<<rows, 256, 0, st>>>(x, w, y_out, q, d, nullptr, n_in, eps);
}

void conv_quantize(const float* x, const float* dyn, const float* base, int stage, int rows, int8_t* q, float* d,
                   int32_t* xsum, cudaStream_t st) {
    const dim3 grid(div_up(kWidth / 32, 8), rows);
    if (xsum) conv_quantize_kernel<true><<<grid, 256, 0, st>>>(x, dyn, base, stage, q, d, xsum);
    else conv_quantize_kernel<false><<<grid, 256, 0, st>>>(x, dyn, base, stage, q, d, nullptr);
}

void swiglu_quantize(const float* g, const float* u, int n_in, int rows, int8_t* q, float* d, int32_t* xsum,
                     cudaStream_t st) {
    const dim3 grid(div_up(n_in / 32, 8), rows);
    if (xsum) swiglu_quantize_kernel<true><<<grid, 256, 0, st>>>(g, u, n_in, q, d, xsum);
    else swiglu_quantize_kernel<false><<<grid, 256, 0, st>>>(g, u, n_in, q, d, nullptr);
}

void attend3(const float* q, const float* kb, const float* vb, const __half* kc, const __half* vc, int64_t lo,
             int64_t pos, float* scratch, float* out, cudaStream_t st) {
    const int64_t lo0 = lo > pos - (kWindow - 1) ? lo : pos - (kWindow - 1);
    const int n_ring = (int) (pos - (lo0 < pos ? lo0 : pos));
    const int chunks = div_up(n_ring + kBlock, kChunk3);
    attend3_part_kernel<<<dim3(kKv, chunks), 256, 0, st>>>(q, kb, vb, kc, vc, lo0, n_ring, pos, chunks, scratch);
    attend_add_kernel<<<kKv * kQ, kHead, 0, st>>>(scratch, chunks, out);
}

void merge2(const int32_t* ids0, const float* vals0, const int32_t* ids1, const float* vals1, int rows, int32_t* ids,
            float* vals, cudaStream_t st) {
    merge2_kernel<<<rows, 64, 0, st>>>(ids0, vals0, ids1, vals1, ids, vals);
}
void pack_top(const int32_t* ids, const float* vals, int rows, int offset, float* out, cudaStream_t st) {
    pack_top_kernel<<<rows, 64, 0, st>>>(ids, vals, offset, out);
}
void unpack_top(const float* in, int rows, int32_t* ids, float* vals, cudaStream_t st) {
    unpack_top_kernel<<<rows, 64, 0, st>>>(in, ids, vals);
}

void select(const int32_t* ids, const float* vals, const float* h, Book pred, Book succ,
            int32_t anchor, float cap, float* edges, Guess* out, cudaStream_t st) {
    static const float inv_temp = [] {
        const char* e = std::getenv("STRATA_DFLASH2_PROB_TEMP");
        if (!e || !*e) return 1.f;
        const double t = std::strtod(e, nullptr);
        if (t > 0.1 && t <= 10.0) {
            std::fprintf(stderr, "strata-poly: STRATA_DFLASH2_PROB_TEMP=%.3f: calibrated DFlash2 walk probabilities\n",
                         t);
            return (float) (1.0 / t);
        }
        return 1.f;
    }();
    if (pred.kind == BookKind::kF16 && succ.kind == BookKind::kF16)
        edges_kernel<true><<<kGuess, 256, 0, st>>>(ids, vals, h, pred, succ, anchor, cap, edges);
    else
        edges_kernel<false><<<kGuess, 256, 0, st>>>(ids, vals, h, pred, succ, anchor, cap, edges);
    walk_kernel<<<1, 64, 0, st>>>(ids, edges, inv_temp, out);
}

}  // namespace strata::poly::dflash2
