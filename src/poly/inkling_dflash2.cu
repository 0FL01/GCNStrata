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
__global__ void __launch_bounds__(256) edges_kernel(const int32_t* __restrict__ ids, const float* __restrict__ vals,
                                                     const float* __restrict__ h, const __half* __restrict__ pred,
                                                     const __half* __restrict__ succ, int32_t anchor, float cap,
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
            ps[a][r] = __half2float(pred[(size_t) tok * kRank + r]) * ht[r];
        }
        qs[a][r] = __half2float(succ[(size_t) ids[t * kTop + a] * kRank + r]);
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
__global__ void pack_kernel(const uint8_t* __restrict__ src, int blocks, int8_t* q, float* d, size_t total) {
    const size_t b = (size_t) blockIdx.x * 8 + threadIdx.x / 32;   // a block of 32 values a 32-thread group
    if (b >= total) return;
    const int j = threadIdx.x % 32;
    const uint8_t* blk = src + b * 34;
    q[b * 32 + j] = (int8_t) blk[2 + j];
    if (j == 0) d[b] = __half2float(*reinterpret_cast<const __half*>(blk));
}
__global__ void quantize_kernel(const float* __restrict__ x, int n_in, int8_t* q, float* d) {
    // a 32-value block a 32-lane group: d = amax / 127, q = round(x / d)
    const int r = blockIdx.y, b = blockIdx.x * 8 + threadIdx.x / 32, j = threadIdx.x % 32;
    if (b * 32 >= n_in) return;
    const float v = x[(size_t) r * n_in + b * 32 + j];
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
    const float s = a / 127.f;
    q[(size_t) r * n_in + b * 32 + j] = (int8_t) (s > 0.f ? __float2int_rn(v / s) : 0);
    if (j == 0) d[(size_t) r * (n_in / 32) + b] = s;
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

__global__ void __launch_bounds__(256) rms_norm_quantize_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                                float* __restrict__ y_out, int8_t* __restrict__ q,
                                                                float* __restrict__ d, int n_in, float eps) {
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
    float* dr = d + (size_t) r * blocks;
    for (int b = sub; b < blocks; b += 8) {
        const int c = b * 32 + lane;
        const float val = xr[c] * scale * w[c];
        if (yr) yr[c] = val;
        float a = fabsf(val);
        for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
        const float s = a / 127.f;
        qr[c] = (int8_t) (s > 0.f ? __float2int_rn(val / s) : 0);
        if (lane == 0) dr[b] = s;
    }
}

__global__ void __launch_bounds__(256) conv_quantize_kernel(const float* __restrict__ x, const float* __restrict__ dyn,
                                                            const float* __restrict__ base, int stage,
                                                            int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = int(blockIdx.y), b = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5), j = int(threadIdx.x) & 31;
    const int c = b * 32 + j;
    if (c >= kWidth) return;
    const int g = c / 16;
    const float* dn = dyn + (size_t) r * kDyn + stage * 2 * kGroups;
    float val = (base[(stage * 2 + 0) * kWidth + c] + dn[g]) * x[(size_t) r * kWidth + c];
    if (r) val += (base[(stage * 2 + 1) * kWidth + c] + dn[kGroups + g]) * x[(size_t) (r - 1) * kWidth + c];
    float a = fabsf(val);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
    const float s = a / 127.f;
    q[(size_t) r * kWidth + c] = (int8_t) (s > 0.f ? __float2int_rn(val / s) : 0);
    if (j == 0) d[(size_t) r * (kWidth / 32) + b] = s;
}

__global__ void __launch_bounds__(256) swiglu_quantize_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                                              int n_in, int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = int(blockIdx.y), b = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5), j = int(threadIdx.x) & 31;
    const int c = b * 32 + j;
    if (c >= n_in) return;
    const size_t at = (size_t) r * n_in + c;
    const float gv = g[at];
    const float val = (gv / (1.0f + expf(-gv))) * u[at];
    float a = fabsf(val);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
    const float s = a / 127.f;
    q[at] = (int8_t) (s > 0.f ? __float2int_rn(val / s) : 0);
    if (j == 0) d[(size_t) r * (n_in / 32) + b] = s;
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

void pack_q8_0(const void* q8_0, int n_in, int n_out, int8_t* q, float* d, cudaStream_t st) {
    const size_t total = (size_t) n_out * (n_in / 32);
    pack_kernel<<<(unsigned) ((total + 7) / 8), 256, 0, st>>>(static_cast<const uint8_t*>(q8_0), n_in / 32, q, d, total);
}

void quantize(const float* x, int n_in, int rows, int8_t* q, float* d, cudaStream_t st) {
    quantize_kernel<<<dim3(div_up(n_in / 32, 8), rows), 256, 0, st>>>(x, n_in, q, d);
}

void gemv8(const Mat8& w, const int8_t* xq, const float* xd, float* y, int rows, cudaStream_t st) {
    gemv8_split(w, xq, xd, y, w.n_out, nullptr, 0, nullptr, 0, rows, st);
}

void gemv8_split(const Mat8& w, const int8_t* xq, const float* xd, float* y0, int n0, float* y1, int n1, float* y2,
                 int n2, int rows, cudaStream_t st) {
    static const bool fast = [] {
        const char* e = std::getenv("STRATA_DFLASH2_FAST");
        const bool on = e && e[0] == '1' && !e[1];
        if (on) std::fputs("strata-poly: STRATA_DFLASH2_FAST=1: 32B-block L8/R2 gemv8 and fused drafter quantize\n", stderr);
        return on;
    }();
    const Outs out{{y0, y1, y2}, {n0, n1 ? n1 : 1 << 30, n2 ? n2 : 1 << 30}};
    if (fast && w.n_out >= 4096 && w.n_in % 32 == 0) {
        const dim3 grid(div_up(w.n_out, 64));
        switch (rows) {
#define DF2_G2(N) case N: gemv8_l8_r2_kernel<N><<<grid, 256, 0, st>>>(w.q, w.d, xq, xd, out, w.n_in, w.n_out); break;
            DF2_G2(1) DF2_G2(2) DF2_G2(3) DF2_G2(4) DF2_G2(5) DF2_G2(6) DF2_G2(7) DF2_G2(8)
#undef DF2_G2
            default: break;
        }
        return;
    }
    const dim3 grid(div_up(w.n_out, 16));
    switch (rows) {
#define DF2_G(N) case N: gemv8_kernel<N><<<grid, 256, 0, st>>>(w.q, w.d, xq, xd, out, w.n_in, w.n_out); break;
        DF2_G(1) DF2_G(2) DF2_G(3) DF2_G(4) DF2_G(5) DF2_G(6) DF2_G(7) DF2_G(8)
#undef DF2_G
        default: break;
    }
}

void rms_norm_quantize(const float* x, const float* w, float* y_out, int8_t* q, float* d, int n_in, int rows, float eps,
                       cudaStream_t st) {
    rms_norm_quantize_kernel<<<rows, 256, 0, st>>>(x, w, y_out, q, d, n_in, eps);
}

void conv_quantize(const float* x, const float* dyn, const float* base, int stage, int rows, int8_t* q, float* d,
                   cudaStream_t st) {
    conv_quantize_kernel<<<dim3(div_up(kWidth / 32, 8), rows), 256, 0, st>>>(x, dyn, base, stage, q, d);
}

void swiglu_quantize(const float* g, const float* u, int n_in, int rows, int8_t* q, float* d, cudaStream_t st) {
    swiglu_quantize_kernel<<<dim3(div_up(n_in / 32, 8), rows), 256, 0, st>>>(g, u, n_in, q, d);
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

void select(const int32_t* ids, const float* vals, const float* h, const __half* pred, const __half* succ,
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
    edges_kernel<<<kGuess, 256, 0, st>>>(ids, vals, h, pred, succ, anchor, cap, edges);
    walk_kernel<<<1, 64, 0, st>>>(ids, edges, inv_temp, out);
}

}  // namespace strata::poly::dflash2
