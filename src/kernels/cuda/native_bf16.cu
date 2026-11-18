#include "strata/kernels/bf16_gemv.hpp"
#include "s26_tsum.cuh"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>
#include "route_v2.cuh"
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

// The FP32-activation MMVF implementation below is adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/{mmvf.cu,common.cuh}.
// Scope: ordinary contiguous BF16 matrix, one FP32 activation vector, no fusion/ids/channels.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
__device__ __forceinline__ float mmvf_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

template <int BLOCK_SIZE>
__global__ void bf16_f32_mmvf_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                    float* __restrict__ y, int n_in) {
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const float2* inputs2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if (t < 32) partials[t] = 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float2 input = inputs2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        acc = __fmaf_rn(f32_from_bf16((uint16_t) weight), input.x, acc);
        acc = __fmaf_rn(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc);
    }
    acc = mmvf_warp_sum(acc);
    if constexpr (BLOCK_SIZE > 32) {
        // All lanes have the same reduced value; one store avoids a same-value shared-memory race.
        if ((t & 31) == 0) partials[t / 32] = acc;
        __syncthreads();
        if (t < 32) acc = mmvf_warp_sum(partials[t]);
    }
    if (t == 0) y[blockIdx.x] = acc;
}

// the same kernel for up to 8 activation rows - the weight row is read ONCE and every
// token keeps its own accumulator with exactly the single-row kernel's order (pairs, two ordered FMAs, the same warp
// and block reductions), so each output is bit-identical to a bf16_f32_mmvf_kernel launch of its own.
template <int BLOCK_SIZE, int NT, bool EXACT_T = true>
__global__ void bf16_f32_mmvf_multi_kernel(const float* x_, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* y_, int64_t ldy, int n_in, int n_tok) {
    const float* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    float* STRATA_PDL_RESTRICT y = y_;
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    __shared__ float partials[NT][32];
    if (t < 32) {
#pragma unroll
        for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
    }
    __syncthreads();
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    // PDL (pdl.hpp): this thread's first PRE weight pairs are loaded before waiting for the activations
    constexpr int PRE = kPdlPrefetch ? 8 : 0;
    uint32_t wpre[PRE > 0 ? PRE : 1];
#pragma unroll
    for (int q = 0; q < PRE; ++q) {
        const int pair = t + q * BLOCK_SIZE;
        wpre[q] = pair < n_in / 2 ? __ldg(weights2 + pair) : 0u;
    }
    pdl_wait();
    int q = 0;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE, ++q) {
        const uint32_t weight = q < PRE ? wpre[q] : __ldg(weights2 + pair);
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (EXACT_T || k < n_tok) {
                const float2 input = __ldg(reinterpret_cast<const float2*>(x + (size_t) k * ldx) + pair);
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k)
        if (EXACT_T || k < n_tok) acc[k] = mmvf_warp_sum(acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (EXACT_T || k < n_tok) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k)
                if (EXACT_T || k < n_tok) acc[k] = mmvf_warp_sum(partials[k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (EXACT_T || k < n_tok) y[(size_t) k * ldy + blockIdx.x] = acc[k];
}

// S25 (STRATA_MMVF_ROWS=1): RPB output rows per block. Each block read its row's weights once but all T activation
// rows again (n_out blocks x T x n_in floats through L2 - the part that grew with T); here a block reads the
// activations once for RPB rows. Per output, thread t still walks pairs t, t + BLOCK_SIZE, ... with the same two
// ordered FMAs and the same warp and block reductions: bit-identical to bf16_f32_mmvf_multi_kernel.
// S26 STRATA_LFUSE=1 (AUX): one more block computes row 0 of w_aux into y_aux[k * ldy_aux] - the same per-output
// code, so each aux output is bitwise what its own bf16_f32_mmvf_multi_kernel launch gave (the shared expert's gate)
// the body: block `bx` of `gdim` (the kernel's blockIdx.x / gridDim.x; the gfx906 fused router chain runs it too)
template <int BLOCK_SIZE, int NT, int RPB, bool TS, bool AUX>
__device__ __forceinline__ void mmvf_rows_body(const int bx, const int gdim, const float* __restrict__ x, int64_t ldx,
                                               const uint16_t* __restrict__ w, float* __restrict__ y, int64_t ldy, int n_in,
                                               int n_out, int n_tok, const uint16_t* __restrict__ w_aux,
                                               float* __restrict__ y_aux, int64_t ldy_aux) {
    const int t = threadIdx.x;
    int o0 = bx * RPB;
    if constexpr (AUX) {
        if (bx == gdim - 1) { w = w_aux; y = y_aux; ldy = ldy_aux; n_out = 1; o0 = 0; }
    }
    __shared__ float partials[RPB][NT][32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) partials[r][k][t] = 0.0f;
        __syncthreads();
    }
    float acc[RPB][NT];
#pragma unroll
    for (int r = 0; r < RPB; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        float2 in[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) in[k] = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[pair];
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            if (o0 + r >= n_out) break;
            const uint32_t weight = reinterpret_cast<const uint32_t*>(w + (size_t) (o0 + r) * n_in)[pair];
            const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                if (k < n_tok) {
                    acc[r][k] = __fmaf_rn(w0, in[k].x, acc[r][k]);
                    acc[r][k] = __fmaf_rn(w1, in[k].y, acc[r][k]);
                }
            }
        }
    }
    if constexpr (TS) {   // S26 STRATA_TSUM=1: the RPB x NT warp sums (both stages) as transposed butterflies, bitwise the same
        constexpr int V = RPB * NT, P = s26ts::pow2_ceil(V);
        const int lane = t & 31, j = s26ts::tsum_token<P>(lane);
        const bool own = lane == s26ts::tsum_lane<P>(j) && j < V;
        float v[P];
#pragma unroll
        for (int q = 0; q < P; ++q) v[q] = q < V ? acc[q / NT][q % NT] : 0.0f;
        float sum = s26ts::tsum<P>(v, lane);
        if constexpr (BLOCK_SIZE > 32) {
            if (own) partials[j / NT][j % NT][t / 32] = sum;
            __syncthreads();
            if (t >= 32) return;
#pragma unroll
            for (int q = 0; q < P; ++q) v[q] = q < V ? partials[q / NT][q % NT][t] : 0.0f;
            sum = s26ts::tsum<P>(v, lane);
        }
        const int r = j / NT, k = j % NT;
        if (own && o0 + r < n_out && k < n_tok) y[(size_t) k * ldy + o0 + r] = sum;
        return;
    }
#pragma unroll
    for (int r = 0; r < RPB; ++r)
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[r][k] = mmvf_warp_sum(acc[r][k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) partials[r][k][t / 32] = acc[r][k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int r = 0; r < RPB; ++r)
#pragma unroll
                for (int k = 0; k < NT; ++k) acc[r][k] = mmvf_warp_sum(partials[r][k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int r = 0; r < RPB; ++r)
            if (o0 + r < n_out)
#pragma unroll
                for (int k = 0; k < NT; ++k)
                    if (k < n_tok) y[(size_t) k * ldy + o0 + r] = acc[r][k];
}
template <int BLOCK_SIZE, int NT, int RPB, bool TS = false, bool AUX = false>
__global__ void bf16_f32_mmvf_rows_kernel(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* __restrict__ y, int64_t ldy, int n_in, int n_out, int n_tok,
                                          const uint16_t* __restrict__ w_aux = nullptr, float* __restrict__ y_aux = nullptr,
                                          int64_t ldy_aux = 0) {
    mmvf_rows_body<BLOCK_SIZE, NT, RPB, TS, AUX>((int) blockIdx.x, (int) gridDim.x, x, ldx, w, y, ldy, n_in, n_out, n_tok,
                                                 w_aux, y_aux, ldy_aux);
}

#if defined(STRATA_HIP_GFX906)
// ---- gfx906 (STRATA_GFX906_ROUTER_CHAIN, default on): the FFN router chain of a verify window's layer in ONE launch -
// the 512 x 2560 router GEMV (the T1 / T2 rows kernel's 128 blocks of 4 rows), then in the LAST block to finish (a
// fenced counter in the caller's per-instance word) the top-10 of each token (route_v2_body, a wave per token) and the
// doorbell publish (doorbell_publish_kernel / doorbell_publish_res_kernel's copies and ring).  Every block runs those
// kernels' bodies on the same data in the same order, so the logits, ids, weights and published bytes are theirs bit
// for bit; what goes is two launches and their gaps on the layer's serial path (router -> bell -> host plan).
template <int NT, int PUB>   // PUB 1: publish x only if a routed expert is not resident (doorbell_publish_res); 2: always (+ weights)
__global__ void __launch_bounds__(256) router_chain_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                                           float* __restrict__ logits, int32_t* __restrict__ ids,
                                                           float* __restrict__ weights, const int32_t* __restrict__ d_res,
                                                           int n_expert, float* x_out, int32_t* ids_out, float* w_out,
                                                           uint32_t* seq, unsigned* cnt) {
    constexpr int n = NT * 2560, k = NT * 10;
    if constexpr (PUB == 2) {
        // publish 2 copies x whatever the routing: every block its slice first (one block alone was the slower copy),
        // fenced to the system before the block counts, so the ring below follows every byte of it
        constexpr int S = (n + 127) / 128;
        for (int i = (int) blockIdx.x * S + (int) threadIdx.x; i < ((int) blockIdx.x + 1) * S && i < n; i += blockDim.x)
            x_out[i] = x[i];
        __threadfence_system();
    }
    mmvf_rows_body<256, NT, 4, false, false>((int) blockIdx.x, (int) gridDim.x, x, 2560, w, logits, 512, 2560, 512, NT,
                                             nullptr, nullptr, 0);
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(cnt, 1u) == gridDim.x - 1;
        if (last) atomicExch(cnt, 0u);   // every block has counted: free for the next layer's launch
    }
    __syncthreads();
    if (!last) return;
    __threadfence();   // the other blocks' logits
    const int wv = (int) threadIdx.x >> 6;
    if (wv < NT) rv2::route_v2_body(logits + wv * 512, ids + wv * 10, weights + wv * 10, (int) threadIdx.x & 63);
    __threadfence();
    __syncthreads();
    if constexpr (PUB == 2) {   // doorbell_publish_kernel (x: above, by every block)
        if ((int) threadIdx.x < k) {
            ids_out[threadIdx.x] = ids[threadIdx.x];
            w_out[threadIdx.x] = weights[threadIdx.x];
        }
        __threadfence_system();
        __syncthreads();
    } else {                    // doorbell_publish_res_kernel
        int any_miss = 0;
        if ((int) threadIdx.x < k) {
            const int32_t id = ids[threadIdx.x];
            ids_out[threadIdx.x] = id;
            if (d_res == nullptr || id < 0 || id >= n_expert || d_res[id] < 0) any_miss = 1;
        }
        if (__syncthreads_or(any_miss)) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
            __threadfence_system();
        } else if ((int) threadIdx.x < k) {
            __threadfence_system();
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
        __threadfence_system();
    }
}
#endif

template <int B>
void launch_rows(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int n_in, int n_out, int n_tok,
                 cudaStream_t st, const uint16_t* w_aux = nullptr, float* y_aux = nullptr, int64_t ldy_aux = 0) {
    constexpr int RPB = 4;
    const unsigned nb = (unsigned) ((n_out + RPB - 1) / RPB);
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (w_aux != nullptr) {   // S26 STRATA_LFUSE: + one aux row block
#define S26_AUX(TSV) \
        if (n_tok <= 4) bf16_f32_mmvf_rows_kernel<B, 4, RPB, TSV, true><<<nb + 1, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok, w_aux, y_aux, ldy_aux); \
        else bf16_f32_mmvf_rows_kernel<B, 8, RPB, TSV, true><<<nb + 1, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok, w_aux, y_aux, ldy_aux);
        if (ts) { S26_AUX(true) } else { S26_AUX(false) }
#undef S26_AUX
        return;
    }
    if (ts) {
        if (n_tok <= 4) bf16_f32_mmvf_rows_kernel<B, 4, RPB, true><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
        else bf16_f32_mmvf_rows_kernel<B, 8, RPB, true><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
        return;
    }
    if (n_tok <= 4) bf16_f32_mmvf_rows_kernel<B, 4, RPB><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
    else bf16_f32_mmvf_rows_kernel<B, 8, RPB><<<nb, B, 0, st>>>(x, ldx, w, y, ldy, n_in, n_out, n_tok);
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

}  // namespace

bool bf16_gemv_fp32_mmvf_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                   int64_t n_in, int64_t n_out, int n_tok, const uint16_t* w_aux, float* y_aux,
                                   int64_t ldy_aux, void* stream) {
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    // exactly the conditions under which both calls would take the rows / multi kernels at the same block size
    if (!rows || n_out < 64 || n_tok < 2 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || w_aux == nullptr || y_aux == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        return false;
    const cudaStream_t st = (cudaStream_t) stream;
    const int ni = (int) n_in, no = (int) n_out;
    switch (mmvf_block_size(n_in)) {
        case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
    }
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi_aux launch: ") + cudaGetErrorString(result));
    return true;
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    const cudaStream_t st = (cudaStream_t) stream;
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    if (rows && n_out >= 64) {
#if defined(STRATA_HIP_GFX906)
        // Exact router T2: omit only the two inactive columns of the existing NT=4 rows kernel.
        // Live output pair/FMA/XOR32 order is unchanged. Other shapes, strides and TSUM keep baseline.
        if (n_tok == 2 && n_in == 2560 && n_out == 512 && ldx == n_in && ldy == n_out) {
            const char* exact_t2 = std::getenv("STRATA_GFX906_ROUTER_T2_EXACT");
            static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
            if (exact_t2 && exact_t2[0] == '1' && exact_t2[1] == 0 && !ts) {
                bf16_f32_mmvf_rows_kernel<256, 2, 4><<<128, 256, 0, st>>>(x, ldx, w, y, ldy, 2560, 512, 2);
                const cudaError_t result = cudaGetLastError();
                if (result != cudaSuccess)
                    throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf router T2 exact launch: ") + cudaGetErrorString(result));
                return;
            }
        }
#endif
        const int ni = (int) n_in, no = (int) n_out;
        switch (mmvf_block_size(n_in)) {
            case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
        }
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
        return;
    }
#define STRATA_MMVF_M(N) case N: \
    switch (n_tok) { \
        case 1: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 1, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 2: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 2, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 3: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 3, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 4: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 4, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 5: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 5, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 6: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 6, true>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        default: launch_pdl(bf16_f32_mmvf_multi_kernel<N, 8, false>, dim3((unsigned) n_out), dim3(N), 0, st, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
    } break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_M(32); STRATA_MMVF_M(64); STRATA_MMVF_M(96); STRATA_MMVF_M(128);
        STRATA_MMVF_M(160); STRATA_MMVF_M(192); STRATA_MMVF_M(224); STRATA_MMVF_M(256);
    }
#undef STRATA_MMVF_M
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    const cudaStream_t st = (cudaStream_t) stream;
#if defined(STRATA_HIP_GFX906)
    // Opt-in exact router T1 shape. Reuse the existing four-output-row kernel:
    // each output keeps the same pair sequence, two ordered FMAs, and XOR32 reductions.
    // Respect disabling row reuse or selecting the alternate TSUM arithmetic path.
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    const char* router_t1_rows = std::getenv("STRATA_GFX906_ROUTER_T1_ROWS");
    if (router_t1_rows && router_t1_rows[0] == '1' && router_t1_rows[1] == 0 &&
        rows && !ts && n_in == 2560 && n_out == 512) {
        bf16_f32_mmvf_rows_kernel<256, 1, 4><<<128, 256, 0, st>>>(x, n_in, w, y, n_out, 2560, 512, 1);
        const cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf router T1 rows launch: ") + cudaGetErrorString(result));
        return;
    }
#endif
#define STRATA_MMVF_CASE(N) case N: \
    bf16_f32_mmvf_kernel<N><<<(unsigned) n_out, N, 0, st>>>(x, w, y, (int) n_in); break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_CASE(32);
        STRATA_MMVF_CASE(64);
        STRATA_MMVF_CASE(96);
        STRATA_MMVF_CASE(128);
        STRATA_MMVF_CASE(160);
        STRATA_MMVF_CASE(192);
        STRATA_MMVF_CASE(224);
        STRATA_MMVF_CASE(256);
    }
#undef STRATA_MMVF_CASE
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + cudaGetErrorString(result));
}



void bf16_gemv_fp32_mmvf_cols(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int ncols,
                              void* stream) {
    // columns contiguous: the existing multi-row kernel (weight read once, each output bitwise its one-row call),
    // up to 8 rows per launch
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = ncols - c0 < 8 ? ncols - c0 : 8;
        if (nc == 1) bf16_gemv_fp32_mmvf(x + (size_t) c0 * n_in, w, y + (size_t) c0 * n_out, n_in, n_out, stream);
        else bf16_gemv_fp32_mmvf_multi(x + (size_t) c0 * n_in, n_in, w, y + (size_t) c0 * n_out, n_out, n_in, n_out, nc,
                                       stream);
    }
}


bool router_chain_fused(const float* x, const uint16_t* w, float* logits, int32_t* ids, float* weights, int n_tok,
                        int publish, const int32_t* d_res, int n_expert, float* x_out, int32_t* ids_out, float* w_out,
                        uint32_t* seq, unsigned* cnt, void* stream) {
#if defined(STRATA_HIP_GFX906)
    // only where the separate launches are the T1 / T2 rows kernels and the v2 top-10 (the bits this kernel reproduces)
    static const bool on = [] { const char* e = std::getenv("STRATA_GFX906_ROUTER_CHAIN"); return e == nullptr || std::atoi(e) != 0; }();
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    static const bool t1 = [] { const char* v = std::getenv("STRATA_GFX906_ROUTER_T1_ROWS"); return v && v[0] == '1' && v[1] == 0; }();
    static const bool t2 = [] { const char* v = std::getenv("STRATA_GFX906_ROUTER_T2_EXACT"); return v && v[0] == '1' && v[1] == 0; }();
    if (!on || !rows || ts || !native_router_v2_enabled() || cnt == nullptr || seq == nullptr || x_out == nullptr ||
        ids_out == nullptr || (publish == 2 && w_out == nullptr) || (publish != 1 && publish != 2) ||
        !((n_tok == 1 && t1) || (n_tok == 2 && t2)))
        return false;
    const cudaStream_t st = (cudaStream_t) stream;
    if (n_tok == 1) {
        if (publish == 2) router_chain_kernel<1, 2><<<128, 256, 0, st>>>(x, w, logits, ids, weights, d_res, n_expert, x_out, ids_out, w_out, seq, cnt);
        else router_chain_kernel<1, 1><<<128, 256, 0, st>>>(x, w, logits, ids, weights, d_res, n_expert, x_out, ids_out, w_out, seq, cnt);
    } else {
        if (publish == 2) router_chain_kernel<2, 2><<<128, 256, 0, st>>>(x, w, logits, ids, weights, d_res, n_expert, x_out, ids_out, w_out, seq, cnt);
        else router_chain_kernel<2, 1><<<128, 256, 0, st>>>(x, w, logits, ids, weights, d_res, n_expert, x_out, ids_out, w_out, seq, cnt);
    }
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(std::string("router_chain_fused launch: ") + cudaGetErrorString(result));
    return true;
#else
    (void) x; (void) w; (void) logits; (void) ids; (void) weights; (void) n_tok; (void) publish; (void) d_res; (void) n_expert;
    (void) x_out; (void) ids_out; (void) w_out; (void) seq; (void) cnt; (void) stream;
    return false;
#endif
}
}  // namespace strata::kernels
