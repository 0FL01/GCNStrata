// Included by poly/mmq.cu, after the pinned ggml MMQ headers.
#pragma once

#include "inkling_prefill.hpp"

namespace strata::poly::inkling_prefill {
namespace {

constexpr int threads = 256, wave_size = 64, cta_waves = threads / wave_size;
constexpr int offsets[classes] = {Scratch::list_offset(0), Scratch::list_offset(1), Scratch::list_offset(2)};

__global__ void mask_picks(Scratch* s, const int32_t* ids, int r, SlotOrder order) {
    if (s->summary.invalid) return;
    const int e = blockIdx.x, part = blockIdx.y, lane = threadIdx.x % wave_size, wave = threadIdx.x / wave_size;
    const int expert = order.expert[e];
    __shared__ int totals[cta_waves];
    int n = 0;
    for (int w = wave; w < waves; w += cta_waves) {
        const int i = part * slice + w * wave_size + lane;
        const uint64_t bits = __ballot(i < r && ids[i] == expert);
        if (lane == 0) s->mask[e][part][w] = bits;
        n += __popcll(bits);
    }
    if (lane == 0) totals[wave] = n;
    __syncthreads();
    if (threadIdx.x == 0) s->count[e][part] = totals[0] + totals[1] + totals[2] + totals[3];
}

__global__ void make_plan(Scratch* s, int32_t* cb, int parts, int tokens, bool lists) {
    if (s->summary.invalid) return;
    const int e = threadIdx.x;
    int n = 0;
    for (int p = 0; p < parts; ++p) {
        s->first[e][p] = n;
        n += s->count[e][p];
    }
    s->summary.count[e] = n;
    __shared__ int counts[experts], list_first[experts];
    counts[e] = n;
    __syncthreads();
    if (e == 0) {
        int total = 0;
        for (int v = 0; v < experts; ++v) {
            total += counts[v];
            if (counts[v] > tokens) s->summary.invalid = 1;
        }
        if (total != tokens * picks) s->summary.invalid = 1;
        if (!s->summary.invalid) {
            int row = 0;
            for (int c = 0; c < classes; ++c) {
                int most = 0, used = 0;
                int32_t* b = cb + c * (experts + 1);
                for (int v = 0; v < experts; ++v) {
                    b[v] = row;
                    const int nn = counts[v];
                    const int cls = nn <= cutoffs[0] ? 0 : nn <= cutoffs[1] ? 1 : 2;
                    if (nn == 0 || c != cls) continue;
                    s->begin[v] = row;
                    list_first[v] = used;
                    row += nn;
                    const int j = tiles[c];
                    used += (nn + j - 1) / j;
                    most = most < nn ? nn : most;
                }
                b[experts] = row;
                s->summary.most[c] = most;
                s->used[c] = used;
            }
        }
    }
    __syncthreads();
    if (!lists || s->summary.invalid || n == 0) return;
    const int c = n <= cutoffs[0] ? 0 : n <= cutoffs[1] ? 1 : 2;
    const int j = tiles[c], row = s->begin[e], at = offsets[c] + list_first[e];
    for (int k = 0; k < n; k += j) s->list[at + k / j] = {e, row + k, n - k};
}

__global__ void scatter_picks(const Scratch* s, bool inverse, int32_t* tok, int32_t* place) {
    const int e = blockIdx.x, part = blockIdx.y, lane = threadIdx.x % wave_size, wave = threadIdx.x / wave_size;
    if (s->summary.invalid || !s->count[e][part]) return;
    __shared__ uint64_t masks[waves];
    if (threadIdx.x < waves) masks[threadIdx.x] = s->mask[e][part][threadIdx.x];
    __syncthreads();
    __shared__ int prior[waves];
    if (threadIdx.x < waves) {
        int n = 0;
        for (int w = 0; w < threadIdx.x; ++w) n += __popcll(masks[w]);
        prior[threadIdx.x] = n;
    }
    __syncthreads();
    const int begin = s->begin[e] + s->first[e][part];
    const uint64_t bit = uint64_t(1) << lane;
    for (int w = wave; w < waves; w += cta_waves) {
        const uint64_t mask = masks[w];
        if (mask & bit) {
            const int i = part * slice + w * wave_size + lane;
            const int row = begin + prior[w] + __popcll(mask & (bit - 1));
            if (inverse) tok[i] = row;
            else tok[row] = i / picks;
            place[row] = i;
        }
    }
}

// The compact grid replaces only the stock (expert, J-column tile) enumeration. Preserve J,
// the pinned ggml tile body, and the optional stock Q4 down-tail body for the same J.
template<ggml_type T, int J, int K, int M, bool Tail = false>
__global__ __launch_bounds__(threads, ggml_cuda_mmq_get_occupancy(T, J, false))
void compact_mmq(const char* x, const int* y, const int32_t* ids, const Task* tasks, const int32_t* used,
                 float* dst, int stride_expert, int ncols_y) {
#if defined(__gfx906__)
    constexpr auto cfg = ggml_cuda_mmq_get_config_gcn(T, J, false);
    static_assert(cfg.I == 64 && cfg.nthreads == threads && cfg.K_vram == 256 && !cfg.stream_k);
    static_assert(!cfg.use_mma_data_layout() && QK8_1_MMQ == 128);
    constexpr int stride_row = K / 32, words = sizeof(block_q8_1_mmq) / sizeof(int);
    if (blockIdx.y >= (unsigned) *used) return;
    const Task task = tasks[blockIdx.y];
    const int row = blockIdx.x * 64, lane = threadIdx.y * wave_size + threadIdx.x;
    extern __shared__ int shared[];
    if (lane < J) shared[lane] = lane < task.remaining ? ids[task.col + lane] : 0;
    __syncthreads();
    y += task.col * words;
    const int at = task.slot * stride_expert + row * stride_row;
    if constexpr (!Tail) {
        mul_mat_q_process_tile<T, J, false, false>(x, at, y, shared, dst + row, nullptr, nullptr,
                                                 stride_row, ncols_y, M, 63, task.remaining - 1, 0, stride_row);
    } else {
        static_assert(K == ff && M == width);
        int* ty = shared + J;
        int* tx = ty + GGML_PAD(J * MMQ_TILE_Y_K, threads);
        constexpr auto load = ggml_cuda_mmq_get_load_tiles<T, J, false>();
        constexpr auto dot = ggml_cuda_mmq_get_vec_dot<T, J, false>();
        constexpr auto write = ggml_cuda_mmq_get_write_back<T, J, false>();
        float sum[J * 64 / threads] = {};
        const auto half_tile = [&](int k, int half) {
            const int* src = y + ncols_y * (k / QK8_1_MMQ + half) * words;
#pragma unroll
            for (int i = 0; i < J * MMQ_TILE_Y_K; i += threads) ty[i + lane] = src[i + lane];
            __syncthreads();
            dot(tx, ty, sum, half * MMQ_TILE_NE_K);
            __syncthreads();
        };
        for (int kb = 0; kb < stride_row; kb += 8) {
            load(x, tx, at + kb, 63, stride_row);
            half_tile(kb * 32, 0);
            if (kb * 32 + QK8_1_MMQ < K) half_tile(kb * 32, 1);
        }
        write(sum, shared, dst + row, nullptr, M, 63, task.remaining - 1);
    }
#endif
}

}  // namespace

}  // namespace strata::poly::inkling_prefill
