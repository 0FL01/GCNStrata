// Included by poly/mmq.cu: grouped Inkling TP2 Q4 down on gfx906.
#pragma once
#include <climits>

namespace strata::poly {
namespace {

constexpr int kQ4DownM = 2304, kQ4DownK = 576, kQ4DownI = 64, kQ4DownThreads = 256, kQ4Experts = 128;
constexpr int kQ4Words = sizeof(block_q8_1_mmq) / sizeof(int);

// Keep ggml's weight loader and the dot order for K=0..639. The Q8_1 half starting at K=640 is padding.
template <ggml_type T, int J>
__global__ __launch_bounds__(kQ4DownThreads, ggml_cuda_mmq_get_occupancy(T, J, false))
void q4_down_tail_gfx906(const char* x, const int* y, const int32_t* ids, const int32_t* bounds, float* dst,
                        int stride_expert, int ncols_y, int dst_stride) {
#if defined(__gfx906__)
    constexpr auto cfg = ggml_cuda_mmq_get_config_gcn(T, J, false);
    static_assert(cfg.I == kQ4DownI && cfg.nthreads == kQ4DownThreads && cfg.K_vram == 256 && !cfg.stream_k);
    static_assert(!cfg.use_mma_data_layout());
    constexpr int qk = ggml_cuda_type_traits<T>::qk;
    static_assert(qk == 32 && QK8_1_MMQ == 128);
    constexpr auto load = ggml_cuda_mmq_get_load_tiles<T, J, false>();
    constexpr auto dot = ggml_cuda_mmq_get_vec_dot<T, J, false>();
    constexpr auto write = ggml_cuda_mmq_get_write_back<T, J, false>();
    const int expert = blockIdx.z, col = bounds[expert] + blockIdx.y * J, rem = bounds[expert + 1] - col;
    if (rem <= 0) return;
    const int row = blockIdx.x * kQ4DownI;
    const int lane = threadIdx.y * 64 + threadIdx.x;
    extern __shared__ int q4_down_shared[];
    int* ty = q4_down_shared + J;
    int* tx = ty + GGML_PAD(J * MMQ_TILE_Y_K, kQ4DownThreads);
    if (lane < J) q4_down_shared[lane] = lane < rem ? ids[col + lane] : 0;
    __syncthreads();
    y += col * kQ4Words;
    float sum[J * kQ4DownI / kQ4DownThreads] = {};
    const auto half_tile = [&](int k, int half) {
        const int* src = y + ncols_y * (k / QK8_1_MMQ + half) * kQ4Words;
#pragma unroll
        for (int i = 0; i < J * MMQ_TILE_Y_K; i += kQ4DownThreads) ty[i + lane] = src[i + lane];
        __syncthreads();
        dot(tx, ty, sum, half * MMQ_TILE_NE_K);
        __syncthreads();
    };
    constexpr int stride_row = kQ4DownK / qk;
    for (int kb = 0; kb < stride_row; kb += 256 / qk) {
        load(x, tx, expert * stride_expert + row * stride_row + kb, kQ4DownI - 1, stride_row);
        half_tile(kb * qk, 0);
        if (kb * qk + QK8_1_MMQ < kQ4DownK) half_tile(kb * qk, 1);
    }
    write(sum, q4_down_shared, dst + row, nullptr, dst_stride, kQ4DownI - 1, rem - 1);
#endif
}

template <ggml_type T> bool try_q4_down_tail(const mmq_args& a, cudaStream_t s) {
    constexpr int row_blocks = kQ4DownK / 32;
    if (a.nrows_x != kQ4DownM || a.ncols_x != kQ4DownK || a.stride_row_x != row_blocks || a.y_scale ||
        !a.expert_bounds || !a.ids_dst || a.nchannels_x != kQ4Experts || a.nchannels_y != kQ4Experts ||
        a.nsamples_x != 1 || a.nsamples_y != 1 || a.nrows_dst != kQ4DownM || a.ncols_max <= 0 ||
        a.ncols_max > a.ncols_y || a.ncols_y > INT_MAX / (8 * kQ4Words) || a.ncols_opt <= 0 ||
        a.ncols_opt > INT_MAX - 128 || a.stride_channel_x < kQ4DownM * row_blocks ||
        a.stride_channel_x > (INT_MAX - kQ4DownM * row_blocks - 6) / (kQ4Experts - 1)) return false;
    static const bool asked = [] {
        const char* v = std::getenv("STRATA_TP2_PREFILL_Q4_TAIL");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (!asked) return false;
    const auto& dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (dev.cc != GGML_CUDA_CC_VEGA20 || dev.warp_size != 64) return false;
    // Keep ggml's ascending J tie break. Fall back on any other geometry or selector result.
    int best = 0, tiles = INT_MAX;
    for (int j = 8; j <= 128 && tiles > 1; j += 8) {
        const auto cfg = ggml_cuda_mmq_get_config(T, j, false, dev.cc);
        if (cfg.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(cfg, dev.cc) > dev.smpbo) continue;
        const int n = (a.ncols_opt + cfg.J - 1) / cfg.J;
        if (n < tiles) { best = j; tiles = n; }
    }
    const auto run = [&]<int J>() {
        const auto cfg = ggml_cuda_mmq_get_config(T, J, false, dev.cc);
        if (cfg.I != kQ4DownI || cfg.J != J || cfg.nthreads != kQ4DownThreads || cfg.K_vram != 256 ||
            cfg.stream_k || cfg.use_mma_data_layout(dev.cc)) return false;
        const int lds = mmq_get_nbytes_shared(cfg, dev.cc);
        CUDA_SET_SHARED_MEMORY_LIMIT((q4_down_tail_gfx906<T, J>), lds);
        const dim3 grid(kQ4DownM / kQ4DownI, (a.ncols_max + J - 1) / J, a.nchannels_y);
        q4_down_tail_gfx906<T, J><<<grid, dim3(64, 4), lds, s>>>(
            a.x, a.y, a.ids_dst, a.expert_bounds, a.dst, a.stride_channel_x, a.ncols_y, a.nrows_dst);
        return true;
    };
    if (best == 32) return run.template operator()<32>();
    if (best == 64) return run.template operator()<64>();
    return false;
}

}  // namespace
}  // namespace strata::poly
