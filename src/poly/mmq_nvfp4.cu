// src/poly/mmq_nvfp4.cu - llama.cpp's NVFP4 tiles with 8-bit activations (W4A8), on CUDA and gfx1031.
//
// On sm_120 mmq.cuh multiplies NVFP4 as FP4 x FP4 (W4A4): the activations are rounded to E2M1 too.  Every other
// card gets its int8 path for NVFP4: weights decoded to int8, activations Q8_1, the precision decode runs at.  This
// unit compiles that path for sm_120 as well by hiding Blackwell's FP4 MMA from mmq.cuh on both sides, the device
// code and the host's tile config, which must agree on the tiles' shared memory (Strata's
// src/prefill/mmq_nvfp4_w4a8.cu).  The two non-static templates are renamed so the linker cannot fold them with
// the W4A4 instance built for 12xa (CMakeLists.txt).
// For HIP gfx1031, ggml's RDNA2 config already uses wave32 DP4A and the same Q8_1 layout.
#include "common.cuh"

#if !defined(GGML_USE_HIP)
#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#endif
#ifndef STRATA_W4A8_NAME   // mmq_nvfp4_lean.cu builds this unit again under other names
#define STRATA_W4A8_NAME(x) strata_poly_w4a8_##x
#endif
#define mul_mat_q_switch_J STRATA_W4A8_NAME(mul_mat_q_switch_J)
#define mul_mat_q_case STRATA_W4A8_NAME(mul_mat_q_case)
#include "mmq.cuh"

#include <cstdlib>

DECL_MMQ_CASE(GGML_TYPE_NVFP4);

namespace strata::poly {

#if defined(STRATA_HIP_GFX906)
namespace {
constexpr ggml_type kDownType = GGML_TYPE_NVFP4;
constexpr int kDownM = 2304, kDownK = 576, kDownI = 64, kDownThreads = 256;
constexpr int kDownWords = sizeof(block_q8_1_mmq) / sizeof(int);

// The regular grouped tile, except the last iteration only takes its first Q8_1 half. At K=576 the
// half starting at K=640 has zero activation scales. Keep the same weight loader and dot order for K=0..639.
template <int J>
__global__ __launch_bounds__(kDownThreads, 2)
void nvfp4_down_tail(const char* x, const int* y, const int32_t* write_ids, const int32_t* bounds, float* dst,
                    int stride_expert, int ncols_y, int dst_stride) {
#if defined(__gfx906__)
    constexpr auto cfg = ggml_cuda_mmq_get_config_gcn(kDownType, J, false);
    static_assert(cfg.I == kDownI && cfg.nthreads == kDownThreads && cfg.K_vram == 256 && !cfg.stream_k);
    static_assert(!cfg.use_mma_data_layout());
    static_assert(QK_NVFP4 == 64 && QK8_1_MMQ == 128);
    constexpr auto load = ggml_cuda_mmq_get_load_tiles<kDownType, J, false>();
    constexpr auto dot = ggml_cuda_mmq_get_vec_dot<kDownType, J, false>();
    constexpr auto write = ggml_cuda_mmq_get_write_back<kDownType, J, false>();
    const int e = blockIdx.z, col = bounds[e] + blockIdx.y * J, rem = bounds[e + 1] - col;
    if (rem <= 0) return;
    const int row = blockIdx.x * kDownI;
    const int lane = threadIdx.y * 64 + threadIdx.x;
    extern __shared__ int down_shared[];
    int* ty = down_shared + J;
    int* tx = ty + GGML_PAD(J * MMQ_TILE_Y_K, kDownThreads);
    if (lane < J) down_shared[lane] = lane < rem ? write_ids[col + lane] : 0;
    __syncthreads();
    y += col * kDownWords;
    float accum[J * kDownI / kDownThreads] = {};
    const auto half_tile = [&](int k, int half) {
        const int* src = y + ncols_y * (k / QK8_1_MMQ + half) * kDownWords;
#pragma unroll
        for (int i = 0; i < J * MMQ_TILE_Y_K; i += kDownThreads) ty[i + lane] = src[i + lane];
        __syncthreads();
        dot(tx, ty, accum, half * MMQ_TILE_NE_K);
        __syncthreads();
    };
    constexpr int stride_row = kDownK / QK_NVFP4;
    for (int kb = 0; kb < stride_row; kb += 4) {
        load(x, tx, e * stride_expert + row * stride_row + kb, kDownI - 1, stride_row);
        half_tile(kb * QK_NVFP4, 0);
        if (kb != 8) half_tile(kb * QK_NVFP4, 1);
    }
    write(accum, down_shared, dst + row, nullptr, dst_stride, kDownI - 1, rem - 1);
#endif
}

bool nvfp4_down_tail_try(const mmq_args& a, cudaStream_t s) {
    if (a.nrows_x != kDownM || a.ncols_x != kDownK || a.stride_row_x != kDownK / QK_NVFP4 || a.y_scale ||
        !a.expert_bounds || !a.ids_dst || a.nchannels_x != a.nchannels_y || a.nchannels_x <= 0 ||
        a.nchannels_x > 65535 || a.nsamples_x != 1 || a.nsamples_y != 1 || a.ncols_max <= 0 ||
        a.ncols_max > a.ncols_y || a.ncols_y > INT_MAX / (8 * kDownWords) || a.ncols_opt <= 0 ||
        a.ncols_opt > INT_MAX - 128 || a.nrows_dst < kDownM || a.nrows_dst > INT_MAX ||
        a.stride_channel_x < 0 || a.stride_channel_x > INT_MAX ||
        (a.nchannels_x - 1) * a.stride_channel_x + kDownM * (kDownK / QK_NVFP4) + 3 > INT_MAX) return false;
    static const bool asked = [] {
        const char* v = std::getenv("STRATA_TP2_PREFILL_NVFP4_TAIL");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (!asked) return false;
    const auto& dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (dev.warp_size != 64) return false;
    // Match ggml's ascending-J selection, including its tie rule; only the measured J32/J64 have a tail kernel.
    int best = 0, tiles = INT_MAX;
    for (int j = 8; j <= 128 && tiles > 1; j += 8) {
        const auto cfg = ggml_cuda_mmq_get_config(kDownType, j, false, dev.cc);
        if (cfg.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(cfg, dev.cc) > dev.smpbo) continue;
        const int count = (a.ncols_opt + cfg.J - 1) / cfg.J;
        if (count < tiles) { best = j; tiles = count; }
    }
    const auto launch = [&]<int J>() {
        const auto cfg = ggml_cuda_mmq_get_config(kDownType, J, false, dev.cc);
        if (cfg.I != kDownI || cfg.nthreads != kDownThreads || cfg.K_vram != 256 || cfg.stream_k ||
            cfg.sram_layout != GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4) return false;
        const int lds = mmq_get_nbytes_shared(cfg, dev.cc);
        CUDA_SET_SHARED_MEMORY_LIMIT(nvfp4_down_tail<J>, lds);
        const dim3 grid(kDownM / kDownI, (a.ncols_max + J - 1) / J, a.nchannels_y);
        nvfp4_down_tail<J><<<grid, dim3(64, kDownThreads / 64), lds, s>>>(
            a.x, a.y, a.ids_dst, a.expert_bounds, a.dst, a.stride_channel_x, a.ncols_y, a.nrows_dst);
        return true;
    };
    if (best == 32) return launch.template operator()<32>();
    if (best == 64) return launch.template operator()<64>();
    return false;
}
}  // namespace
#endif

void nvfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
#if defined(STRATA_HIP_GFX906)
    if (nvfp4_down_tail_try(a, s)) return;
#endif
    mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s);
}

// Mmq::fits() and Mmq::split_exact() for these tiles: the host's config as this unit sees the card
bool nvfp4_w4a8_fits(int64_t w_rows, int cc, size_t smpbo) {
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, w_rows % 128 != 0, cc);
        if (c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo) return true;
    }
    return false;
}

bool nvfp4_w4a8_stream_k(int64_t w_rows, int cc) {
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, w_rows % 128 != 0, cc);
        if (c.type != GGML_TYPE_COUNT && c.stream_k) return true;
    }
    return false;
}

}  // namespace strata::poly
