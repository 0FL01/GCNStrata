// src/kernels/cuda/dflash_qwen.cu - Qwen3.8-Flash-Next DFlash2 target taps (see include/strata/kernels/dflash_qwen.hpp).
#include "strata/kernels/dflash_qwen.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::kernels::dflash_qwen {
namespace {

__global__ void hc_stream_mean_kernel(const float* __restrict__ R, int n, float* __restrict__ out, long long out_ld) {
    const int r = blockIdx.y;
    const float* x = R + (size_t) r * 4 * n;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += gridDim.x * blockDim.x) {
        const float s01 = __fadd_rn(x[c], x[n + c]);
        const float s012 = __fadd_rn(s01, x[2 * n + c]);
        out[(size_t) r * out_ld + c] = __fmul_rn(__fadd_rn(s012, x[3 * n + c]), 0.25f);
    }
}

std::mutex g_mu;
std::FILE* g_file = nullptr;

}  // namespace

void hc_stream_mean(const float* R, int rows, int hc, int n, float* out, int64_t out_ld, void* stream) {
    if (rows <= 0) return;
    if (hc != 4) { std::fprintf(stderr, "dflash_qwen: hc_stream_mean needs 4 streams (got %d)\n", hc); std::exit(1); }
    const dim3 grid((unsigned) ((n + 255) / 256), (unsigned) rows);
    hc_stream_mean_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(R, n, out, (long long) out_ld);
}

bool taps_dump_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_DFLASH2_TAPS_DUMP"); return v != nullptr && v[0] != 0; }();
    return on;
}

void taps_dump_rows(const float* dev, int rows, int64_t ld, int n, int tap, int64_t pos0, void* stream) {
    if (!taps_dump_enabled() || rows <= 0) return;
    std::vector<float> h((size_t) rows * (size_t) ld);
    cudaStream_t s = (cudaStream_t) stream;
    if (cudaStreamSynchronize(s) != cudaSuccess ||
        cudaMemcpy(h.data(), dev, h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "dflash_qwen: taps dump copy failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_file == nullptr) {
        g_file = std::fopen(std::getenv("STRATA_DFLASH2_TAPS_DUMP"), "ab");
        if (g_file == nullptr) { std::fprintf(stderr, "dflash_qwen: cannot open the taps dump\n"); return; }
    }
    for (int r = 0; r < rows; ++r) {
        const int64_t pos = pos0 + r;
        const int32_t hdr[2] = {tap, n};
        std::fwrite(&pos, sizeof pos, 1, g_file);
        std::fwrite(hdr, sizeof hdr, 1, g_file);
        std::fwrite(h.data() + (size_t) r * (size_t) ld, sizeof(float), (size_t) n, g_file);
    }
    std::fflush(g_file);
}

void taps_dump_residual(const float* R, int rows, int hc, int n, int tap, int64_t pos0, void* stream) {
    if (!taps_dump_enabled() || rows <= 0) return;
    constexpr int kChunk = 256;
    float* tmp = nullptr;
    if (cudaMalloc(&tmp, (size_t) kChunk * (size_t) n * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "dflash_qwen: taps dump scratch failed\n");
        (void) cudaGetLastError();
        return;
    }
    for (int r0 = 0; r0 < rows; r0 += kChunk) {
        const int nr = rows - r0 < kChunk ? rows - r0 : kChunk;
        hc_stream_mean(R + (size_t) r0 * hc * n, nr, hc, n, tmp, n, stream);
        taps_dump_rows(tmp, nr, n, n, tap, pos0 + r0, stream);
    }
    cudaFree(tmp);
}

}  // namespace strata::kernels::dflash_qwen
