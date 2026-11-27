// src/kernels/cuda/dflash_qwen.cu - Qwen3.8-Flash-Next DFlash2 target taps (see include/strata/kernels/dflash_qwen.hpp).
#include "strata/kernels/dflash_qwen.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

namespace {
struct Sink {
    float* ring = nullptr;     // [kSinkRows][kTaps][n] pinned, portable
    int64_t* tag = nullptr;    // [kSinkRows][kTaps]: the position each slice holds, -1: none
    int n = 0;
    int64_t lo = 0;
};
Sink g_sink;
std::mutex g_sink_mu;   // allocation only; slices of different taps/positions are disjoint
}  // namespace

bool sink_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_DFLASH2"); return v != nullptr && v[0] != 0; }();
    return on;
}

void sink_begin(int64_t lo) {
    if (!sink_enabled()) return;
    g_sink.lo = lo < 0 ? 0 : lo;
    if (g_sink.tag != nullptr)
        for (size_t i = 0; i < (size_t) kSinkRows * kTaps; ++i) g_sink.tag[i] = -1;
}

namespace {
void sink_alloc(int n) {
    std::lock_guard<std::mutex> lk(g_sink_mu);
    if (g_sink.ring != nullptr) return;
    void* r = nullptr;
    if (cudaHostAlloc(&r, (size_t) kSinkRows * kTaps * (size_t) n * sizeof(float), cudaHostAllocPortable) != cudaSuccess) {
        std::fprintf(stderr, "dflash_qwen: the taps ring (%zu MiB pinned) cannot be allocated\n",
                     ((size_t) kSinkRows * kTaps * (size_t) n * sizeof(float)) >> 20);
        std::exit(1);
    }
    g_sink.ring = (float*) r;
    g_sink.tag = new int64_t[(size_t) kSinkRows * kTaps];
    for (size_t i = 0; i < (size_t) kSinkRows * kTaps; ++i) g_sink.tag[i] = -1;
    g_sink.n = n;
}
// rows [a, b) of `dev` (row 0 at pos0) into the ring, enqueued; returns the first position copied (b: none)
int64_t sink_copy(const float* dev, int rows, int64_t ld, int n, int tap, int64_t pos0, cudaStream_t s) {
    sink_alloc(n);
    if (n != g_sink.n) { std::fprintf(stderr, "dflash_qwen: taps ring width %d != %d\n", n, g_sink.n); std::exit(1); }
    int64_t a = pos0 > g_sink.lo ? pos0 : g_sink.lo;
    const int64_t b = pos0 + rows;
    if (b - a > kSinkRows) a = b - kSinkRows;
    const int64_t first = a < b ? a : b;
    while (a < b) {
        const int64_t slot = a % kSinkRows;
        const int64_t cnt = std::min<int64_t>(b - a, kSinkRows - slot);
        const cudaError_t e = cudaMemcpy2DAsync(g_sink.ring + ((size_t) slot * kTaps + tap) * n, (size_t) kTaps * n * sizeof(float),
                                                dev + (size_t) (a - pos0) * ld, (size_t) ld * sizeof(float), (size_t) n * sizeof(float),
                                                (size_t) cnt, cudaMemcpyDeviceToHost, s);
        if (e != cudaSuccess) { std::fprintf(stderr, "dflash_qwen: taps ring copy: %s\n", cudaGetErrorString(e)); std::exit(1); }
        a += cnt;
    }
    return first;
}
}  // namespace

void sink_stage_rows(const float* dev, int rows, int64_t ld, int n, int tap, int64_t pos0, void* stream) {
    if (!sink_enabled() || rows <= 0 || tap < 0 || tap >= kTaps) return;
    const int64_t b = pos0 + rows;
    for (int64_t q = std::max<int64_t>(pos0, b - kSinkRows); q < b; ++q)   // a slot being rewritten is not complete
        if (g_sink.tag != nullptr) g_sink.tag[(size_t) (q % kSinkRows) * kTaps + tap] = -1;
    (void) sink_copy(dev, rows, ld, n, tap, pos0, (cudaStream_t) stream);
}

void sink_tag_rows(int tap, int64_t pos0, int rows) {
    if (!sink_enabled() || rows <= 0 || g_sink.tag == nullptr) return;
    const int64_t b = pos0 + rows;
    for (int64_t q = std::max({pos0, g_sink.lo, b - kSinkRows}); q < b; ++q) g_sink.tag[(size_t) (q % kSinkRows) * kTaps + tap] = q;
}

void sink_rows(const float* dev, int rows, int64_t ld, int n, int tap, int64_t pos0, void* stream) {
    if (!sink_enabled() || rows <= 0 || tap < 0 || tap >= kTaps) return;
    sink_alloc(n);
    const int64_t b = pos0 + rows;
    for (int64_t q = std::max<int64_t>(pos0, b - kSinkRows); q < b; ++q) g_sink.tag[(size_t) (q % kSinkRows) * kTaps + tap] = -1;
    const int64_t first = sink_copy(dev, rows, ld, n, tap, pos0, (cudaStream_t) stream);
    if (first >= b) return;
    const cudaError_t e = cudaStreamSynchronize((cudaStream_t) stream);
    if (e != cudaSuccess) { std::fprintf(stderr, "dflash_qwen: taps ring sync: %s\n", cudaGetErrorString(e)); std::exit(1); }
    for (int64_t q = first; q < b; ++q) g_sink.tag[(size_t) (q % kSinkRows) * kTaps + tap] = q;
}

void sink_residual(const float* R, int rows, int hc, int n, int tap, int64_t pos0, void* stream) {
    if (!sink_enabled() || rows <= 0) return;
    // only the rows the ring keeps
    int64_t a = pos0 > g_sink.lo ? pos0 : g_sink.lo;
    const int64_t b = pos0 + rows;
    if (b - a > kSinkRows) a = b - kSinkRows;
    if (a >= b) return;
    constexpr int kChunk = 256;
    static thread_local float* tmp = nullptr;
    static thread_local int tmp_dev = -1;
    int dev = -1;
    (void) cudaGetDevice(&dev);
    if (tmp == nullptr || tmp_dev != dev) {
        if (cudaMalloc(&tmp, (size_t) kChunk * (size_t) n * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "dflash_qwen: taps ring scratch failed\n");
            std::exit(1);
        }
        tmp_dev = dev;   // (a thread's scratch on one device; never freed - a few MiB)
    }
    for (int64_t r0 = a; r0 < b; r0 += kChunk) {
        const int nr = (int) std::min<int64_t>(kChunk, b - r0);
        hc_stream_mean(R + (size_t) (r0 - pos0) * hc * n, nr, hc, n, tmp, n, stream);
        sink_rows(tmp, nr, n, n, tap, r0, stream);
    }
}

int64_t sink_complete_from(int64_t a, int64_t b) {
    if (g_sink.tag == nullptr) return b;
    int64_t s = b;
    while (s > a) {
        const int64_t q = s - 1;
        bool ok = true;
        for (int k = 0; k < kTaps && ok; ++k) ok = g_sink.tag[(size_t) (q % kSinkRows) * kTaps + k] == q;
        if (!ok) break;
        s = q;
    }
    return s;
}

const float* sink_row(int64_t pos) { return g_sink.ring + (size_t) (pos % kSinkRows) * kTaps * (size_t) g_sink.n; }
int64_t sink_wrap_rows(int64_t pos) { return kSinkRows - pos % kSinkRows; }

}  // namespace strata::kernels::dflash_qwen
