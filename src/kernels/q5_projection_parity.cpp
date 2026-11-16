// Q5_1 shared-expert projection: Strata CUDA dequantization vs ggml.
#define NOMINMAX
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "ggml.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

// Actual Q4_1 dense rows: decode uses the GPU's exact Q8_1 activations, while prefill uses the same FP16
// weights and activations as Gemm::native. Both references are independent ggml/scalar CPU calculations.
int q4_1_projection(const char* path, const char* name) {
    strata::GgufFile gguf(path);
    const auto* t = gguf.find(name);
    if (!t || t->type != GGML_TYPE_Q4_1 || t->shape.size() != 2 || t->shape[0] % 32 || t->shape[1] < 32) {
        std::fprintf(stderr, "expected Q4_1 matrix with at least 32 rows: %s\n", name);
        return 2;
    }
    const int K = (int) t->shape[0], N = 32, T = 3;
    const size_t rb = (size_t) K / 32 * 20, count = (size_t) N * K;
    if (!strata::kernels::native_mmvq_supported(3) ||
        strata::kernels::native_mmvq_weight_bytes(3, K, N) != rb * N ||
        !strata::kernels::dequant_bf16_supported(3)) return 1;
    const uint8_t* source = gguf.tensor_data(*t);
    std::vector<float> ref(count), x((size_t) T * K);
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q4_1);
    for (int r = 0; r < N; ++r) traits->to_float(source + r * rb, ref.data() + (size_t) r * K, K);
    std::vector<uint16_t> xh(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = (float) ((int) (i * 37 % 101) - 50) / 64.0f;
        xh[i] = ggml_fp32_to_fp16(x[i]);
    }
    void *dw = nullptr, *dxq = nullptr;
    float *dx = nullptr, *dy = nullptr, *df = nullptr;
    uint16_t *dxh = nullptr, *dh = nullptr, *db = nullptr;
    auto alloc = [](void** p, size_t bytes) { return cudaMalloc(p, bytes) == cudaSuccess; };
    if (!alloc(&dw, rb * N) || !alloc(&dxq, (size_t) T * K / 32 * 36) ||
        !alloc((void**) &dx, x.size() * 4) || !alloc((void**) &dy, (size_t) T * N * 4) ||
        !alloc((void**) &df, count * 4) || !alloc((void**) &dxh, xh.size() * 2) ||
        !alloc((void**) &dh, (size_t) (N - 1) * (K + 8) * 2) || !alloc((void**) &db, count * 2)) {
        std::fprintf(stderr, "Q4_1 allocation failed\n"); return 1;
    }
    if (cudaMemcpy(dw, source, rb * N, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dxh, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    strata::kernels::quantize_q8_1_rows(dx, T, K, dxq, nullptr);
    strata::kernels::native_mmvq(3, dw, dxq, dy, K, N, T, nullptr);
    std::vector<float> got((size_t) T * N);
    std::vector<uint8_t> q8((size_t) T * K / 32 * 36);
    if (cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(q8.data(), dxq, q8.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    double dot_err = 0, dot_den = 0;
    for (int c = 0; c < T; ++c) for (int r = 0; r < N; ++r) {
        double want = 0;
        for (int b = 0; b < K / 32; ++b) {
            const uint8_t* a = q8.data() + ((size_t) c * K / 32 + b) * 36;
            ggml_fp16_t d; std::memcpy(&d, a, 2);
            const double scale = ggml_fp16_to_fp32(d);
            for (int j = 0; j < 32; ++j)
                want += (double) ref[(size_t) r * K + b * 32 + j] * scale * (int8_t) a[4 + j];
        }
        if (!std::isfinite(got[(size_t) c * N + r]) || !std::isfinite(want)) return 1;
        dot_err += std::fabs(got[(size_t) c * N + r] - want); dot_den += std::fabs(want);
    }
    strata::kernels::dequant_f32(3, dw, 0, N, K, df, nullptr);
    strata::kernels::dequant_bf16(3, dw, 0, N, K, db, nullptr);
    if (!strata::kernels::dequant_f16_ld(3, dw, 1, N - 1, K, K + 8, dh, nullptr)) return 1;
    std::vector<float> gf(count);
    std::vector<uint16_t> gb(count), gh((size_t) (N - 1) * (K + 8));
    if (cudaMemcpy(gf.data(), df, count * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(gb.data(), db, count * 2, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(gh.data(), dh, gh.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    double dq_err = 0, dq_den = 0;
    size_t bf_bad = 0, f16_bad = 0;
    for (int r = 0; r < N; ++r) for (int k = 0; k < K; ++k) {
        const size_t i = (size_t) r * K + k;
        dq_err += std::fabs((double) gf[i] - ref[i]); dq_den += std::fabs((double) ref[i]);
        uint32_t bits; std::memcpy(&bits, &ref[i], 4);
        const uint16_t bf = (uint16_t) ((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        bf_bad += gb[i] != bf;
        if (r) f16_bad += gh[(size_t) (r - 1) * (K + 8) + k] != ggml_fp32_to_fp16(ref[i]);
    }
    std::string err;
    strata::prefill::Gemm gemm;
    if (!gemm.init(nullptr, (int64_t) 7 * K, err)) { std::fprintf(stderr, "Gemm init: %s\n", err.c_str()); return 1; }
    gemm.native(dxh, 3, dw, dy, T, N, K);
    if (cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    double gemm_err = 0, gemm_den = 0;
    for (int c = 0; c < T; ++c) for (int r = 0; r < N; ++r) {
        double want = 0;
        for (int k = 0; k < K; ++k)
            want += (double) ggml_fp16_to_fp32(ggml_fp32_to_fp16(ref[(size_t) r * K + k])) *
                    ggml_fp16_to_fp32(xh[(size_t) c * K + k]);
        if (!std::isfinite(got[(size_t) c * N + r]) || !std::isfinite(want)) return 1;
        gemm_err += std::fabs(got[(size_t) c * N + r] - want); gemm_den += std::fabs(want);
    }
    cudaFree(dw); cudaFree(dxq); cudaFree(dx); cudaFree(dy); cudaFree(df); cudaFree(dxh); cudaFree(dh); cudaFree(db);
    const double dot_rel = dot_err / (dot_den + 1e-30), dq_rel = dq_err / (dq_den + 1e-30);
    const double gemm_rel = gemm_err / (gemm_den + 1e-30);
    const bool ok = dot_rel < 1e-5 && dq_rel < 1e-6 && bf_bad == 0 && f16_bad == 0 && gemm_rel < 1e-3;
    std::printf("Q4_1 %s K=%d rows=%d: MMVQ %.3e, dequant %.3e bf16/f16 diffs %zu/%zu, sliced GEMM %.3e %s\n",
                name, K, N, dot_rel, dq_rel, bf_bad, f16_bad, gemm_rel, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--q4-1") == 0) return q4_1_projection(argv[2], argv[3]);
    if (argc != 2) {
        std::fprintf(stderr, "usage: q5_projection_parity <shard1.gguf>\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    const auto* t = gguf.find("blk.0.ffn_down_shexp.weight");
    if (!t || t->type != 7 || t->shape.size() != 2) {
        std::fprintf(stderr, "expected blk.0.ffn_down_shexp.weight Q5_1\n");
        return 2;
    }
    const int64_t cols = (int64_t) t->shape[0], rows = std::min<int64_t>(4, (int64_t) t->shape[1]);
    const size_t weight_bytes = (size_t) rows * (size_t) (cols / 32) * 24;
    const size_t elements = (size_t) rows * (size_t) cols;
    const uint8_t* source = gguf.tensor_data(*t);
    void *dweight = nullptr, *dfloat = nullptr, *dbf16 = nullptr;
    if (cudaMalloc(&dweight, weight_bytes) != cudaSuccess ||
        cudaMalloc(&dfloat, elements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dbf16, elements * sizeof(uint16_t)) != cudaSuccess) {
        std::fprintf(stderr, "CUDA allocation failed\n");
        return 1;
    }
    cudaMemcpy(dweight, source, weight_bytes, cudaMemcpyHostToDevice);
    strata::kernels::dequant_f32(7, dweight, 0, rows, cols, (float*) dfloat, nullptr);
    strata::kernels::dequant_bf16(7, dweight, 0, rows, cols, (uint16_t*) dbf16, nullptr);
    std::vector<float> got(elements), ref(elements);
    std::vector<uint16_t> got_bf16(elements);
    cudaMemcpy(got.data(), dfloat, elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_bf16.data(), dbf16, elements * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(dweight); cudaFree(dfloat); cudaFree(dbf16);
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q5_1);
    for (int64_t r = 0; r < rows; ++r)
        traits->to_float(source + (size_t) r * (size_t) (cols / 32) * 24,
                         ref.data() + (size_t) r * (size_t) cols, cols);
    double max_abs = 0;
    size_t bf16_bad = 0;
    for (size_t i = 0; i < elements; ++i) {
        max_abs = std::max(max_abs, (double) std::fabs(got[i] - ref[i]));
        uint32_t bits;
        std::memcpy(&bits, &ref[i], sizeof(bits));
        const uint16_t expected = (uint16_t) ((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
        if (got_bf16[i] != expected) ++bf16_bad;
    }
    const bool ok = max_abs <= 1e-6 && bf16_bad == 0;
    std::printf("Q5_1 projection: %zu elements, max_abs %.3e, bf16 differences %zu %s\n",
                elements, max_abs, bf16_bad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
