// src/kernels/layer_chain_mb.cpp - gfx906 micro-benchmark of short decode launch chains in a graph, cold weights:
// two separate native_mmvq launches against native_mmvq_two (one grid), at the B3 shapes (GDN qkv + z Q4_0, shared
// expert gate + up Q5_0 / Q8_0, QSA K + V Q8_0), 1 and 2 columns; outputs compared bit for bit.
//
//     build/layer_chain_mb [reps=200] [sets=24]
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

namespace {
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

struct Case { const char* name; int type, n_in, n1, n2; };
const Case CASES[] = {
    {"gdn qkv+z Q4_0", 2, 2560, 10240, 6144},
    {"shexp gate+up Q5_0", 6, 2560, 640, 640},
    {"shexp gate+up Q8_0", 8, 2560, 640, 640},
    {"qsa k+v Q8_0", 8, 2560, 512, 512},
};

// synthetic blocks with sane fp16 scales (scale at byte 0 for Q4_0 / Q5_0 / Q8_0)
std::vector<uint8_t> weights(int type, int n_in, int n_out, std::mt19937& rng) {
    const std::size_t bytes = K::native_mmvq_weight_bytes(type, n_in, n_out);
    std::vector<uint8_t> w(bytes);
    for (auto& b : w) b = (uint8_t) rng();
    const int bb = type == 2 ? 18 : type == 6 ? 22 : 34;
    for (std::size_t o = 0; o + bb <= bytes; o += bb) {
        const uint16_t h = (uint16_t) (((5u + rng() % 5u) << 10) | (rng() & 0x3ffu));
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200, sets = argc > 2 ? std::atoi(argv[2]) : 24;
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::mt19937 rng(7);
    int bad = 0;
    for (const Case& c : CASES) {
        std::vector<void*> w1(sets), w2(sets);
        for (int i = 0; i < sets; ++i) {
            auto a = weights(c.type, c.n_in, c.n1, rng), b = weights(c.type, c.n_in, c.n2, rng);
            CK(cudaMalloc(&w1[i], a.size()));
            CK(cudaMalloc(&w2[i], b.size()));
            CK(cudaMemcpy(w1[i], a.data(), a.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(w2[i], b.data(), b.size(), cudaMemcpyHostToDevice));
        }
        for (int nc = 1; nc <= 2; ++nc) {
            std::vector<float> xf((size_t) nc * c.n_in);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : xf) v = nd(rng);
            float *dx, *y1, *y2, *z1, *z2;
            void* xq;
            CK(cudaMalloc((void**) &dx, xf.size() * 4));
            CK(cudaMemcpy(dx, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&xq, K::native_q8_1_bytes(c.n_in, nc)));
            K::native_quantize_q8_1(dx, xq, c.n_in, nc, s);
            CK(cudaMalloc((void**) &y1, (size_t) nc * c.n1 * 4));
            CK(cudaMalloc((void**) &y2, (size_t) nc * c.n2 * 4));
            CK(cudaMalloc((void**) &z1, (size_t) nc * c.n1 * 4));
            CK(cudaMalloc((void**) &z2, (size_t) nc * c.n2 * 4));
            double us[2] = {0, 0};
            for (int mode = 0; mode < 2; ++mode) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                for (int i = 0; i < sets; ++i) {
                    if (mode == 0) {
                        K::native_mmvq(c.type, w1[i], xq, y1, c.n_in, c.n1, nc, s);
                        K::native_mmvq(c.type, w2[i], xq, y2, c.n_in, c.n2, nc, s);
                    } else if (!K::native_mmvq_two(c.type, w1[i], z1, c.n1, w2[i], z2, c.n2, xq, c.n_in, nc, s)) {
                        std::printf("%s: native_mmvq_two declined\n", c.name);
                        std::exit(1);
                    }
                }
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                for (int r = 0; r < 5; ++r) CK(cudaGraphLaunch(ge, s));
                cudaEvent_t e0, e1;
                CK(cudaEventCreate(&e0));
                CK(cudaEventCreate(&e1));
                CK(cudaEventRecord(e0, s));
                for (int r = 0; r < reps; ++r) CK(cudaGraphLaunch(ge, s));
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                us[mode] = ms * 1000.0 / reps / sets;
            }
            std::vector<float> a((size_t) nc * c.n1), b((size_t) nc * c.n1), a2((size_t) nc * c.n2), b2((size_t) nc * c.n2);
            CK(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), z1, b.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(a2.data(), y2, a2.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b2.data(), z2, b2.size() * 4, cudaMemcpyDeviceToHost));
            const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0 && std::memcmp(a2.data(), b2.data(), a2.size() * 4) == 0;
            bad += !same;
            std::printf("%-20s cols %d: two launches %7.2f us, one launch %7.2f us (%+.1f%%) %s\n", c.name, nc, us[0], us[1],
                        100.0 * (us[1] - us[0]) / us[0], same ? "bitwise" : "DIFFERENT");
        }
    }
    return bad ? 1 : 0;
}
