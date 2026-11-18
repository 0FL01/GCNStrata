// src/kernels/head_tail_bench.cpp - what the verify head's tail costs after the output projection: the greedy pick
// (sample_tokens greedy, as Verifier::record_window records it), the nonfinite guard, and the sampled pick (split
// top-k, run after the window), on T rows of a 248,320-token vocabulary, as a graph of 48 calls (us per call).
//
//     build/head_tail_bench [T=1] [reps=20]
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace K = strata::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int T = argc > 1 ? std::atoi(argv[1]) : 1, reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int NV = 248320, L = 48;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> h((size_t) T * NV);
    for (auto& v : h) v = nd(rng);
    float* dl;
    int* dout;
    uint32_t* dnf;
    CK(cudaMalloc((void**) &dl, h.size() * 4));
    CK(cudaMemcpy(dl, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaHostAlloc((void**) &dout, 64, cudaHostAllocMapped));   // the verifier's m_out_ is mapped host memory
    int* dout_d;
    CK(cudaHostGetDevicePointer((void**) &dout_d, dout, 0));
    CK(cudaMalloc((void**) &dnf, 4096));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const char* names[] = {"greedy (sample_tokens)", "nonfinite guard", "greedy + guard", "sampled (split top-k)"};
    {   // the sampled pick as the engine runs it: launched after the replay, not captured (a captured call takes the
        // one-block kernel)
        for (int i = 0; i < 20; ++i) { K::SamplerParams sp; sp.seed = 7; sp.counter = (uint64_t) i; K::sample_tokens(dl, T, NV, nullptr, 0, sp, dout_d, s); }
        CK(cudaStreamSynchronize(s));
        std::vector<float> ms;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(e0, s));
            for (int l = 0; l < L; ++l) { K::SamplerParams sp; sp.seed = 7; sp.counter = (uint64_t) l; K::sample_tokens(dl, T, NV, nullptr, 0, sp, dout_d, s); }
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float m = 0; CK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("T %d %-24s : %7.2f us per call (median of %d x %d direct calls)\n", T, "sampled split (direct)", 1e3 * ms[ms.size() / 2] / L, reps, L);
    }
    for (int mode = 0; mode < 3; ++mode) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < L; ++l) {
            K::SamplerParams sp;
            if (mode != 3) { sp.greedy = true; sp.temperature = 0.0f; }
            else { sp.seed = 7; sp.counter = (uint64_t) l; }
            if (mode == 0 || mode == 2 || mode == 3) K::sample_tokens(dl, T, NV, nullptr, 0, sp, dout_d, s);
            if (mode == 1 || mode == 2) K::logits_nonfinite_rows(dl, T, NV, dnf, s);
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        std::vector<float> ms;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float m = 0; CK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("T %d %-24s : %7.2f us per call (median of %d graphs of %d calls)\n", T, names[mode], 1e3 * ms[ms.size() / 2] / L, reps, L);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    return 0;
}
