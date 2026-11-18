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
#include <cmath>
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
    {   // argmax_rows (the verifier's multi-block pick where argmax_rows_wanted) vs the one-block greedy, token for token
        const int R = 64;
        std::vector<float> a((size_t) R * NV);
        std::uniform_int_distribution<int> pos(0, NV - 1);
        for (int r = 0; r < R; ++r) {
            float* x = &a[(size_t) r * NV];
            const int kind = r % 8;
            for (int v = 0; v < NV; ++v) x[v] = kind == 3 ? 0.0f : nd(rng);
            if (kind == 1) { const float m = 50.f; for (int k = 0; k < 9; ++k) x[pos(rng)] = m; }      // tied maxima
            if (kind == 2) for (int v = 0; v < NV; v += 3) x[v] = NAN;                                  // NaN never wins
            if (kind == 4) for (int v = 0; v < NV; ++v) x[v] = -INFINITY;                               // none: 0
            if (kind == 5) { for (int v = 0; v < NV; ++v) x[v] = -INFINITY; x[NV - 1] = -0.0f; x[NV - 7] = 0.0f; }
            if (kind == 6) { x[0] = 99.f; x[NV - 1] = 99.f; }
            if (kind == 7) for (int v = 0; v < NV; ++v) x[v] = (float) (v % 1000) * 0.001f;             // many ties at the top
        }
        float* da;
        int32_t *o1, *o2;
        void* scr;
        CK(cudaMalloc((void**) &da, a.size() * 4));
        CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**) &o1, R * 4)); CK(cudaMalloc((void**) &o2, R * 4));
        const uint64_t sb = K::argmax_rows_scratch_bytes(std::max(T, 8));   // the parity's 8 rows and the timing's T
        CK(cudaMalloc(&scr, sb ? sb : 16)); CK(cudaMemset(scr, 0, sb ? sb : 16));
        int bad = 0;
        if (K::argmax_rows_wanted()) {
            for (int r0 = 0; r0 < R; r0 += 8) {
                K::SamplerParams sp; sp.greedy = true; sp.temperature = 0.0f;
                K::sample_tokens(da + (size_t) r0 * NV, 8, NV, nullptr, 0, sp, o1 + r0, s);
                K::argmax_rows(da + (size_t) r0 * NV, (r0 / 8) % 2 ? 2 : 8, NV, scr, o2 + r0, s);
                if ((r0 / 8) % 2) K::argmax_rows(da + (size_t) (r0 + 2) * NV, 6, NV, scr, o2 + r0 + 2, s);
            }
            CK(cudaStreamSynchronize(s));
            std::vector<int32_t> h1(R), h2(R);
            CK(cudaMemcpy(h1.data(), o1, R * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2.data(), o2, R * 4, cudaMemcpyDeviceToHost));
            for (int r = 0; r < R; ++r) if (h1[r] != h2[r]) { if (bad < 5) std::printf("row %d: one-block %d, argmax_rows %d\n", r, h1[r], h2[r]); ++bad; }
            std::printf("ARGMAX PARITY: %d of %d rows differ (adversarial rows: ties, NaN, -inf, +-0)\n", bad, R);
            {   // argmax_rows_nf: the same pick, and per row "any flag word set" as logits_nonfinite_rows
                uint32_t *f1, *f2;
                int32_t* o3;
                CK(cudaMalloc((void**) &f1, 8 * K::kNonfiniteBlocks * 4)); CK(cudaMalloc((void**) &f2, 8 * K::kNonfiniteBlocks * 4));
                CK(cudaMalloc((void**) &o3, R * 4));
                int nbad = 0;
                for (int r0 = 0; r0 < R; r0 += 8) {
                    CK(cudaMemset(f1, 0x5a, 8 * K::kNonfiniteBlocks * 4)); CK(cudaMemset(f2, 0x5a, 8 * K::kNonfiniteBlocks * 4));
                    K::logits_nonfinite_rows(da + (size_t) r0 * NV, 8, NV, f1, s);
                    K::argmax_rows_nf(da + (size_t) r0 * NV, 8, NV, scr, o3 + r0, f2, s);
                    CK(cudaStreamSynchronize(s));
                    std::vector<uint32_t> g1(8 * K::kNonfiniteBlocks), g2(8 * K::kNonfiniteBlocks);
                    CK(cudaMemcpy(g1.data(), f1, g1.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(g2.data(), f2, g2.size() * 4, cudaMemcpyDeviceToHost));
                    for (int t = 0; t < 8; ++t) {
                        bool a1 = false, a2 = false;
                        for (int b = 0; b < K::kNonfiniteBlocks; ++b) { a1 |= g1[t * K::kNonfiniteBlocks + b] != 0; a2 |= g2[t * K::kNonfiniteBlocks + b] != 0; }
                        if (a1 != a2) ++nbad;
                    }
                }
                std::vector<int32_t> h3(R);
                CK(cudaMemcpy(h3.data(), o3, R * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h1.data(), o1, R * 4, cudaMemcpyDeviceToHost));
                for (int r = 0; r < R; ++r) nbad += h3[r] != h1[r];
                std::printf("ARGMAX+NF PARITY: %d mismatches (row masks vs logits_nonfinite_rows, picks vs one-block)\n", nbad);
                bad += nbad;
            }
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            for (int l = 0; l < L; ++l) K::argmax_rows_nf(dl, T, NV, scr, dout_d, dnf, s);
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
            std::printf("T %d %-24s : %7.2f us per call (median of %d graphs of %d calls)\n", T, "argmax_rows_nf", 1e3 * ms[ms.size() / 2] / L, reps, L);
        } else std::printf("argmax_rows not wanted on this card\n");
        if (bad) return 1;
    }
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
    {   // gfx906 round-free sampled pick (STRATA_GFX906_SAMPLER_RF) vs the split kernels: the drawn tokens of many (row,
        // seed, parameters) on adversarial rows, with and without penalties, then the time per call
        const int R = 32, SEEDS = 40;
        std::vector<float> a((size_t) R * NV);
        std::uniform_int_distribution<int> pos(0, NV - 1);
        for (int r = 0; r < R; ++r) {
            float* x = &a[(size_t) r * NV];
            const int kind = r % 8;
            for (int v = 0; v < NV; ++v) x[v] = nd(rng);
            if (kind == 1) for (int q = 0; q < 40; ++q) x[pos(rng)] = 12.0f;                         // many tied tops
            if (kind == 2) for (int v = 0; v < NV; v += 5) x[v] = NAN;
            if (kind == 3) for (int v = 0; v < NV; ++v) x[v] = (v % 4096) < 4090 ? -INFINITY : 0.5f;  // few a block
            if (kind == 4) { for (int v = 0; v < NV; ++v) x[v] = -INFINITY; for (int q = 0; q < 7; ++q) x[pos(rng)] = 1.0f; }
            if (kind == 5) for (int v = 0; v < NV; ++v) x[v] = (v & 1) ? 0.0f : -0.0f;              // +-0 ties
            if (kind == 6) for (int v = 0; v < NV; ++v) x[v] = (float) ((v * 7919) % 97) * 0.1f;     // dense ties
            if (kind == 7) { x[pos(rng)] = INFINITY; x[pos(rng)] = INFINITY; }                      // +inf wins
        }
        float* da;
        int *ta, *tb, *hist;
        CK(cudaMalloc((void**) &da, a.size() * 4));
        CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**) &ta, R * SEEDS * 4)); CK(cudaMalloc((void**) &tb, R * SEEDS * 4));
        std::vector<int> hh((size_t) 8 * 64);
        for (auto& v : hh) v = pos(rng);
        CK(cudaMalloc((void**) &hist, hh.size() * 4)); CK(cudaMemcpy(hist, hh.data(), hh.size() * 4, cudaMemcpyHostToDevice));
        long long diff = 0;
        for (int pen = 0; pen < 2; ++pen) {
            for (int rf = 0; rf < 2; ++rf) {
                if (setenv("STRATA_GFX906_SAMPLER_RF", rf ? "1" : "0", 1) != 0) return 2;
                for (int sd = 0; sd < SEEDS; ++sd)
                    for (int r0 = 0; r0 < R; r0 += 8) {
                        K::SamplerParams sp; sp.seed = 1000 + sd; sp.counter = (uint64_t) (sd * 13 + r0);
                        const int ks[5] = {20, 1, 5, 64, 0};
                        sp.top_k = ks[sd % 5]; sp.top_p = sd % 2 ? 0.95f : 1.0f; sp.temperature = 0.5f + 0.1f * (sd % 7);
                        sp.min_p = sd % 3 == 0 ? 0.05f : 0.0f;
                        if (pen) { sp.penalty_last_n = 64; sp.penalty_repeat = 1.3f; sp.penalty_freq = 0.2f; sp.penalty_present = 0.1f; }
                        K::sample_tokens(da + (size_t) r0 * NV, 8, NV, pen ? hist : nullptr, pen ? 64 : 0, sp, (rf ? tb : ta) + sd * R + r0, s);
                    }
            }
            CK(cudaStreamSynchronize(s));
            std::vector<int> h1(R * SEEDS), h2(R * SEEDS);
            CK(cudaMemcpy(h1.data(), ta, h1.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2.data(), tb, h2.size() * 4, cudaMemcpyDeviceToHost));
            long long d = 0;
            for (size_t i = 0; i < h1.size(); ++i) d += h1[i] != h2[i];
            std::printf("SAMPLER RF PARITY (penalties %d): %lld of %zu drawn tokens differ\n", pen, d, h1.size());
            diff += d;
        }
        if (diff) return 1;
        for (int rf = 0; rf < 2; ++rf) {
            if (setenv("STRATA_GFX906_SAMPLER_RF", rf ? "1" : "0", 1) != 0) return 2;
            std::vector<float> ms;
            for (int r = 0; r < reps; ++r) {
                CK(cudaEventRecord(e0, s));
                for (int l = 0; l < L; ++l) { K::SamplerParams sp; sp.seed = 7; sp.counter = (uint64_t) l; K::sample_tokens(dl, T, NV, nullptr, 0, sp, dout_d, s); }
                CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
                float m = 0; CK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m);
            }
            std::sort(ms.begin(), ms.end());
            std::printf("T %d %-24s : %7.2f us per call (median of %d x %d direct calls)\n", T, rf ? "sampled, round-free" : "sampled, split", 1e3 * ms[ms.size() / 2] / L, reps, L);
        }
        unsetenv("STRATA_GFX906_SAMPLER_RF");
    }
    {   // the same direct-launch loop with one small kernel (the guard), for the launch cost alone
        std::vector<float> ms;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(e0, s));
            for (int l = 0; l < L; ++l) K::logits_nonfinite_rows(dl, T, NV, dnf, s);
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float m = 0; CK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("T %d %-24s : %7.2f us per call (median of %d x %d direct calls)\n", T, "guard (direct)", 1e3 * ms[ms.size() / 2] / L, reps, L);
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
