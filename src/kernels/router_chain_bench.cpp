// src/kernels/router_chain_bench.cpp - what the FFN-side router chain of one layer costs in a captured graph:
// the router GEMV (bf16 [512][2560], T rows), the top-10, the doorbell publish; 48 layers' worth with distinct
// weights (cold), as one graph, per-layer microseconds.  Bench only (no parity: the kernels are the engine's own).
//
//     build/router_chain_bench [T=2] [reps=20]
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)


int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int T = argc > 1 ? std::atoi(argv[1]) : 2;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int L = 48, N = 2560, NE = 512, KK = 10;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> w((size_t) NE * N);
    for (auto& x : w) {
        const float f = 0.05f * nd(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        x = (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    std::vector<float> xh((size_t) T * N);
    for (auto& x : xh) x = nd(rng);
    std::vector<uint16_t*> dw(L);
    for (int l = 0; l < L; ++l) {
        CK(cudaMalloc((void**) &dw[l], w.size() * 2));
        CK(cudaMemcpy(dw[l], w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    }
    float *dx, *dlog, *dwt, *dxo;
    int32_t *dids, *dids_o, *dres;
    CK(cudaMalloc((void**) &dx, (size_t) 8 * N * 4));   // the shape loop below runs up to 6 rows
    CK(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &dlog, (size_t) 8 * NE * 4));
    CK(cudaMalloc((void**) &dwt, (size_t) T * KK * 4));
    CK(cudaMalloc((void**) &dids, (size_t) T * KK * 4));
    CK(cudaMalloc((void**) &dids_o, (size_t) T * KK * 4));
    CK(cudaMalloc((void**) &dxo, xh.size() * 4));
    CK(cudaMalloc((void**) &dres, NE * 4));
    std::vector<int32_t> res(NE);
    for (int i = 0; i < NE; ++i) res[i] = i;
    CK(cudaMemcpy(dres, res.data(), NE * 4, cudaMemcpyHostToDevice));
    uint32_t* hseq;
    CK(cudaHostAlloc((void**) &hseq, 64, cudaHostAllocMapped | cudaHostAllocPortable));
    *hseq = 0;
    uint32_t* dseq;
    CK(cudaHostGetDevicePointer((void**) &dseq, hseq, 0));
    int* dz;
    CK(cudaMalloc((void**) &dz, 16));
    CK(cudaMemset(dz, 0, 16));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    {   // the 512 x 10 top-10 as the engine calls it (single / multi-token) against the reference kernel, bit for bit, on
        // adversarial logits: wide, integer ties, all equal, -inf, NaN, underflow, a tied top group across lanes
        const int NR = 4096;
        std::vector<float> lh((size_t) NR * NE);
        for (int t = 0; t < NR; ++t)
            for (int e = 0; e < NE; ++e) {
                const int kind = t % 8;
                float v = nd(rng) * (kind == 1 ? 8.f : 1.f);
                if (kind == 2) v = (float) (int) (v * 2);
                if (kind == 3) v = 0.0f;
                if (kind == 4 && e % 7 == 0) v = -INFINITY;
                if (kind == 5 && e % 50 == 3) v = NAN;
                if (kind == 6) v = nd(rng) * 200.f;
                if (kind == 7 && e < 20) v = 30.0f;
                lh[(size_t) t * NE + e] = v;
            }
        float *dl, *dw0, *dw1;
        int32_t *di0, *di1;
        CK(cudaMalloc((void**) &dl, lh.size() * 4));
        CK(cudaMemcpy(dl, lh.data(), lh.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**) &dw0, (size_t) NR * KK * 4)); CK(cudaMalloc((void**) &dw1, (size_t) NR * KK * 4));
        CK(cudaMalloc((void**) &di0, (size_t) NR * KK * 4)); CK(cudaMalloc((void**) &di1, (size_t) NR * KK * 4));
        long long bad = 0;
        for (int tk : {1, 2, 3, 8}) {
            CK(cudaMemset(dw0, 0xff, (size_t) NR * KK * 4)); CK(cudaMemset(dw1, 0xfe, (size_t) NR * KK * 4));
            CK(cudaMemset(di0, 0xff, (size_t) NR * KK * 4)); CK(cudaMemset(di1, 0xfe, (size_t) NR * KK * 4));
            K::native_router_top10_reference(dl, di0, dw0, NR, s);
            for (int t0 = 0; t0 < NR; t0 += tk) {
                const size_t o = (size_t) t0;
                if (tk == 1) K::native_router_top10(dl + o * NE, di1 + o * KK, dw1 + o * KK, s);
                else K::native_router_top10_multi(dl + o * NE, di1 + o * KK, dw1 + o * KK, std::min(tk, NR - t0), s);
            }
            CK(cudaStreamSynchronize(s));
            std::vector<uint32_t> a((size_t) NR * KK * 2), b((size_t) NR * KK * 2);
            CK(cudaMemcpy(a.data(), di0, (size_t) NR * KK * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(a.data() + NR * KK, dw0, (size_t) NR * KK * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), di1, (size_t) NR * KK * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data() + NR * KK, dw1, (size_t) NR * KK * 4, cudaMemcpyDeviceToHost));
            long long d = 0;
            for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
            std::printf("TOP10 PARITY n_tok %d: %lld of %zu ids/weights differ from the reference kernel\n", tk, d, a.size());
            bad += d;
        }
        cudaFree(dl); cudaFree(dw0); cudaFree(dw1); cudaFree(di0); cudaFree(di1);
        if (bad) return 1;
    }
    {   // outputs of the router GEMV for several shapes and token counts as FNV hashes: compare across STRATA_MMVF_PF=0/1
        const int shapes[][2] = {{2560, 512}, {2560, 256}, {2560, 64}, {1280, 513}, {640, 100}, {4096, 50}, {2048, 7}, {256, 33}, {8192, 90}};
        for (const auto& sh : shapes) {
            const int ni = sh[0], no = sh[1];
            std::vector<uint16_t> wh((size_t) no * ni);
            for (auto& v : wh) {
                const float f = 0.05f * nd(rng);
                uint32_t u;
                std::memcpy(&u, &f, 4);
                v = (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
            }
            std::vector<float> xs((size_t) 8 * ni);
            for (auto& v : xs) v = nd(rng);
            uint16_t* dwp;
            float *dxp, *dyp;
            CK(cudaMalloc((void**) &dwp, wh.size() * 2));
            CK(cudaMemcpy(dwp, wh.data(), wh.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc((void**) &dxp, xs.size() * 4));
            CK(cudaMemcpy(dxp, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc((void**) &dyp, (size_t) 8 * no * 4));
            for (int tk = 1; tk <= 8; ++tk) {
                CK(cudaMemset(dyp, 0xff, (size_t) 8 * no * 4));
                K::bf16_gemv_fp32_mmvf_multi(dxp, ni, dwp, dyp, no, ni, no, tk, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> yh((size_t) 8 * no);
                CK(cudaMemcpy(yh.data(), dyp, yh.size() * 4, cudaMemcpyDeviceToHost));
                unsigned long long h = 1469598103934665603ull;
                for (size_t b = 0; b < yh.size() * 4; ++b) h = (h ^ ((const unsigned char*) yh.data())[b]) * 1099511628211ull;
                std::printf("PARITY n_in %d n_out %d T %d hash %016llx\n", ni, no, tk, h);
            }
            cudaFree(dwp); cudaFree(dxp); cudaFree(dyp);
        }
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    {   // the BF16 GEMV by shape and token count, 48 distinct weight copies in one graph: us per call (STRATA_MMVF_PF=0 / 1 to compare)
        const int outs[] = {512, 256, 128, 64, 48, 8, 1};
        const int ins[] = {2560, 2560, 2560, 2560, 2560, 2560, 2560};
        for (int si = 0; si < 7; ++si) {
            const int no = outs[si], ni = ins[si];
            std::vector<uint16_t*> dws(L);
            for (int l = 0; l < L; ++l) {
                CK(cudaMalloc((void**) &dws[l], (size_t) no * ni * 2));
                CK(cudaMemcpy(dws[l], w.data(), (size_t) no * ni * 2 < w.size() * 2 ? (size_t) no * ni * 2 : w.size() * 2, cudaMemcpyHostToDevice));
            }
            for (int tk : {1, 2, 3, 4, 6}) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                for (int l = 0; l < L; ++l) K::bf16_gemv_fp32_mmvf_multi(dx, N, dws[l], dlog, NE, ni, no, tk, s);
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaStreamSynchronize(s));
                std::vector<float> ms;
                for (int r = 0; r < reps; ++r) {
                    CK(cudaEventRecord(e0, s));
                    CK(cudaGraphLaunch(ge, s));
                    CK(cudaEventRecord(e1, s));
                    CK(cudaEventSynchronize(e1));
                    float m = 0;
                    CK(cudaEventElapsedTime(&m, e0, e1));
                    ms.push_back(m);
                }
                std::sort(ms.begin(), ms.end());
                std::printf("SHAPE n_in %d n_out %3d T %d : %6.2f us per call\n", ni, no, tk, 1e3 * ms[ms.size() / 2] / L);
                cudaGraphExecDestroy(ge);
                cudaGraphDestroy(g);
            }
            for (int l = 0; l < L; ++l) cudaFree(dws[l]);
        }
    }
    {   // gfx906 router_chain_fused vs the separate launches: logits, ids, weights, published x / ids / weights, ring, bitwise
        unsigned* dcnt;
        float* dwo;
        CK(cudaMalloc((void**) &dcnt, 64)); CK(cudaMemset(dcnt, 0, 64));
        CK(cudaMalloc((void**) &dwo, (size_t) T * KK * 4));
        // half the layers' residency tables say every expert is resident (publish_res skips x), half say none is
        std::vector<int32_t> nores(NE, -1);
        int32_t* dnores;
        CK(cudaMalloc((void**) &dnores, NE * 4)); CK(cudaMemcpy(dnores, nores.data(), NE * 4, cudaMemcpyHostToDevice));
        long long bad = 0;
        for (int pub = 1; pub <= 2; ++pub)
            for (int l = 0; l < 4; ++l) {
                const int32_t* rt = (l & 1) ? dnores : dres;
                std::vector<uint8_t> out[2];
                uint32_t seq_d[2];
                for (int f = 0; f < 2; ++f) {
                    CK(cudaMemset(dlog, 0xff, (size_t) T * NE * 4)); CK(cudaMemset(dids, 0xff, (size_t) T * KK * 4));
                    CK(cudaMemset(dwt, 0xff, (size_t) T * KK * 4)); CK(cudaMemset(dxo, 0xff, xh.size() * 4));
                    CK(cudaMemset(dids_o, 0xff, (size_t) T * KK * 4)); CK(cudaMemset(dwo, 0xff, (size_t) T * KK * 4));
                    const uint32_t s0 = *hseq;
                    bool ran = false;
                    if (f == 1) ran = K::router_chain_fused(dx, dw[l], dlog, dids, dwt, T, pub, rt, NE, dxo, dids_o, dwo, dseq, dcnt, s);
                    if (f == 0 || !ran) {
                        if (f == 1) { std::printf("CHAIN PARITY: router_chain_fused not available here (T %d)\n", T); bad = -1; break; }
                        K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dlog, NE, N, NE, T, s);
                        K::native_router_top10_multi(dlog, dids, dwt, T, s);
                        if (pub == 2) K::doorbell_publish(dx, dids, dwt, (int64_t) T * N, (int64_t) T * KK, dxo, dids_o, dwo, dseq, s);
                        else K::doorbell_publish_res(dx, dids, rt, NE, (int64_t) T * N, (int64_t) T * KK, dxo, dids_o, dseq, s);
                    }
                    CK(cudaStreamSynchronize(s));
                    seq_d[f] = *hseq - s0;
                    out[f].resize(((size_t) T * NE + 4 * (size_t) T * KK + xh.size()) * 4);
                    uint8_t* q = out[f].data();
                    CK(cudaMemcpy(q, dlog, (size_t) T * NE * 4, cudaMemcpyDeviceToHost)); q += (size_t) T * NE * 4;
                    CK(cudaMemcpy(q, dids, (size_t) T * KK * 4, cudaMemcpyDeviceToHost)); q += (size_t) T * KK * 4;
                    CK(cudaMemcpy(q, dwt, (size_t) T * KK * 4, cudaMemcpyDeviceToHost)); q += (size_t) T * KK * 4;
                    CK(cudaMemcpy(q, dids_o, (size_t) T * KK * 4, cudaMemcpyDeviceToHost)); q += (size_t) T * KK * 4;
                    CK(cudaMemcpy(q, dwo, (size_t) T * KK * 4, cudaMemcpyDeviceToHost)); q += (size_t) T * KK * 4;
                    CK(cudaMemcpy(q, dxo, xh.size() * 4, cudaMemcpyDeviceToHost));
                }
                if (bad < 0) break;
                const bool same = out[0] == out[1] && seq_d[0] == 1 && seq_d[1] == 1;
                std::printf("CHAIN PARITY T %d publish %d layer %d (%s): %s (rings %u / %u)\n", T, pub, l, (l & 1) ? "none resident" : "all resident",
                            same ? "bitwise equal" : "DIFFERS", seq_d[0], seq_d[1]);
                bad += !same;
            }
        if (bad > 0) return 1;
        if (bad == 0) {
            for (int pub = 1; pub <= 2; ++pub) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                for (int l = 0; l < L; ++l) K::router_chain_fused(dx, dw[l], dlog, dids, dwt, T, pub, dres, NE, dxo, dids_o, dwo, dseq, dcnt, s);
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaStreamSynchronize(s));
                std::vector<float> ms;
                for (int r = 0; r < reps; ++r) {
                    CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
                    float m = 0; CK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m);
                }
                std::sort(ms.begin(), ms.end());
                std::printf("T %d %-20s : %6.2f us per layer (median of %d graphs of %d layers)\n", T, pub == 1 ? "fused chain (res)" : "fused chain (all)",
                            1e3 * ms[ms.size() / 2] / L, reps, L);
                cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
                // the separate chain with the always-publish doorbell, for the (all) comparison
                if (pub == 2) {
                    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                    for (int l = 0; l < L; ++l) {
                        K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dlog, NE, N, NE, T, s);
                        K::native_router_top10_multi(dlog, dids, dwt, T, s);
                        K::doorbell_publish(dx, dids, dwt, (int64_t) T * N, (int64_t) T * KK, dxo, dids_o, dwo, dseq, s);
                    }
                    CK(cudaStreamEndCapture(s, &g));
                    CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                    for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
                    CK(cudaStreamSynchronize(s));
                    ms.clear();
                    for (int r = 0; r < reps; ++r) {
                        CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
                        float m = 0; CK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m);
                    }
                    std::sort(ms.begin(), ms.end());
                    std::printf("T %d %-20s : %6.2f us per layer (median of %d graphs of %d layers)\n", T, "separate chain (all)",
                                1e3 * ms[ms.size() / 2] / L, reps, L);
                    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
                }
            }
        }
    }
    // mode bits: 1 mmvf, 2 route, 4 doorbell, 8 an empty kernel per layer (the node floor)
    const char* names[] = {"empty node", "mmvf", "route", "mmvf+route", "doorbell", "mmvf+doorbell", "route+doorbell", "mmvf+route+doorbell"};
    for (int mode = 0; mode < 8; ++mode) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < L; ++l) {
            if (mode == 0) K::gpu_stamp((unsigned long long*) dz, 0, s);
            if (mode & 1) K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dlog, NE, N, NE, T, s);
            if (mode & 2) K::native_router_top10_multi(dlog, dids, dwt, T, s);
            if (mode & 4) K::doorbell_publish_res(dx, dids, dres, NE, (int64_t) T * N, (int64_t) T * KK, dxo, dids_o, dseq, s);
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        std::vector<float> ms;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(e0, s));
            CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float m = 0;
            CK(cudaEventElapsedTime(&m, e0, e1));
            ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("T %d %-20s : %6.2f us per layer (median of %d graphs of %d layers)\n", T, names[mode], 1e3 * ms[ms.size() / 2] / L, reps, L);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    return 0;
}
