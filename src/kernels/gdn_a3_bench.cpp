// Lab-only gfx906 projection microbench: real GGUF/native-pack weights, synthetic activations.
// Run in separate processes with STRATA_GDN_SEG / STRATA_GDN_A3 unset and with both set to 1.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = strata::kernels;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class X> struct Buf {
    X* p = nullptr;
    explicit Buf(size_t n) { ck(cudaMalloc((void**) &p, n * sizeof(X))); }
    ~Buf() { if (p) cudaFree(p); }
    Buf(const Buf&) = delete;
};
struct Weight { int type; uint64_t off, bytes; std::string path; };
using Weights = std::map<std::string, std::vector<Weight>>;
static void upload(const Weight& w, void* p) {
    std::vector<char> bytes(w.bytes);
    std::ifstream in(w.path, std::ios::binary);
    if (!in.seekg(w.off) || !in.read(bytes.data(), w.bytes)) throw std::runtime_error("weight read failed: " + w.path);
    ck(cudaMemcpy(p, bytes.data(), w.bytes, cudaMemcpyHostToDevice));
}
static std::vector<uint32_t> read(const float* y, size_t n, cudaStream_t s) {
    std::vector<uint32_t> h(n);
    ck(cudaStreamSynchronize(s));
    ck(cudaMemcpy(h.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}
static uint64_t hash(const std::vector<uint32_t>& y) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t v : y) h = (h ^ v) * 1099511628211ull;
    return h;
}
static unsigned diff(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    unsigned d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    return d;
}
static bool enabled(const char* name) {
    const char* e = std::getenv(name);
    return e && e[0] == '1' && !e[1];
}
template<class Run> static void capture_once(Run run, cudaStream_t s, unsigned ab_grid) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    run();
    ck(cudaStreamEndCapture(s, &graph));
    size_t nodes = 0;
    ck(cudaGraphGetNodes(graph, nullptr, &nodes));
    if (nodes != 1) throw std::runtime_error("expected one projection node");
    if (ab_grid) {
        cudaGraphNode_t node;
        cudaKernelNodeParams params{};
        ck(cudaGraphGetNodes(graph, &node, &nodes));
        ck(cudaGraphKernelNodeGetParams(node, &params));
        if (params.gridDim.x != ab_grid) throw std::runtime_error("unexpected GDN a/b dispatch");
    }
    ck(cudaGraphInstantiate(&exec, graph, 0));
    ck(cudaGraphLaunch(exec, s));
    ck(cudaStreamSynchronize(s));
    ck(cudaGraphExecDestroy(exec));
    ck(cudaGraphDestroy(graph));
}
template<class Run> static double timed(Run run, cudaStream_t s, int count) {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    for (int i = 0; i < 48; ++i) run(i);
    ck(cudaEventRecord(a, s));
    for (int i = 0; i < count; ++i) run(i);
    ck(cudaEventRecord(b, s)); ck(cudaEventSynchronize(b));
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b));
    ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
    return ms * 1000.0 / count;
}
struct AbWeights {
    Buf<uint16_t> alpha{2560 * 48}, beta{2560 * 48};
    Buf<float> dt{48}, a{48};
};
int main(int argc, char** argv) try {
    if (argc != 2) return 2;
    std::ifstream in(argv[1]);
    if (!in) return 2;
    Weights weights;
    std::string name;
    Weight row;
    while (in >> name >> row.type >> row.off >> row.bytes >> row.path) weights[name].push_back(row);
    if (!in.eof()) return 2;
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool seg_on = enabled("STRATA_GDN_SEG"), a3_on = enabled("STRATA_GDN_A3");
    std::printf("STRATA_GDN_SEG=%d STRATA_GDN_A3=%d\n", int(seg_on), int(a3_on));
    cudaStream_t s;
    ck(cudaStreamCreate(&s));
    std::mt19937 gen(1209);
    std::normal_distribution<float> normal(0.f, 1.f);
    unsigned failures = 0;
    struct Shape { const char* name; int k, m, type; };
    for (auto c : { Shape{"qkv",2560,10240,2}, {"z",2560,6144,2}, {"out-q4",6144,2560,2},
                    {"out-q8",6144,2560,8}, {"ab",2560,96,30} }) {
        const bool ab = c.type == 30;
        std::vector<float> x(c.k * 8);
        for (float& f : x) f = normal(gen);
        Buf<float> xg(x.size()), y(c.m * 8);
        Buf<uint8_t> xq(K::native_q8_1_bytes(c.k, 8));
        ck(cudaMemcpy(xg.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        K::native_quantize_q8_1(xg.p, xq.p, c.k, 8, s);
        std::vector<const void*> w;
        std::vector<AbWeights*> wa;
        if (ab) {
            const char* n[] = {"alpha", "beta", "dt", "a"};
            for (size_t j = 0; j < weights.at(n[0]).size(); ++j) {
                auto* p = new AbWeights;
                void* d[] = {p->alpha.p, p->beta.p, p->dt.p, p->a.p};
                for (int q = 0; q < 4; ++q) {
                    const auto& ww = weights.at(n[q]).at(j);
                    if (ww.type != (q < 2 ? 30 : 0) || ww.bytes != (q < 2 ? 2560 * 48 * 2 : 48 * 4)) return 2;
                    upload(ww, d[q]);
                }
                wa.push_back(p);
            }
        } else {
            for (const auto& ww : weights.at(c.name)) {
                if (ww.type != c.type || ww.bytes != K::native_mmvq_weight_bytes(c.type, c.k, c.m)) return 2;
                void* p = nullptr;
                ck(cudaMalloc(&p, ww.bytes));
                upload(ww, p); w.push_back(p);
            }
        }
        const int sets = ab ? wa.size() : w.size();
        if (sets == 0) return 2;
        for (int t = 1; t <= K::kVerifyMaxT; ++t) {
            const bool expect_seg = seg_on && c.type == 2 && t >= 2 && t <= (c.k == 6144 ? 3 : 5);
            auto call = [&](int i, int tt, int tok) {
                if (ab) {
                    const auto* a = wa[i % sets];
                    K::gdn_ab_multi(xg.p + tok * c.k, a->alpha.p, a->beta.p, a->dt.p, a->a.p,
                                    y.p + tok * 48, y.p + t * 48 + tok * 48, c.k, 48, tt, s);
                } else {
                    const void* x0 = xq.p + tok * K::native_q8_1_bytes(c.k);
                    float* yy = y.p + tok * c.m;
                    const bool seg = K::native_gdn_segments(c.type, w[i % sets], x0, yy, c.k, c.m, tt, s);
                    if (seg != (tt != 1 && expect_seg)) throw std::runtime_error("unexpected GDN segments dispatch");
                    if (!seg) K::native_mmvq(c.type, w[i % sets], x0, yy, c.k, c.m, tt, s);
                }
            };
            unsigned mismatches = 0, graph_diff = 0;
            std::vector<uint32_t> all;
            uint64_t first = 0;
            for (int i = 0; i < sets; ++i) {
                for (int j = 0; j < t; ++j) call(i, 1, j);
                const auto ref = read(y.p, t * c.m, s);
                ck(cudaMemsetAsync(y.p, 0xff, t * c.m * sizeof(float), s));
                call(i, t, 0);
                const auto result = read(y.p, t * c.m, s);
                mismatches += diff(ref, result);
                all.insert(all.end(), result.begin(), result.end());
                if (i == 0) {
                    first = hash(result);
                    ck(cudaMemsetAsync(y.p, 0xff, t * c.m * sizeof(float), s));
                    ck(cudaStreamSynchronize(s));
                    capture_once([&] { call(i, t, 0); }, s, ab ? (a3_on && t > 1 ? 96 : 48) : 0);
                    graph_diff = diff(ref, read(y.p, t * c.m, s));
                }
            }
            failures += mismatches + graph_diff;
            // Cycle through weight sets to measure streaming rather than an L2-hot resident matrix.
            double times[5];
            for (double& us : times) us = timed([&](int i) { call(i, t, 0); }, s, 256);
            std::sort(times, times + 5);
            std::printf("%-7s T %d sets %d %.3f us diff %u graphdiff %u hash %016llx first %016llx\n",
                        c.name, t, sets, times[2], mismatches, graph_diff,
                        (unsigned long long) hash(all), (unsigned long long) first);
        }
        for (auto* p : wa) delete p;
        for (auto* p : w) ck(cudaFree(const_cast<void*>(p)));
    }
    ck(cudaStreamDestroy(s));
    return failures != 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
