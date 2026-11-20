// src/kernels/head_split_bench.cpp - the verify head (Q6_K 248320 x 2560, T columns) split over the last card A and
// peer cards: A sends the q8_1 input to each peer's uncached inbox, computes its rows [0, XA) while each peer computes
// its rows into a local buffer and copies them into A's logits (one 16-byte-store pass), then flags A; A waits for the
// flags and runs the greedy pick + nonfinite guard (argmax_rows_nf) as the verifier does. Compared bitwise with the
// one-card head (native_mmvq) on rotating inputs, timed from A's publish to A's pick (graphs, as in the engine).
//
//     build/head_split_bench T A peers fracA iters uncached_logits
//     e.g. head_split_bench 2 3 2 0.55 200 0      (peers: comma list, e.g. 0,1,2; fracA: A's share of the rows)
#include <hip/hip_runtime.h>

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace K = strata::kernels;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

namespace {
constexpr int V = 248320, N = 2560, QK = 256, BLK = 210, MAXP = 3, KX = 4;
constexpr std::size_t ROWB = std::size_t(N / QK) * BLK;   // 2100 B a row
constexpr int XQB = (N / 32) * 36;                        // q8_1 bytes a column (2880)

__device__ inline unsigned ld_sys(const unsigned* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline void st_sys(unsigned* p, unsigned v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
// wait until *f reaches want (wrapping compare); 2 s timeout -> err bit (a lost peer must not hang the card)
__device__ inline void spin_ge(const unsigned* f, unsigned want, unsigned* err) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    while (int(ld_sys(f) - want) < 0) {
        __builtin_amdgcn_s_sleep(2);
        if (__builtin_amdgcn_s_memrealtime() - t0 > 200000000ull) { atomicOr(err, 1u); return; }
    }
}

struct Peers {
    int n;
    uint4* inbox[MAXP];       // on peer p (uncached): the q8_1 input
    unsigned* inflag[MAXP];   // on peer p (uncached): A's window counter
    const unsigned* done[MAXP];   // on A (uncached): peer p's counter after its rows landed
};

// A, 1 block: next window number, the input to every peer, then each peer's flag
__global__ void k_publish(const uint4* __restrict__ xq, int n16, Peers P, unsigned* ctr) {
    const unsigned seq = ctr[0] + 1;
    for (int p = 0; p < P.n; ++p)
        for (int i = threadIdx.x; i < n16; i += blockDim.x) P.inbox[p][i] = xq[i];
    __threadfence_system();
    __syncthreads();
    if (int(threadIdx.x) < P.n) st_sys(P.inflag[threadIdx.x], seq);
    if (threadIdx.x == 0) ctr[0] = seq;
}
// A, 1 block: every peer's rows are in
__global__ void k_wait_done(Peers P, const unsigned* ctr, unsigned* err) {
    if (int(threadIdx.x) < P.n) spin_ge(P.done[threadIdx.x], ctr[0], err);
}
// peer, 1 block: wait for A's input, copy it into cached local memory
__global__ void k_peer_in(const uint4* inbox, const unsigned* inflag, uint4* xq, int n16, const unsigned* ctr,
                          unsigned* err) {
    if (threadIdx.x == 0) spin_ge(inflag, ctr[0] + 1, err);
    __syncthreads();
    for (int i = threadIdx.x; i < n16; i += blockDim.x) xq[i] = __builtin_nontemporal_load(inbox + i);
}
// peer: its rows (T columns of n floats, local) into A's logits at column stride V from row r0; the last block flags A
__global__ void k_peer_out(const float4* __restrict__ y, int n4, int T, float* a_logits, int r0, unsigned* ctr,
                           unsigned* count, unsigned* done) {
    const int total = n4 * T;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int t = i / n4, k = i - t * n4;
        reinterpret_cast<float4*>(a_logits + std::size_t(t) * V + r0)[k] = y[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (atomicAdd(count, 1u) == gridDim.x - 1) {
            *count = 0;
            const unsigned seq = ctr[0] + 1;
            ctr[0] = seq;
            __threadfence_system();
            st_sys(done, seq);
        }
    }
}
__global__ void k_cmp(const float* a, const float* b, int n, unsigned* bad) {
    unsigned c = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        c += __float_as_uint(a[i]) != __float_as_uint(b[i]);
    if (c) atomicAdd(bad, c);
}
__global__ void k_cmp_i(const int32_t* a, const int32_t* b, int n, unsigned* bad) {
    if (int(threadIdx.x) < n && a[threadIdx.x] != b[threadIdx.x]) atomicAdd(bad, 1u);
}
__global__ void k_poison(float* a, int n, float v) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) a[i] = v;
}

uint32_t rng_state = 12345;
inline uint32_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }
uint16_t f2h(float f) {   // positive normal floats only
    uint32_t u; std::memcpy(&u, &f, 4);
    return uint16_t(((u >> 16) & 0x8000) | ((((u >> 23) & 0xff) - 127 + 15) << 10) | ((u >> 13) & 0x3ff));
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int T = argc > 1 ? std::atoi(argv[1]) : 1;
    const int A = argc > 2 ? std::atoi(argv[2]) : 3;
    std::vector<int> peers;
    {
        std::string s = argc > 3 ? argv[3] : "2";
        for (size_t p = 0; p < s.size();) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); peers.push_back(std::atoi(s.substr(p, q - p).c_str())); p = q + 1; }
    }
    const double fracA = argc > 4 ? std::atof(argv[4]) : 0.55;
    const int iters = argc > 5 ? std::atoi(argv[5]) : 200;
    const int unc_logits = argc > 6 ? std::atoi(argv[6]) : 0;
    const int NP = int(peers.size());
    if (T < 1 || T > 4 || NP < 1 || NP > MAXP) { std::fprintf(stderr, "T 1..4, 1..3 peers\n"); return 1; }

    // rows: A [0, XA), peer k [b[k], b[k+1]) - multiples of 64
    std::vector<int> b(NP + 1);
    const int XA = (int(fracA * V) / 64) * 64;
    b[0] = XA;
    for (int k = 1; k < NP; ++k) b[k] = XA + ((int((double(V - XA) * k) / NP)) / 64) * 64;
    b[NP] = V;
    std::printf("T=%d A=%d peers=", T, A);
    for (int k = 0; k < NP; ++k) std::printf("%d[%d,%d) ", peers[k], b[k], b[k + 1]);
    std::printf("A rows %d (%.3f) uncached_logits=%d\n", XA, double(XA) / V, unc_logits);

    // weights: random Q6_K blocks with small positive scales d
    std::vector<uint8_t> hw(ROWB * V);
    for (std::size_t i = 0; i < hw.size(); i += 4) { const uint32_t r = rnd(); std::memcpy(&hw[i], &r, std::min<std::size_t>(4, hw.size() - i)); }
    for (std::size_t blk = 0; blk < std::size_t(V) * (N / QK); ++blk) {
        const uint16_t d = f2h(0.0005f + float(rnd() % 1000) * 1e-5f);
        std::memcpy(&hw[blk * BLK + 208], &d, 2);
    }
    for (int d : peers) {
        CK(hipSetDevice(A)); { hipError_t e = hipDeviceEnablePeerAccess(d, 0); if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CK(e); }
        CK(hipSetDevice(d)); { hipError_t e = hipDeviceEnablePeerAccess(A, 0); if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CK(e); }
        (void) hipGetLastError();
    }

    // ---- card A
    CK(hipSetDevice(A));
    hipStream_t sA; CK(hipStreamCreateWithFlags(&sA, hipStreamNonBlocking));
    uint8_t* wA; CK(hipMalloc(&wA, hw.size())); CK(hipMemcpy(wA, hw.data(), hw.size(), hipMemcpyHostToDevice));
    float* xf; CK(hipMalloc(&xf, sizeof(float) * T * N * KX));
    {
        std::vector<float> hx(std::size_t(T) * N * KX);
        for (auto& v : hx) v = (float(rnd() % 20001) - 10000.f) * 1e-4f;
        CK(hipMemcpy(xf, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    }
    uint8_t* xq; CK(hipMalloc(&xq, std::size_t(XQB) * T * KX));
    for (int k = 0; k < KX; ++k) K::native_quantize_q8_1(xf + std::size_t(k) * T * N, xq + std::size_t(k) * XQB * T, N, T, sA);
    float *ref, *logits;
    CK(hipMalloc(&ref, sizeof(float) * T * V * KX));
    if (unc_logits) CK(hipExtMallocWithFlags((void**) &logits, sizeof(float) * T * V, hipDeviceMallocUncached));
    else CK(hipMalloc(&logits, sizeof(float) * T * V));
    void* scr; const std::size_t scr_b = K::argmax_rows_scratch_bytes(T);
    CK(hipMalloc(&scr, scr_b)); CK(hipMemset(scr, 0, scr_b));
    int32_t *ids, *ref_ids; uint32_t* nf;
    CK(hipMalloc(&ids, 4 * 8)); CK(hipMalloc(&ref_ids, 4 * 8 * KX)); CK(hipMalloc(&nf, 4096)); CK(hipMemset(nf, 0, 4096));
    unsigned *ctrA, *bad, *err;
    CK(hipMalloc(&ctrA, 4)); CK(hipMemset(ctrA, 0, 4));
    CK(hipHostMalloc(&bad, 16, hipHostMallocMapped)); std::memset(bad, 0, 16);
    CK(hipHostMalloc(&err, 16, hipHostMallocMapped)); std::memset(err, 0, 16);
    for (int k = 0; k < KX; ++k) {
        K::native_mmvq(14, wA, xq + std::size_t(k) * XQB * T, ref + std::size_t(k) * T * V, N, V, T, sA);
        K::argmax_rows_nf(ref + std::size_t(k) * T * V, T, V, scr, ref_ids + 8 * k, nf, sA);
    }
    CK(hipStreamSynchronize(sA));

    Peers P{};
    P.n = NP;
    std::vector<unsigned*> doneA(NP);
    for (int k = 0; k < NP; ++k) { CK(hipExtMallocWithFlags((void**) &doneA[k], 256, hipDeviceMallocUncached)); CK(hipMemset(doneA[k], 0, 256)); P.done[k] = doneA[k]; }

    // ---- peers
    struct PeerSt { hipStream_t s; uint8_t* w; uint4* xq; float* y; unsigned *ctr, *count; hipGraphExec_t g; };
    std::vector<PeerSt> ps(NP);
    for (int k = 0; k < NP; ++k) {
        const int d = peers[k], n = b[k + 1] - b[k];
        CK(hipSetDevice(d));
        PeerSt& q = ps[k];
        CK(hipStreamCreateWithFlags(&q.s, hipStreamNonBlocking));
        CK(hipMalloc(&q.w, ROWB * n)); CK(hipMemcpy(q.w, hw.data() + ROWB * b[k], ROWB * n, hipMemcpyHostToDevice));
        CK(hipExtMallocWithFlags((void**) &P.inbox[k], XQB * T, hipDeviceMallocUncached));
        CK(hipExtMallocWithFlags((void**) &P.inflag[k], 256, hipDeviceMallocUncached)); CK(hipMemset(P.inflag[k], 0, 256));
        CK(hipMalloc(&q.xq, XQB * T)); CK(hipMalloc(&q.y, sizeof(float) * T * n));
        CK(hipMalloc(&q.ctr, 4)); CK(hipMemset(q.ctr, 0, 4)); CK(hipMalloc(&q.count, 4)); CK(hipMemset(q.count, 0, 4));
        CK(hipDeviceSynchronize());
        CK(hipStreamBeginCapture(q.s, hipStreamCaptureModeRelaxed));
        k_peer_in<<<1, 256, 0, q.s>>>(P.inbox[k], P.inflag[k], q.xq, XQB * T / 16, q.ctr, err);
        if (!K::native_q6_k_head_rows(q.w, q.xq, q.y, N, n, n, T, q.s)) { std::fprintf(stderr, "head_rows refused\n"); return 1; }
        k_peer_out<<<120, 256, 0, q.s>>>(reinterpret_cast<const float4*>(q.y), n / 4, T, logits, b[k], q.ctr, q.count, doneA[k]);
        hipGraph_t g; CK(hipStreamEndCapture(q.s, &g)); CK(hipGraphInstantiate(&q.g, g, nullptr, nullptr, 0));
    }

    // ---- A graphs: the one-card head (ref timing) and the split head, per input k
    CK(hipSetDevice(A));
    hipGraphExec_t g1[KX], g2[KX];
    for (int k = 0; k < KX; ++k) {
        const uint8_t* x = xq + std::size_t(k) * XQB * T;
        hipGraph_t g;
        CK(hipStreamBeginCapture(sA, hipStreamCaptureModeRelaxed));
        K::native_mmvq(14, wA, x, logits, N, V, T, sA);
        K::argmax_rows_nf(logits, T, V, scr, ids, nf, sA);
        CK(hipStreamEndCapture(sA, &g)); CK(hipGraphInstantiate(&g1[k], g, nullptr, nullptr, 0));
        CK(hipStreamBeginCapture(sA, hipStreamCaptureModeRelaxed));
        k_publish<<<1, 256, 0, sA>>>(reinterpret_cast<const uint4*>(x), XQB * T / 16, P, ctrA);
        if (!K::native_q6_k_head_rows(wA, x, logits, N, XA, V, T, sA)) { std::fprintf(stderr, "head_rows refused\n"); return 1; }
        k_wait_done<<<1, 64, 0, sA>>>(P, ctrA, err);
        K::argmax_rows_nf(logits, T, V, scr, ids, nf, sA);
        CK(hipStreamEndCapture(sA, &g)); CK(hipGraphInstantiate(&g2[k], g, nullptr, nullptr, 0));
    }
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    double sum[2] = {0, 0};
    float mn[2] = {1e9f, 1e9f};
    unsigned bad_l[2] = {0, 0}, bad_i[2] = {0, 0};
    for (int it = 0; it < iters + 10; ++it) {
        for (int arm = 0; arm < 2; ++arm) {
            const int k = it % KX;
            CK(hipSetDevice(A));
            k_poison<<<240, 256, 0, sA>>>(logits, T * V, -1.0f);   // a stale or missing row shows
            if (arm == 1)
                for (int p = 0; p < NP; ++p) { CK(hipSetDevice(peers[p])); CK(hipGraphLaunch(ps[p].g, ps[p].s)); }
            CK(hipSetDevice(A));
            CK(hipEventRecord(e0, sA));
            CK(hipGraphLaunch(arm ? g2[k] : g1[k], sA));
            CK(hipEventRecord(e1, sA));
            bad[0] = bad[1] = 0;
            k_cmp<<<240, 256, 0, sA>>>(logits, ref + std::size_t(k) * T * V, T * V, bad);
            k_cmp_i<<<1, 64, 0, sA>>>(ids, ref_ids + 8 * k, T, bad + 1);
            CK(hipStreamSynchronize(sA));
            for (int p = 0; arm == 1 && p < NP; ++p) { CK(hipSetDevice(peers[p])); CK(hipStreamSynchronize(ps[p].s)); }
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            if (it >= 10) { sum[arm] += ms; mn[arm] = std::min(mn[arm], ms); bad_l[arm] += bad[0]; bad_i[arm] += bad[1]; }
            if (*err) { std::fprintf(stderr, "spin timeout at it %d arm %d\n", it, arm); return 2; }
        }
    }
    std::printf("one-card head+pick: avg %.4f min %.4f ms | split: avg %.4f min %.4f ms | saved %.4f ms | "
                "mismatch logits %u/%u ids %u/%u (of %d iters)\n",
                sum[0] / iters, mn[0], sum[1] / iters, mn[1], (sum[0] - sum[1]) / iters, bad_l[0], bad_l[1], bad_i[0],
                bad_i[1], iters);
    return 0;
}
