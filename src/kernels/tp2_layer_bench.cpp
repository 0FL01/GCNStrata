// tp2_layer_bench.cpp - cold Q4_0/BF16 verify-layer stacks on gfx906: one card vs two head/FF-sharded cards.
// Configure Release with -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906.
// Build: cmake --build <dir> --target tp2_layer_bench
//   TP2_DEVS=0,1 <dir>/tp2_layer_bench [ref|tp2|both=both] [T=1|2] [layers=12] [reps=50] [ctx=4096]
// ctx is the prefix length before the T rows. INT8 KV and prefix indexer keys are already resident.
// Timings are launch-through-sync wall means. Class-only graphs run just that class across the same stack;
// skipped producer outputs are frozen scratch left by the full graph. So is the input on the receive side of
// each removed hop in tp2-no-hops. These timings include launch overhead and are not an additive profile.
// Synthetic compute/communication timings only: no scheduler, host experts, acceptance, KV append or head.

#include <hip/hip_runtime.h>

#include "strata/core/geometry.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

void check(hipError_t e, const char* call, int line) {
    if (e == hipSuccess) return;
    std::fprintf(stderr, "tp2_layer_bench:%d: %s: %s\n", line, call, hipGetErrorString(e));
    std::exit(1);
}
#define CK(call) check((call), #call, __LINE__)

[[noreturn]] void fail(const char* msg) {
    std::fprintf(stderr, "tp2_layer_bench: %s\n", msg);
    std::exit(1);
}

constexpr auto G = strata::core::kCompiledGeometry;
constexpr int N = G.n_embd, HC = G.hc, LR = G.hc_lr, SHARD = N / 2;
constexpr int SMALL = LR + HC, TOP = G.n_expert_used, NE = G.n_expert, FF = G.n_ff;
constexpr int Q4_0 = 2, MAX_T = 2, MAX_CTX = 262144;
constexpr int HD = G.head_dim, IQ = G.idx_q_heads, ID = G.idx_key_dim;
constexpr int S = G.ssm_state_size, TAPS = G.ssm_d_conv;
constexpr float EPS = (float) G.rms_eps;
constexpr int HC_THREADS = 256, HOP_THREADS = 1024, WARM = 5;

using Packet = unsigned long long;
static_assert(sizeof(Packet) == 8 && sizeof(float) == 4 && N % 2 == 0);

// tpcomm.hip's LL protocol: the sequence travels in every word, not in a separate flag.
// Fixed stride between parities even when payload sizes change. The sender can't reach seq+2 until
// the receiver has started seq+1, which requires all of the receiver's seq loads to have finished.
__global__ void hop_ll(const float* src, float* out, int width, int rows, int rank, bool gather,
                       Packet* recv, Packet* peer_recv, unsigned* counter) {
    const unsigned seq = *counter + 1;
    const int par = int(seq & 1u) * MAX_T * N;
    const int count = width * rows;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const Packet word = (Packet(seq) << 32) | __float_as_uint(src[i]);
        __hip_atomic_store(peer_recv + par + i, word, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        Packet word;
        do {
            word = __hip_atomic_load(recv + par + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        } while (unsigned(word >> 32) != seq);
        const float remote = __uint_as_float(unsigned(word));
        if (gather) {
            const int base = 2 * width * (i / width) + i % width;
            out[base + rank * width] = src[i];
            out[base + (1 - rank) * width] = remote;
        } else {
            out[i] = rank == 0 ? src[i] + remote : remote + src[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) *counter = seq;
}

__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Local RMS for the down GEMV. With just LR+HC exchanged floats, we can't send HC separate down
// projections to rescale each by its global RMS. Use the global RMS in the subsequent mix instead.
__global__ void hc_prepare(const float* r, float* r_out, const float* bo, const float* inj,
                           const float* wn, float* xn, float* partial) {
    const int t = blockIdx.x / HC, c = blockIdx.x % HC, thread = threadIdx.x;
    const int base = (t * HC + c) * SHARD;
    const float scale = bo ? 2.0f * sigmoid(inj[t * HC + c] / HC) : 0.0f;
    __shared__ float sum[HC_THREADS];
    float ss = 0.0f;
    for (int d = thread; d < SHARD; d += blockDim.x) {
        float v = r[base + d];
        if (bo) {
            v += bo[t * N + d] * scale;  // bo already offset by rank*SHARD
            r_out[base + d] = v;
        }
        ss += v * v;
    }
    sum[thread] = ss;
    __syncthreads();
    for (int half = HC_THREADS / 2; half > 0; half /= 2) {
        if (thread < half) sum[thread] += sum[thread + half];
        __syncthreads();
    }
    if (thread == 0) partial[t * SMALL + LR + c] = sum[0];
    const float rs = rsqrtf(sum[0] / SHARD + EPS);
    const float* updated = bo ? r_out : r;
    for (int d = thread; d < SHARD; d += blockDim.x)
        xn[base + d] = updated[base + d] * wn[c * SHARD + d] * rs;
}

__global__ void hc_activate(const float* merged, float* lo, float* inject) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= LR) return;
    const float x = merged[t * SMALL + i] / HC;
    lo[t * LR + i] = x * sigmoid(x);
    if (i < HC) inject[t * HC + i] = x;  // synthetic pending-write coefficient, no separate TP injection matvec
}

__global__ void hc_mix(const float* r, const float* wn, const float* up, const float* merged,
                       float* slice) {
    const int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= SHARD) return;
    float acc = 0.0f;
    for (int c = 0; c < HC; ++c) {
        const int i = (t * HC + c) * SHARD + d;
        const float rs = rsqrtf(merged[t * SMALL + LR + c] / N + EPS);
        acc += r[i] * wn[c * SHARD + d] * rs * sigmoid(up[i]);
    }
    slice[t * SHARD + d] = acc / HC;
}

struct DeviceMem {
    explicit DeviceMem(int device) : dev(device) {}
    DeviceMem(const DeviceMem&) = delete;
    DeviceMem& operator=(const DeviceMem&) = delete;
    ~DeviceMem() {
        select();
        for (void* p : allocations) CK(hipFree(p));
    }
    void select() const { CK(hipSetDevice(dev)); }
    template <typename T> T* alloc(size_t n, bool uncached = false) {
        void* p = nullptr;
        const size_t bytes = n * sizeof(T);
        if (uncached) CK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
        else CK(hipMalloc(&p, bytes));
        allocations.push_back(p);
        total += bytes;
        return static_cast<T*>(p);
    }
    template <typename T> T* zero(size_t n, bool uncached = false) {
        T* p = alloc<T>(n, uncached);
        CK(hipMemset(p, 0, n * sizeof(T)));
        return p;
    }
    template <typename T> T* upload(const std::vector<T>& h) {
        T* p = alloc<T>(h.size());
        CK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
        return p;
    }
    const int dev;
    size_t total = 0;
    std::vector<void*> allocations;
};

struct Fixture {
    DeviceMem& m;
    std::mt19937 rng;
    float real(float scale) { return scale * (float(int(rng() >> 16) - 32768) / 32768.0f); }
    float* floats(size_t n, float scale, float bias = 0.0f) {
        std::vector<float> h(n);
        for (float& x : h) x = bias + real(scale);
        return m.upload(h);
    }
    uint16_t* bf16(int in, int out) {
        std::vector<uint16_t> h(size_t(in) * out);
        const float scale = in >= N ? 0.035f : 0.07f;
        for (uint16_t& x : h) {
            const float f = real(scale);
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            x = uint16_t((bits + 0x7fff + ((bits >> 16) & 1u)) >> 16);
        }
        return m.upload(h);
    }
    // As in b6_mmvq_bench: arbitrary packed nibbles, finite FP16 scale at the start of every 18-byte block.
    void fill_q4(uint8_t* dst, size_t bytes, int in) {
        const unsigned exponent = in < 1024 ? 8 : (in > N ? 6 : 7);
        if (bytes == 0 || bytes % 18 != 0) fail("invalid synthetic Q4_0 allocation size");
        for (size_t b = 0; b < bytes; b += 18) {
            const uint16_t d = uint16_t((exponent << 10) | (rng() & 0x3ffu));
            std::memcpy(dst + b, &d, sizeof(d));
            for (int i = 2; i < 18; i += 4) {
                const uint32_t codes = uint32_t(rng());
                std::memcpy(dst + b + i, &codes, sizeof(codes));
            }
        }
    }
    uint8_t* q4(int in, int out) {
        std::vector<uint8_t> h(K::native_mmvq_weight_bytes(Q4_0, in, out));
        fill_q4(h.data(), h.size(), in);
        return m.upload(h);
    }
    int8_t* kv_codes(size_t n) {
        std::vector<int8_t> h(n);
        for (auto& x : h) x = int8_t(int(rng() % 255u) - 127);
        return m.upload(h);
    }
    uint16_t* kv_scales(size_t n) {
        std::vector<uint16_t> h(n);
        for (auto& x : h) x = uint16_t((6u << 10) | (rng() & 0x3ffu));
        return m.upload(h);
    }
};

struct HcWeights {
    const float* norm = nullptr;
    const uint16_t *down = nullptr, *up = nullptr, *inject = nullptr;
};

struct ExpertPlan {
    const Packet* ptr = nullptr;
    const int32_t *start = nullptr, *count = nullptr, *dst = nullptr, *tok = nullptr;
    int groups = 0, entries = 0;
};

struct Layer {
    HcWeights hc[2];
    const uint16_t *router = nullptr, *shared_gate = nullptr;
    K::NativeSharedWeights shared;
    ExpertPlan experts;
    const uint8_t *qkv = nullptr, *z = nullptr, *out = nullptr;
    const uint16_t *alpha = nullptr, *beta = nullptr;
    const float *conv = nullptr, *hist = nullptr, *dt = nullptr, *decay = nullptr, *gdn_norm = nullptr;
    float* state = nullptr;
    const uint8_t *q = nullptr, *k = nullptr, *v = nullptr;
    const uint16_t *idx_k = nullptr, *idx_q = nullptr;
    const float *q_norm = nullptr, *idx_q_norm = nullptr, *idx_k_norm = nullptr;
    K::QsaIndexerBuffers indexer;
    K::QsaAttnPools pools;
};

struct HcScratch {
    float *mixed = nullptr, *block = nullptr, *reduced = nullptr, *inj = nullptr;
    float *partial = nullptr, *small_sum = nullptr, *slice = nullptr;
};

// One ordered stream per rank. Scratch is reused across layers like the verifier; weights/state are not.
struct Card {
    Card(int dev, int r, bool tp, int t, int nl, int ctx, bool half_ff);
    ~Card() {
        mem.select();
        CK(hipStreamDestroy(stream));
    }
    DeviceMem mem;
    hipStream_t stream = nullptr;
    const int rank, T, split, D, HK, HV, C, V, NH, NKV, Q, KV, sh_ff;
    const bool whole_experts;
    const size_t q8_bytes;
    const K::NativeExpertLayout expert_layout;
    const int max_blocks = MAX_CTX / G.idx_block + 2;
    K::QsaShapes qsa = K::qsa_real_shapes();
    K::RopeScaling rope;
    int cap = 0;
    std::vector<Layer> layers;
    size_t weight_bytes = 0;
    unsigned *seq = nullptr, *hc_sync = nullptr;
    Packet *recv = nullptr, *peer = nullptr;
    float *r0 = nullptr, *r = nullptr, *xn = nullptr, *lo = nullptr, *rs = nullptr, *hc_up = nullptr;
    HcScratch hc[2];
    void *attn_q8 = nullptr, *out_q8 = nullptr, *ffn_q8 = nullptr, *shared_q8 = nullptr, *expert_tmp = nullptr;
    float *qkv = nullptr, *h = nullptr, *gate = nullptr, *beta = nullptr, *z = nullptr, *y = nullptr;
    float *logits = nullptr, *weights = nullptr, *parts = nullptr, *shared = nullptr;
    float *sh_g = nullptr, *sh_gate = nullptr, *sh_up = nullptr;
    int32_t *ids = nullptr, *pub_ids = nullptr;
    float *pub_x = nullptr, *pub_w = nullptr;
    uint32_t* pub_seq = nullptr;
    bool fused_router = false;
    int32_t *steps = nullptr, *pos_q = nullptr, *pos_idx = nullptr, *selected = nullptr;
    float *qfull = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr;
    float *idx_raw = nullptr, *idx_query = nullptr, *scores = nullptr, *attn_tmp = nullptr;
    float *attn = nullptr, *gated = nullptr;
};

ExpertPlan make_plan(Fixture& f, const K::NativeExpertLayout& layout, int T, bool whole, int rank) {
    // 6 experts are reused across both rows; token 0 picks 0..9 and token 1 picks 4..13.
    // Fallback assigns even/odd expert IDs to the two ranks: 5 picks/row and 5 or 7 groups/rank.
    std::vector<Packet> ptr;
    std::vector<int32_t> start, dst, tok;
    std::vector<uint8_t> blob(layout.bytes);
    for (int expert = 0; expert < (T == 1 ? TOP : 14); ++expert) {
        if (whole && expert % 2 != rank) continue;
        start.push_back(int32_t(dst.size()));
        for (int t = 0; t < T; ++t) {
            const int pick = expert - 4 * t;
            if (pick >= 0 && pick < TOP) {
                dst.push_back(t * TOP + pick);
                tok.push_back(t);
            }
        }
        f.fill_q4(blob.data(), layout.down_off, N);
        f.fill_q4(blob.data() + layout.down_off, layout.bytes - layout.down_off, int(layout.n_ff));
        ptr.push_back(reinterpret_cast<Packet>(f.m.upload(blob)));
    }
    start.push_back(int32_t(dst.size()));
    const int32_t ng = int32_t(ptr.size());
    return {f.m.upload(ptr), f.m.upload(start), f.m.upload(std::vector<int32_t>{ng}),
            f.m.upload(dst), f.m.upload(tok), ng, int(dst.size())};
}

Card::Card(int dev, int rnk, bool tp, int t, int nl, int ctx, bool half_ff)
    : mem(dev), rank(rnk), T(t), split(tp ? 2 : 1), D(N * HC / split),
      HK(G.ssm_k_heads / split), HV(G.ssm_v_heads / split), C(S * (2 * HK + HV)), V(S * HV),
      NH(G.n_head / split), NKV(G.n_head_kv / split), Q(NH * HD), KV(NKV * HD), sh_ff(G.n_ff_shexp / split),
      whole_experts(tp && !half_ff), q8_bytes(K::native_q8_1_bytes(N, T)),
      expert_layout(K::native_expert_layout(Q4_0, Q4_0, N, whole_experts ? FF : FF / split)), layers(nl) {
    mem.select();
    CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    if (tp) {
        // The slots live on the receiving card. Fine-grained/coarse VRAM can stay invisible to a spinner on MI50.
        recv = mem.zero<Packet>(2 * MAX_T * N, true);
        seq = mem.zero<unsigned>(1);
    } else {
        K::fused_gr_check();  // same gfx906 variant selection as Verifier::init, before capture
        CK(hipGetLastError());
    }
    qsa.n_head = NH;
    qsa.n_head_kv = NKV;
    cap = int(K::qsa_selection_width(K::kTopkMaxCells, qsa));
    const int pages = (ctx + T + int(qsa.page_size) - 1) / int(qsa.page_size);
    int32_t* page_table = nullptr;
    if (nl >= G.qsa_interval) {
        std::vector<int32_t> table(pages), step(T * K::kStepCount), pq(T * NH), pi(T * IQ);
        std::iota(table.begin(), table.end(), 0);
        for (int i = 0; i < T; ++i) {
            K::qsa_step_fill(step.data() + i * K::kStepCount, ctx + i, qsa);
            for (int head = 0; head < NH; ++head) pq[i * NH + head] = ctx + i;
            for (int head = 0; head < IQ; ++head) pi[i * IQ + head] = ctx + i;
        }
        page_table = mem.upload(table);
        steps = mem.upload(step);
        pos_q = mem.upload(pq);
        pos_idx = mem.upload(pi);
    }
    Fixture local{mem, std::mt19937(20261119u + 991u * unsigned(rank))};
    // Separately seeded so replicated weights/state are identical on both ranks, despite sharded allocation sizes.
    Fixture replicated{mem, std::mt19937(83u)};
    r0 = local.floats(size_t(T) * D, 1.0f);
    r = mem.zero<float>(size_t(T) * D);
    xn = mem.zero<float>(size_t(T) * D);
    lo = mem.zero<float>(T * LR);
    rs = mem.zero<float>(T * HC);
    hc_up = tp ? mem.zero<float>(size_t(T) * D) : nullptr;
    hc_sync = mem.zero<unsigned>(K::kFusedGrSyncWords);
    for (auto& a : hc) {
        a.mixed = mem.zero<float>(T * N);
        a.block = mem.zero<float>(T * N);
        a.inj = mem.zero<float>(T * HC);
        if (tp) {
            a.reduced = mem.zero<float>(T * N);
            a.partial = mem.zero<float>(T * SMALL);
            a.small_sum = mem.zero<float>(T * SMALL);
            a.slice = mem.zero<float>(T * SHARD);
        } else a.reduced = a.block;
    }
    attn_q8 = mem.zero<uint8_t>(q8_bytes);
    out_q8 = mem.zero<uint8_t>(K::native_q8_1_bytes(V, T));  // V == Q at both shard factors
    ffn_q8 = mem.zero<uint8_t>(q8_bytes);
    shared_q8 = mem.zero<uint8_t>(q8_bytes);
    expert_tmp = mem.zero<uint8_t>(K::native_expert_scratch_bytes(T * TOP, expert_layout.n_ff));
    qkv = mem.zero<float>(T * C);
    h = mem.zero<float>(T * C);
    gate = mem.zero<float>(T * HV);
    beta = mem.zero<float>(T * HV);
    z = mem.zero<float>(T * V);
    y = mem.zero<float>(T * V);
    logits = mem.zero<float>(T * NE);
    weights = mem.zero<float>(T * TOP);
    ids = mem.zero<int32_t>(T * TOP);
    parts = mem.zero<float>(T * TOP * N); // unowned pick rows in the whole-expert fallback stay zero
    shared = mem.zero<float>(T * N);
    sh_g = mem.zero<float>(T);
    sh_gate = mem.zero<float>(T * sh_ff);
    sh_up = mem.zero<float>(T * sh_ff);
    pub_x = mem.zero<float>(T * N);
    pub_ids = mem.zero<int32_t>(T * TOP);
    pub_w = mem.zero<float>(T * TOP);
    pub_seq = mem.zero<uint32_t>(1);
    if (page_table) {
        qfull = mem.zero<float>(T * 2 * Q);
        qcur = mem.zero<float>(T * Q);
        kcur = mem.zero<float>(T * KV);
        vcur = mem.zero<float>(T * KV);
        idx_raw = mem.zero<float>(T * ID);
        idx_query = mem.zero<float>(T * IQ * ID);
        scores = mem.zero<float>(size_t(T) * max_blocks);
        selected = mem.zero<int32_t>(T * cap);
        attn_tmp = mem.zero<float>(T * K::qsa_decode_attn_scratch_floats(cap, qsa));
        attn = mem.zero<float>(T * Q);
        gated = mem.zero<float>(T * Q);
    }

    const size_t before_layers = mem.total;
    for (int i = 0; i < nl; ++i) {
        Layer& l = layers[i];
        for (auto& a : l.hc) {
            a.norm = local.floats(D, 0.04f, 1.0f);
            a.down = local.bf16(D, LR);
            a.up = local.bf16(LR, D);
            if (!tp) a.inject = local.bf16(D, HC);
        }
        l.router = replicated.bf16(N, NE);
        l.shared_gate = replicated.bf16(N, 1);
        l.shared.gate_type = l.shared.up_type = l.shared.down_type = Q4_0;
        l.shared.gate_data = local.q4(N, sh_ff);
        l.shared.up_data = local.q4(N, sh_ff);
        l.shared.down_data = local.q4(sh_ff, N);
        l.shared.q8_1 = shared_q8;
        l.experts = make_plan(local, expert_layout, T, whole_experts, rank);
        l.out = local.q4(V, N);
        if (strata::core::is_qsa_layer(G, i)) {
            l.q = local.q4(N, 2 * Q);
            l.k = local.q4(N, KV);
            l.v = local.q4(N, KV);
            l.idx_k = replicated.bf16(N, ID);
            l.idx_q = replicated.bf16(N, IQ * ID);
            l.q_norm = local.floats(HD, 0.04f, 1.0f);
            l.idx_q_norm = replicated.floats(ID, 0.04f, 1.0f);
            l.idx_k_norm = replicated.floats(ID, 0.04f, 1.0f);
        } else {
            l.qkv = local.q4(N, C);
            l.z = local.q4(N, V);
            l.alpha = local.bf16(N, HV);
            l.beta = local.bf16(N, HV);
            l.conv = local.floats(C * TAPS, 0.4f);
            l.dt = local.floats(HV, 0.2f, -1.0f);
            l.decay = local.floats(HV, 0.15f, -0.5f);
            l.gdn_norm = local.floats(S, 0.04f, 1.0f);
        }
    }
    weight_bytes = mem.total - before_layers;
    for (Layer& l : layers) {
        if (l.q) {
            // Keep the verifier score stride/capacity even at short contexts. Only completed prefix
            // blocks need seeded keys; append writes any blocks that complete in the measured window.
            const int active = ctx / int(qsa.idx_block);
            std::vector<float> keys(size_t(max_blocks) * ID, 0.0f);
            for (int j = 0; j < active * ID; ++j) keys[j] = replicated.real(0.15f);
            l.indexer.pooled = mem.upload(keys);
            l.indexer.dead = replicated.floats(ID, 0.15f);
            // Tail slots before ctx % idx_block are the read-only prefix on replay.
            l.indexer.tail = mem.upload(std::vector<float>((qsa.idx_block - 1) * ID, 0.125f));
            l.indexer.block_pos = mem.zero<int32_t>(1);
            const size_t kv_rows = size_t(pages) * NKV * qsa.page_size;
            l.pools.page_table = page_table;
            l.pools.k_q = local.kv_codes(kv_rows * HD);
            l.pools.v_q = local.kv_codes(kv_rows * HD);
            l.pools.k_scale = local.kv_scales(kv_rows * (HD / K::KV_Q8_GROUP));
            l.pools.v_scale = local.kv_scales(kv_rows * (HD / K::KV_Q8_GROUP));
        } else {
            l.hist = local.floats(C * (TAPS - 1), 0.2f);
            l.state = local.floats(size_t(S) * HV * S, 0.02f);
        }
    }
    CK(hipDeviceSynchronize());
    std::printf("  dev %d: %.1f MiB layer weights/plans, %.1f MiB total; experts FF=%lld, %d groups/%d entries per layer\n",
                mem.dev, weight_bytes / 1048576.0, mem.total / 1048576.0,
                (long long) expert_layout.n_ff, layers.front().experts.groups, layers.front().experts.entries);
}

enum Class {
    Hc, Quant, Qkv, Conv, Ab, Z, Rec, Out, Router, Shared, Experts, Combine,
    QsaKv, QsaQ, QsaIdx, QsaScore, QsaTopk, QsaAttn, QsaGate, Hop, ClassCount, All, NoHops
};
constexpr const char* class_names[] = {
    "hc", "quant", "qkv", "conv", "ab", "z", "rec", "out", "router", "shared", "experts", "combine",
    "qsa-kv", "qsa-q", "qsa-indexer", "qsa-scores", "qsa-topk", "qsa-attn", "qsa-gate", "hop"
};
static_assert(sizeof(class_names) / sizeof(*class_names) == ClassCount);

void capture_stack(Card& c, Class filter) {
    const auto stream = c.stream;
    const bool tp = c.split == 2;
    auto emit = [&](Class cls, auto fn) {
        if (filter != cls && filter != All && !(filter == NoHops && cls != Hop)) return;
        fn();
        CK(hipGetLastError());
    };
    auto bf16 = [&](const float* x, const uint16_t* w, float* y, int ni, int no, int stride = 0) {
        K::bf16_gemv_fp32_mmvf_multi(x, ni, w, y, stride ? stride : no, ni, no, c.T, stream);
    };
    auto mm = [&](const uint8_t* w, const void* x, float* y, int ni, int no) {
        K::native_mmvq(Q4_0, w, x, y, ni, no, c.T, stream);
    };
    auto quant = [&](const float* x, void* y, int width) {
        K::native_quantize_q8_1(x, y, width, c.T, stream);
    };
    auto hop = [&](const float* src, float* dst, int width, bool ag = false) {
        emit(Hop, [&] {
            hipLaunchKernelGGL(hop_ll, dim3(1), dim3(HOP_THREADS), 0, stream, src, dst, width,
                               c.T, c.rank, ag, c.recv, c.peer, c.seq);
        });
    };
    const bool fuse_nr = K::native_norm_rope_usable(HD, int(c.qsa.n_rot));
    auto norm_rope = [&](const float* x, int stride, const float* gamma, float* dst, int rows, int cols,
                         const int32_t* pos) {
        if (fuse_nr) {
            K::native_qsa_rms_norm_rope(x, stride, gamma, dst, rows, cols, int(c.qsa.n_rot), EPS, c.rope, pos, stream);
        } else {
            if (x != dst) K::copy_rows_strided(dst, x, rows, cols, stride, stream);
            K::native_qsa_rms_norm_weighted(dst, gamma, dst, cols, rows, EPS, stream);
            K::native_rope_apply(dst, dst, rows, cols, int(c.qsa.n_rot), c.rope, pos, stream);
        }
    };
    auto hc_read = [&](const HcWeights& w, int half, bool pending) {
        auto& a = c.hc[half];
        const auto& prev = c.hc[1 - half];
        if (!tp) {
            emit(Hc, [&] {
                K::FusedGrArgs args[MAX_T];
                for (int t = 0; t < c.T; ++t) {
                    auto& x = args[t];
                    x.R = x.R_out = c.r + t * c.D;
                    x.apply = pending;
                    x.bo_prev = prev.reduced + t * N;
                    x.inj_prev = prev.inj + t * HC;
                    x.w_norm = w.norm;
                    x.w_down = w.down;
                    x.w_up = w.up;
                    x.w_inject = w.inject;
                    x.eps = EPS;
                    x.lo = c.lo + t * LR;
                    x.rs = c.rs + t * HC;
                    x.inject_out = a.inj + t * HC;
                    x.mixed = a.mixed + t * N;
                }
                K::fused_gr_read_multi(args, c.T, c.xn, stream, nullptr, 0, c.hc_sync);
            });
            return;
        }
        emit(Hc, [&] {
            const float* bo = pending ? prev.reduced + c.rank * SHARD : nullptr;
            hipLaunchKernelGGL(hc_prepare, dim3(c.T * HC), dim3(HC_THREADS), 0, stream,
                               c.r, c.r, bo, prev.inj, w.norm, c.xn, a.partial);
            CK(hipGetLastError());
            bf16(c.xn, w.down, a.partial, c.D, LR, SMALL);
        });
        hop(a.partial, a.small_sum, SMALL);
        emit(Hc, [&] {
            hipLaunchKernelGGL(hc_activate, dim3((LR + HC_THREADS - 1) / HC_THREADS, c.T), dim3(HC_THREADS),
                               0, stream, a.small_sum, c.lo, a.inj);
            CK(hipGetLastError());
            bf16(c.lo, w.up, c.hc_up, LR, c.D);
            hipLaunchKernelGGL(hc_mix, dim3((SHARD + HC_THREADS - 1) / HC_THREADS, c.T), dim3(HC_THREADS),
                               0, stream, c.r, w.norm, c.hc_up, a.small_sum, a.slice);
        });
        hop(a.slice, a.mixed, SHARD, true);
    };

    for (size_t i = 0; i < c.layers.size(); ++i) {
        const auto& l = c.layers[i];
        hc_read(l.hc[0], 0, i != 0);
        auto& a = c.hc[0];
        emit(Quant, [&] { quant(a.mixed, c.attn_q8, N); });
        if (!l.q) {
            emit(Qkv, [&] { mm(l.qkv, c.attn_q8, c.qkv, N, c.C); });
            emit(Conv, [&] { K::gdn_conv_l2_multi(l.hist, c.qkv, l.conv, c.h, c.C, c.HK, EPS, c.T, stream); });
            emit(Ab, [&] {
                K::gdn_ab_multi(a.mixed, l.alpha, l.beta, l.dt, l.decay, c.gate, c.beta, N, c.HV, c.T, stream);
            });
            emit(Z, [&] { mm(l.z, c.attn_q8, c.z, N, c.V); });
            emit(Rec, [&] {
                K::gdn_step_norm_multi(l.state, c.h, c.C, c.gate, c.beta, c.z, l.gdn_norm, EPS,
                                       c.y, c.HK, c.HV, c.T, nullptr, stream);
            });
            emit(Out, [&] {
                quant(c.y, c.out_q8, c.V);
                mm(l.out, c.out_q8, a.block, c.V, N);
            });
        } else {
            emit(QsaKv, [&] {
                bf16(a.mixed, l.idx_k, c.idx_raw, N, ID);
                mm(l.k, c.attn_q8, c.kcur, N, c.KV);
                mm(l.v, c.attn_q8, c.vcur, N, c.KV);
                // The current K/V cells are pre-filled. Still project K and V and append the indexer key.
                K::native_qsa_indexer_append_steps(c.idx_raw, c.steps + K::kStepPos, K::kStepCount, c.T, 0,
                                                    l.idx_k_norm, EPS, l.indexer, c.qsa, MAX_CTX, c.rope, stream);
            });
            emit(QsaQ, [&] {
                mm(l.q, c.attn_q8, c.qfull, N, 2 * c.Q);
                norm_rope(c.qfull, 2 * HD, l.q_norm, c.qcur, c.T * c.NH, HD, c.pos_q);
            });
            emit(QsaIdx, [&] {
                bf16(a.mixed, l.idx_q, c.idx_query, N, IQ * ID);
                norm_rope(c.idx_query, ID, l.idx_q_norm, c.idx_query, c.T * IQ, ID, c.pos_idx);
            });
            emit(QsaScore, [&] {
                K::qsa_block_scores(l.indexer.pooled, l.indexer.dead, c.idx_query, c.steps, c.T, c.max_blocks, c.qsa, c.scores, stream);
            });
            emit(QsaTopk, [&] {
                K::qsa_block_topk(c.scores, c.steps, c.T, c.max_blocks, c.cap, c.qsa, c.selected, stream);
            });
            emit(QsaAttn, [&] {
                K::qsa_decode_attn_batch(c.qcur, l.pools, c.selected, c.steps, c.cap, c.qsa, c.attn_tmp, c.attn, c.T, stream);
            });
            emit(QsaGate, [&] { K::native_qsa_gate_apply(c.attn, c.qfull, c.gated, c.T * c.NH, HD, stream); });
            emit(Out, [&] {
                quant(c.gated, c.out_q8, c.Q);
                mm(l.out, c.out_q8, a.block, c.Q, N);
            });
        }
        if (tp) hop(a.block, a.reduced, N);

        hc_read(l.hc[1], 1, true);
        auto& f = c.hc[1];
        emit(Router, [&] {
            // The fused kernel only publishes, it doesn't wait for a consumer. Give it private device outputs.
            const bool fused = K::router_chain_fused(f.mixed, l.router, c.logits, c.ids, c.weights, c.T, 2, nullptr,
                                                     NE, c.pub_x, c.pub_ids, c.pub_w, c.pub_seq, c.hc_sync + 48, stream);
            c.fused_router = fused;
            if (!fused && c.T == 1) {
                K::bf16_gemv_fp32_mmvf(f.mixed, l.router, c.logits, N, NE, stream);
                K::native_router_top10(c.logits, c.ids, c.weights, stream);
            } else if (!fused) {
                bf16(f.mixed, l.router, c.logits, N, NE);
                K::native_router_top10_multi(c.logits, c.ids, c.weights, c.T, stream);
            }
        });
        emit(Quant, [&] { quant(f.mixed, c.ffn_q8, N); });
        emit(Shared, [&] {
            K::shared_expert_multi(c.T, f.mixed, nullptr, l.shared, l.shared_gate, c.sh_gate, c.sh_up, c.sh_g,
                                   c.shared, N, c.sh_ff, stream, c.ffn_q8);
        });
        emit(Experts, [&] {
            const auto& p = l.experts;
            K::native_expert_grouped(c.expert_layout, p.ptr, p.start, p.count, p.dst, p.tok,
                                     p.groups, p.entries, c.ffn_q8, c.expert_tmp, c.parts, stream);
        });
        emit(Combine, [&] { K::native_moe_combine_multi(c.parts, c.weights, c.shared, f.block, N, TOP, c.T, stream); });
        if (tp) hop(f.block, f.reduced, N);
    }
}

using Cards = std::vector<Card*>;

struct Graphs {
    Graphs(const Cards& devices, Class filter) : cards(devices) {
        for (size_t i = 0; i < cards.size(); ++i) {
            auto& c = *cards[i];
            c.mem.select();
            CK(hipStreamBeginCapture(c.stream, hipStreamCaptureModeThreadLocal));
            capture_stack(c, filter);
            hipGraph_t graph = nullptr;
            CK(hipStreamEndCapture(c.stream, &graph));
            CK(hipGraphInstantiate(&exec[i], graph, nullptr, nullptr, 0));
            CK(hipGraphDestroy(graph));
        }
    }
    Graphs(const Graphs&) = delete;
    Graphs& operator=(const Graphs&) = delete;
    ~Graphs() {
        for (size_t i = 0; i < cards.size(); ++i) {
            cards[i]->mem.select();
            CK(hipGraphExecDestroy(exec[i]));
        }
    }
    void sync() const {
        for (const auto* c : cards) {
            c->mem.select();
            CK(hipStreamSynchronize(c->stream));
        }
    }
    void reset_residual() const {
        for (const auto* c : cards) {
            c->mem.select();
            CK(hipMemcpyAsync(c->r, c->r0, size_t(c->T) * c->D * sizeof(float), hipMemcpyDeviceToDevice, c->stream));
        }
        sync();
    }
    void replay() const {
        // Both graph launches must be queued before any stream is synchronized.
        for (size_t i = 0; i < cards.size(); ++i) {
            cards[i]->mem.select();
            CK(hipGraphLaunch(exec[i], cards[i]->stream));
        }
        sync();
    }
    const Cards& cards;
    std::array<hipGraphExec_t, 2> exec{};
};

double time_graph(const Graphs& g, int reps) {
    using Clock = std::chrono::steady_clock;
    double total = 0.0;
    const auto run = [&](bool timed) {
        g.reset_residual(); // prevent replay-to-replay drift; this copy is not a timed graph node
        const auto start = Clock::now();
        g.replay();
        if (timed) total += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    };
    for (int i = 0; i < WARM; ++i) run(false);
    for (int i = 0; i < reps; ++i) run(true);
    return total / reps;
}

void sanity_hops(const Cards& cards, unsigned expected_seq) {
    const int T = cards.front()->T;
    // Last layer's two AG results and their source slices live in separate attn/FFN scratch.
    size_t words = 0;
    double checksum = 0.0;
    for (int half = 0; half < 2; ++half) {
        std::vector<float> expected(T * N), gathered(T * N), slice(T * SHARD);
        for (const auto* c : cards) {
            c->mem.select();
            CK(hipMemcpy(slice.data(), c->hc[half].slice, slice.size() * sizeof(float), hipMemcpyDeviceToHost));
            for (int t = 0; t < T; ++t)
                std::memcpy(expected.data() + t * N + c->rank * SHARD, slice.data() + t * SHARD, SHARD * sizeof(float));
        }
        for (const auto* c : cards) {
            c->mem.select();
            CK(hipMemcpy(gathered.data(), c->hc[half].mixed, gathered.size() * sizeof(float), hipMemcpyDeviceToHost));
            if (std::memcmp(expected.data(), gathered.data(), gathered.size() * sizeof(float)) != 0)
                fail("TP2 all-gather does not match the concatenated source slices");
            words += gathered.size();
        }
        checksum += std::accumulate(expected.begin(), expected.end(), 0.0);
    }
    for (const auto* c : cards) {
        c->mem.select();
        unsigned seq = 0;
        CK(hipMemcpy(&seq, c->seq, sizeof(seq), hipMemcpyDeviceToHost));
        if (seq != expected_seq) fail("TP2 hop replay counter mismatch");
    }
    std::printf("  hop sanity: %zu AG words matched, checksum %.9g, seq %u on both cards\n", words, checksum, expected_seq);
}

void print_time(const char* label, double us, size_t nl) {
    std::printf("  %-12s %10.2f us/stack %8.2f us/layer\n", label, us, us / double(nl));
}

void run_stack(const char* name, const Cards& cards, int reps) {
    const size_t nl = cards.front()->layers.size();
    const bool tp = cards.size() == 2;
    std::printf("%s timing (%d warm-ups per graph):\n", name, WARM);
    const Graphs full(cards, All);
    const double us = time_graph(full, reps);
    print_time(name, us, nl);
    std::printf("  router: %s\n", cards.front()->fused_router ? "fused, device-only publish" : "separate GEMV and top-10");
    if (tp) {
        sanity_hops(cards, unsigned(reps + WARM) * 6u * unsigned(nl));
        const Graphs no_hops(cards, NoHops);
        const double compute = time_graph(no_hops, reps);
        print_time("tp2-no-hops", compute, nl);
        print_time("tp2 delta", us - compute, nl);
    }
    std::puts("  approximate class-only breakdown (averaged over all layers, not additive):");
    for (int i = 0; i < ClassCount; ++i) {
        const Class cls = static_cast<Class>(i);
        if ((cls == Hop && !tp) || (cls >= QsaKv && cls <= QsaGate && nl < size_t(G.qsa_interval))) continue;
        // Refill scratch so every isolated run sees valid input even where another class is the producer.
        full.reset_residual();
        full.replay();
        const Graphs isolated(cards, cls);
        print_time(class_names[i], time_graph(isolated, reps), nl);
    }
}

int positive(const char* s, int min, int max, const char* msg) {
    char* end = nullptr;
    errno = 0;
    const long n = std::strtol(s, &end, 10);
    if (errno || end == s || *end || n < min || n > max) fail(msg);
    return int(n);
}

void enable_peer(int a, int b) {
    CK(hipSetDevice(a));
    int can = 0;
    CK(hipDeviceCanAccessPeer(&can, a, b));
    if (!can) fail("TP2_DEVS must support peer access both ways");
    const auto e = hipDeviceEnablePeerAccess(b, 0);
    if (e == hipErrorPeerAccessAlreadyEnabled) {
        const auto last = hipGetLastError();
        if (last != hipErrorPeerAccessAlreadyEnabled) CK(last);
    } else CK(e);
}

void print_device(int dev) {
    hipDeviceProp_t p{};
    CK(hipGetDeviceProperties(&p, dev));
    std::printf("  dev %d: %s (%s)\n", dev, p.name, p.gcnArchName);
    if (std::strncmp(p.gcnArchName, "gfx906", 6) != 0) fail("this benchmark targets gfx906");
}

void bench_defaults() {
    // Production-shaped gfx906 kernels; let the caller supply environment overrides.
    const char* flags[] = {"STRATA_MMVF_ROWS", "STRATA_GFX906_ROUTER_T1_ROWS", "STRATA_GFX906_ROUTER_T2_EXACT",
                           "STRATA_HIP_SWIGLU_FUSED"};
    for (const auto* flag : flags) {
        if (setenv(flag, "1", 0) != 0) fail("could not set bench dispatch defaults");
        std::printf("  %s=%s\n", flag, std::getenv(flag));
    }
    K::shared_expert_set_native_bf16(true);
}

} // namespace

int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string mode = argc > 1 ? argv[1] : "both";
    if (argc > 6 || (mode != "ref" && mode != "tp2" && mode != "both"))
        fail("usage: tp2_layer_bench [ref|tp2|both] [T=1|2] [layers=12] [reps=50] [ctx=4096]");
    const int T = argc > 2 ? positive(argv[2], 1, 2, "T must be 1 or 2") : 1;
    const int nl = argc > 3 ? positive(argv[3], 1, int(G.n_layers), "layers must be 1..48") : 12;
    // Leave enough room for all class-graph replays without wrapping the LL sequence.
    const int reps = argc > 4 ? positive(argv[4], 1, 1000000, "reps must be 1..1000000") : 50;
    const int ctx = argc > 5 ? positive(argv[5], 0, MAX_CTX - T, "ctx + T must fit the 262144-cell capacity") : 4096;
    int ndev = 0;
    CK(hipGetDeviceCount(&ndev));
    if (ndev < 1) fail("no gfx906 device found");
    int d0 = 0, d1 = 1;
    if (mode != "ref") {
        const char* env = std::getenv("TP2_DEVS");
        const std::string pair = env ? env : "0,1";
        const size_t comma = pair.find(',');
        if (comma == std::string::npos) fail("TP2_DEVS must be two device ordinals: e.g. 0,1");
        d0 = positive(pair.substr(0, comma).c_str(), 0, ndev - 1, "invalid first device in TP2_DEVS");
        d1 = positive(pair.substr(comma + 1).c_str(), 0, ndev - 1, "invalid second device in TP2_DEVS");
        if (d0 == d1) fail("TP2_DEVS must name two different devices");
    }
    std::printf("%s: T=%d layers=%d (%d GDN, %d QSA) reps=%d ctx=%d QSA capacity=%d\n",
                mode.c_str(), T, nl, nl - nl / int(G.qsa_interval), nl / int(G.qsa_interval), reps, ctx, MAX_CTX);
    bench_defaults();
    const bool half_ff = K::native_expert_supported(Q4_0, Q4_0, N, FF / 2);
    if (!K::native_expert_supported(Q4_0, Q4_0, N, FF)) fail("full-width Q4_0 grouped expert is unsupported");
    std::printf("  routed FF=%d supported: %s%s\n", FF / 2, half_ff ? "yes" : "no",
                half_ff ? " (TP2 uses half-FF shards)" : " (TP2 uses 5/7 groups of whole experts per rank for T=1/2)");
    if (mode != "tp2") {
        print_device(0);
        Card ref(0, 0, false, T, nl, ctx, half_ff);
        run_stack("ref", Cards{&ref}, reps);
    }
    if (mode != "ref") {
        print_device(d0);
        print_device(d1);
        enable_peer(d0, d1);
        enable_peer(d1, d0);
        Card a(d0, 0, true, T, nl, ctx, half_ff), b(d1, 1, true, T, nl, ctx, half_ff);
        a.peer = b.recv;
        b.peer = a.recv;
        run_stack("tp2", Cards{&a, &b}, reps);
    }
    return 0;
} catch (const std::exception& e) {
    fail(e.what());
}
