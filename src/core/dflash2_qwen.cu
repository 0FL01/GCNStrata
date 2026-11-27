// Qwen stand-alone DFlash2. The two-tap conv, split attention, two-stage top-k,
// and selector lattice are adapted from src/poly/inkling_dflash2.cu.
#include "dflash2_qwen.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::core::dflash2_qwen {
namespace {

constexpr int kLayers = 5, kFF = 7680, kH = 128, kQHeads = 32, kKVHeads = 8;
constexpr int kQ = kH * kQHeads, kKV = kH * kKVHeads, kRank = 256;
constexpr int kGroup = 16, kDyn = 4 * (kWidth / kGroup), kSteps = kBlock - 1;
constexpr int kChunk = 32, kMaxParts = kWindow / kChunk;
constexpr int kContextBatch = 128;
constexpr float kEps = 1e-6f;
static_assert(kWindow % kChunk == 0 && (kWindow & (kWindow - 1)) == 0);

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void blas_check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": " + std::to_string(s));
}
int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct OnDevice {
    int old;
    explicit OnDevice(int device) {
        check(cudaGetDevice(&old), "get current device");
        if (old != device) check(cudaSetDevice(device), "set drafter device");
    }
    ~OnDevice() { (void)cudaSetDevice(old); }
};

enum Type : uint32_t { F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8, Q6_K = 14, BF16 = 30 };
struct Matrix {
    const uint8_t* data = nullptr;
    size_t stride = 0;  // bytes per row
    int in = 0, out = 0;
    uint32_t type = F32;
};

__device__ __forceinline__ float half_at(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

template <uint32_t T>
__device__ __forceinline__ float value(const uint8_t* w, int c) {
    if constexpr (T == F32) return reinterpret_cast<const float*>(w)[c];
    if constexpr (T == F16) return half_at(w + 2 * c);
    if constexpr (T == BF16) return __uint_as_float(uint32_t(reinterpret_cast<const uint16_t*>(w)[c]) << 16);
    if constexpr (T == Q8_0) {
        const auto* b = w + c / 32 * 34;
        return half_at(b) * reinterpret_cast<const int8_t*>(b)[2 + c % 32];
    }
    if constexpr (T == Q4_0) {
        const auto* b = w + c / 32 * 18;
        const int j = c % 32;
        return half_at(b) * (((b[2 + j % 16] >> (4 * (j / 16))) & 15) - 8);
    }
    if constexpr (T == Q6_K) {
        const auto* b = w + c / 256 * 210;
        const int j = c % 256, ip = j / 128, q = j % 128 / 32, lane = j % 32;
        const int lo = (b[64 * ip + 32 * (q % 2) + lane] >> (4 * (q / 2))) & 15;
        const int hi = (b[128 + 32 * ip + lane] >> (2 * q)) & 3;
        return half_at(b + 208) * reinterpret_cast<const int8_t*>(b)[192 + j / 16] * ((lo | (hi << 4)) - 32);
    }
}

// Stage FP32 inputs in LDS and reuse each decoded weight across the block's active rows.
// Subgroups of 16 (four per wave64) consume 16 contiguous values; no activation quantization.
template <uint32_t T, int Batch>
__global__ void __launch_bounds__(256) linear_kernel(Matrix w, const float* x, float* y, int rows) {
    constexpr int kTile = 512, kLanes = 16, kOutputs = 256 / kLanes;
    __shared__ float tile[Batch][kTile];
    const int tid = threadIdx.x, lane = tid % kLanes, o = blockIdx.x * kOutputs + tid / kLanes;
    const uint8_t* wr = w.data + o * w.stride;
    float sums[Batch] = {};
    for (int start = 0; start < w.in; start += kTile) {
        const int n = min(w.in - start, kTile);
        for (int j = tid; j < rows * n; j += 256) tile[j / n][j % n] = x[size_t(j / n) * w.in + start + j % n];
        __syncthreads();
        if (o < w.out) {
            for (int j = lane; j < n; j += kLanes) {
                const float d = value<T>(wr, start + j);
#pragma unroll
                for (int r = 0; r < Batch; ++r) if (r < rows) sums[r] = fmaf(d, tile[r][j], sums[r]);
            }
        }
        __syncthreads();
    }
    for (int r = 0; r < rows; ++r) {
        float s = sums[r];
        for (int d = 8; d; d >>= 1) s += __shfl_xor(s, d, kLanes);
        if (o < w.out && lane == 0) y[size_t(r) * w.out + o] = s;
    }
}

template <uint32_t T>
void small_product(Matrix w, const float* x, float* y, int n, cudaStream_t stream) {
    const int grid = ceil_div(w.out, 16);
    if (n == 1) linear_kernel<T, 1><<<grid, 256, 0, stream>>>(w, x, y, n);
    else if (n <= 4) linear_kernel<T, 4><<<grid, 256, 0, stream>>>(w, x, y, n);
    else linear_kernel<T, 8><<<grid, 256, 0, stream>>>(w, x, y, n);
}

template <uint32_t T>
__global__ void expand_kernel(Matrix w, float* out, size_t n) {
    const size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < n) out[j] = value<T>(w.data + (j / w.in) * w.stride, j % w.in);
}

__global__ void embed_kernel(Matrix emb, int32_t anchor, float* out) {
    const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (c < kWidth) out[r * kWidth + c] = value<Q4_0>(emb.data + size_t(r ? kMask : anchor) * emb.stride, c);
}

__device__ __forceinline__ float sum_wave(float x) {
    for (int d = 32; d; d >>= 1) x += __shfl_xor(x, d, 64);
    return x;
}

__global__ void rms_kernel(const float* x, const float* w, float* y) {
    __shared__ float ss[4];
    const int t = threadIdx.x, r = blockIdx.x;
    x += r * kWidth;
    y += r * kWidth;
    float sum = 0.f;
    for (int c = t; c < kWidth; c += 256) sum = fmaf(x[c], x[c], sum);
    sum = sum_wave(sum);
    if (t % 64 == 0) ss[t / 64] = sum;
    __syncthreads();
    const float scale = rsqrtf((ss[0] + ss[1] + ss[2] + ss[3]) / kWidth + kEps);
    for (int c = t; c < kWidth; c += 256) y[c] = (x[c] * scale) * w[c];
}

__global__ void norm_rope_kernel(const float* x, const float* w, const float* freq, float* y, int64_t first) {
    const int c = threadIdx.x, row = blockIdx.y;
    const size_t off = (size_t(row) * gridDim.x + blockIdx.x) * kH;
    const float u = x[off + c], v = x[off + c + 64];
    const float scale = rsqrtf(sum_wave(u * u + v * v) / kH + kEps);
    float s, cs;
    sincosf(float(first + row) * freq[c], &s, &cs);
    const float a = (u * scale) * w[c], b = (v * scale) * w[c + 64];
    y[off + c] = a * cs - b * s;
    y[off + c + 64] = b * cs + a * s;
}

__global__ void conv_kernel(const float* x, const float* dyn, const float* base, int stage, float* out) {
    const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (c >= kWidth) return;
    const int g = c / kGroup, groups = kWidth / kGroup;
    const float* d = dyn + r * kDyn + stage * 2 * groups;
    base += stage * 2 * kWidth;
    float v = (base[c] + d[g]) * x[r * kWidth + c];
    if (r) v += (base[kWidth + c] + d[groups + g]) * x[(r - 1) * kWidth + c];
    if (stage == 0) out[r * kWidth + c] = v;
    else out[r * kWidth + c] += v;
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = (gate[i] / (1.f + expf(-gate[i]))) * up[i];
}

__global__ void store_kernel(const float* k, const float* v, __half* kc, __half* vc, int64_t first) {
    const int r = blockIdx.x;
    const size_t slot = size_t((first + r) & (kWindow - 1));
    const float* src = (blockIdx.y ? v : k) + r * kKV;
    __half* dst = (blockIdx.y ? vc : kc) + slot * kKV;
    for (int c = threadIdx.x; c < kKV; c += 256) dst[c] = __float2half(src[c]);
}

// A CTA covers one KV head and up to 8*4 queries, over 32 keys. Unlike Inkling's
// per-query cutoff, the caller clips the prefix to [p+B-window,p) once for all queries.
__global__ void __launch_bounds__(256) attention_parts(const float* q, const float* k, const float* v,
                                                        const __half* kc, const __half* vc, int64_t first,
                                                        int prefix, int rows, float* partial) {
    constexpr int kQueries = kBlock * 4;
    __shared__ float qs[kQueries][kH + 1], ks[kChunk][kH + 1], vs[kChunk][kH + 1];
    __shared__ float ps[kQueries][kChunk + 1];
    const int g = blockIdx.x, chunk = blockIdx.y, tid = threadIdx.x;
    const int nq = rows * 4, count = min(kChunk, prefix + rows - chunk * kChunk);
    for (int i = tid; i < nq * kH; i += 256) {
        const int j = i / kH, c = i % kH;
        qs[j][c] = q[(j / 4 * kQHeads + g * 4 + j % 4) * kH + c];
    }
    for (int i = tid; i < count * kH; i += 256) {
        const int j = i / kH, c = i % kH, key = chunk * kChunk + j;
        if (key < prefix) {
            const size_t off = size_t((first + key) & (kWindow - 1)) * kKV + g * kH + c;
            ks[j][c] = __half2float(kc[off]);
            vs[j][c] = __half2float(vc[off]);
        } else {
            const int off = (key - prefix) * kKV + g * kH + c;
            ks[j][c] = k[off];
            vs[j][c] = v[off];
        }
    }
    __syncthreads();
    const int qi = tid / 8, s = tid % 8;
    float m = -INFINITY, score[kChunk / 8];
    if (qi < nq) {
#pragma unroll
        for (int i = 0; i < kChunk / 8; ++i) {
            const int j = s + 8 * i;
            float dot = 0.f;
            if (j < count) {
#pragma unroll 8
                for (int d = 0; d < kH; ++d) dot = fmaf(qs[qi][d], ks[j][d], dot);
            }
            score[i] = j < count ? dot * 0.08838834764831845f : -INFINITY;
            m = fmaxf(m, score[i]);
        }
        for (int off = 4; off; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, 8));
    }
    float l = 0.f;
    if (qi < nq) {
#pragma unroll
        for (int i = 0; i < kChunk / 8; ++i) {
            const float e = score[i] == -INFINITY ? 0.f : expf(score[i] - m);
            ps[qi][s + i * 8] = e;
            l += e;
        }
        for (int off = 4; off; off >>= 1) l += __shfl_xor(l, off, 8);
    }
    __syncthreads();
    if (qi >= nq) return;
    float a[16] = {};
    for (int j = 0; j < count; ++j) {
        const float e = ps[qi][j];
#pragma unroll
        for (int d = 0; d < 16; ++d) a[d] = fmaf(e, vs[j][s * 16 + d], a[d]);
    }
    float* o = partial + ((g * gridDim.y + chunk) * nq + qi) * (kH + 2);
#pragma unroll
    for (int d = 0; d < 16; ++d) o[s * 16 + d] = a[d];
    if (s == 0) { o[kH] = m; o[kH + 1] = l; }
}

__global__ void attention_merge(const float* partial, int parts, int nq, float* out) {
    __shared__ float ws[kMaxParts], denom;
    const int g = blockIdx.x, qi = blockIdx.y, d = threadIdx.x;
    const int stride = nq * (kH + 2);
    const float* p = partial + (g * parts * nq + qi) * (kH + 2);
    if (d < 64) {
        const float m = d < parts ? p[d * stride + kH] : -INFINITY;
        float big = m;
        for (int off = 32; off; off >>= 1) big = fmaxf(big, __shfl_xor(big, off, 64));
        const float w = d < parts ? expf(m - big) : 0.f;
        ws[d] = w;
        const float z = sum_wave(d < parts ? w * p[d * stride + kH + 1] : 0.f);
        if (d == 0) denom = z;
    }
    __syncthreads();
    float a = 0.f;
    for (int c = 0; c < parts; ++c) a = fmaf(ws[c], p[c * stride + d], a);
    out[(qi / 4 * kQHeads + g * 4 + qi % 4) * kH + d] = a / denom;
}

struct Candidate { float logit; int32_t id; };
constexpr int kTopParts = 512, kSpan = (kVocab + kTopParts - 1) / kTopParts;
constexpr Candidate kInvalid{-INFINITY, INT32_MAX};
__device__ __forceinline__ bool better(Candidate a, Candidate b) {
    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
}
__device__ __forceinline__ Candidate wave_best(Candidate x) {
    for (int d = 32; d; d >>= 1) {
        const Candidate a{__shfl_xor(x.logit, d, 64), __shfl_xor(x.id, d, 64)};
        if (better(a, x)) x = a;
    }
    return x;
}

__global__ void top_parts(const float* logit, Candidate* out) {
    const int lane = threadIdx.x, first = blockIdx.x * kSpan;
    logit += size_t(blockIdx.y) * kVocab;
    out += (blockIdx.y * kTopParts + blockIdx.x) * kTop;
    constexpr int slots = (kSpan + 63) / 64;
    Candidate items[slots];
#pragma unroll
    for (int j = 0; j < slots; ++j) {
        const int i = lane + j * 64;
        const float v = i < kSpan && first + i < kVocab ? logit[first + i] : -INFINITY;
        items[j] = i < kSpan && first + i < kVocab ? Candidate{isnan(v) ? -INFINITY : v, first + i} : kInvalid;
    }
    for (int i = 0; i < kTop; ++i) {
        Candidate local = kInvalid;
        int at = 0;
#pragma unroll
        for (int j = 0; j < slots; ++j) if (better(items[j], local)) { local = items[j]; at = j; }
        const auto best = wave_best(local);
        if (local.id == best.id) items[at] = kInvalid;
        if (lane == 0) out[i] = best;
    }
}

struct DeviceResult {
    int32_t tokens[kSteps];
    float prob[kSteps];
    int32_t ids[kSteps * kTop];
    float unary[kSteps * kTop];
};

__global__ void top_merge(const Candidate* part, DeviceResult* out) {
    __shared__ Candidate waves[4];
    const int t = threadIdx.x, row = blockIdx.x;
    const Candidate* l = part + (row * kTopParts + 2 * t) * kTop;
    int left = 0, right = 0;
    for (int i = 0; i < kTop; ++i) {
        const auto a = left < kTop ? l[left] : kInvalid;
        const auto b = right < kTop ? l[kTop + right] : kInvalid;
        const bool use_left = better(a, b);
        const auto mine = use_left ? a : b;
        auto best = wave_best(mine);
        if (t % 64 == 0) waves[t / 64] = best;
        __syncthreads();
        best = waves[0];
        for (int j = 1; j < 4; ++j) if (better(waves[j], best)) best = waves[j];
        __syncthreads();
        if (mine.id == best.id && best.id != INT32_MAX) (use_left ? left : right)++;
        if (t == 0) { out->ids[row * kTop + i] = best.id; out->unary[row * kTop + i] = best.logit; }
    }
}

__device__ __forceinline__ float code_at(Matrix w, int id, int c) {
    const auto* p = w.data + size_t(id) * w.stride;
    switch (w.type) {
        case Q8_0: return value<Q8_0>(p, c);
        case Q4_0: return value<Q4_0>(p, c);
        case BF16: return value<BF16>(p, c);
        case F16: return value<F16>(p, c);
        default: return value<F32>(p, c);
    }
}

__global__ void edges_kernel(const DeviceResult* raw, const float* projected, Matrix pred, Matrix succ,
                             int anchor, float* edges) {
    __shared__ float ps[kTop][kRank + 1], ss[kTop][kRank + 1];
    const int s = blockIdx.x, tid = threadIdx.x;
    const int pred_rows = s == 0 ? 1 : kTop;
    for (int i = tid; i < kTop * kRank; i += 256) {
        const int j = i / kRank, c = i % kRank;
        if (j < pred_rows) ps[j][c] = code_at(pred, s ? raw->ids[(s - 1) * kTop + j] : anchor, c)
                                     * projected[s * kRank + c];
        ss[j][c] = code_at(succ, raw->ids[s * kTop + j], c);
    }
    __syncthreads();
    const int a = tid / kTop, b = tid % kTop;
    if (a < pred_rows) {
        float dot = 0.f;
        for (int c = 0; c < kRank; ++c) dot = fmaf(ps[a][c], ss[b][c], dot);
        edges[s * kTop * kTop + tid] = raw->unary[s * kTop + b] + dot;
    }
}

__global__ void walk_kernel(DeviceResult* out, const float* edges, int steps) {
    int prev = 0;
    for (int s = 0; s < steps; ++s) {
        const float* e = edges + (s * kTop + prev) * kTop;
        int best = 0;
        for (int b = 1; b < kTop; ++b) if (e[b] > e[best]) best = b;
        out->tokens[s] = out->ids[s * kTop + best];
        out->prob[s] = 1.f;  // reference's T=0 is a one-hot proposal law
        prev = best;
    }
}

void validate_draft(const GgufFile& f) {
    const auto& meta = f.metadata();
    const auto require = [&](bool ok, const std::string& name) {
        if (!ok) throw std::runtime_error(f.path() + ": unsupported " + name);
    };
    const auto* arch = f.get("general.architecture");
    require(arch && arch->s == "dflash", "general.architecture");
    const auto scalar = [&](const char* name, double v) {
        const auto* m = f.get(name);
        require(m && m->is_num() && m->num() == v, name);
    };
    const auto array = [&](const char* name, std::initializer_list<int> vals) {
        const auto* m = f.get(name);
        require(m && m->type == MetaType::ARRAY && m->items.size() == vals.size(), name);
        size_t j = 0;
        for (int i : vals) require(m->items[j++].num() == i, name);
    };
    scalar("dflash.block_count", kLayers);
    scalar("dflash.embedding_length", kWidth);
    scalar("dflash.feed_forward_length", kFF);
    scalar("dflash.attention.head_count", kQHeads);
    scalar("dflash.attention.head_count_kv", kKVHeads);
    scalar("dflash.attention.key_length", kH);
    scalar("dflash.attention.value_length", kH);
    scalar("dflash.attention.causal", 0);
    scalar("dflash.attention.layer_norm_rms_epsilon", kEps);
    scalar("dflash.rope.freq_base", 10000.0);
    scalar("dflash.block_size", kBlock);
    scalar("dflash.attention.sliding_window", kWindow);
    scalar("dflash.conv_kernel_size", 2);
    scalar("dflash.conv_group_size", kGroup);
    scalar("dflash.selector_rank", kRank);
    scalar("dflash.selector_top_k", kTop);
    scalar("tokenizer.ggml.mask_token_id", kMask);
    array("dflash.target_layers", {6, 16, 26, 36, 46});
    array("dflash.attention.sliding_window_pattern", {1, 1, 1, 1, 1});
    array("dflash.rope.dimension_sections", {64, 0, 0, 0});
    for (const auto& [key, value] : meta)
        if (key.find("softcap") != std::string::npos) require(value.is_num() && value.num() == 0, key);
    require(f.tensors().size() == 6 + kLayers * 15, "draft tensor inventory");
}

struct Layer {
    Matrix ap, fp, q, k, v, o, gate, up, down;
    const float *an, *fn, *qn, *kn, *abase, *fbase;
    const float *bulk_k, *bulk_v;
    __half *kc, *vc;
};

}  // namespace

struct Drafter::Impl {
    explicit Impl(int d) : device(d) {}
    ~Impl() {
        int old = device;
        (void)cudaGetDevice(&old);
        (void)cudaSetDevice(device);
        if (st) (void)cudaStreamSynchronize(st);
        if (blas) (void)cublasDestroy(blas);
        for (auto e : events) if (e) (void)cudaEventDestroy(e);
        if (st) (void)cudaStreamDestroy(st);
        if (readback) (void)cudaFreeHost(readback);
        for (void* p : allocations) (void)cudaFree(p);
        (void)cudaSetDevice(old);
    }
    int device;
    int64_t lo = 0, hi = 0;
    bool empty = true;
    cudaStream_t st = nullptr;
    cublasHandle_t blas = nullptr;
    cudaEvent_t events[4]{};
    std::vector<void*> allocations;
    Matrix fc, embedding, head, selector, pred, succ;
    const float *cn, *fin, *freq, *bulk_fc;
    Layer layers[kLayers]{};
    float *x, *h, *prepared, *dyn, *q, *qr, *k, *kr, *v, *att, *projected;
    float *gate, *up, *gated, *hidden, *logits, *sel, *edges, *partial;
    Candidate* parts;
    DeviceResult *result, *readback = nullptr;

    template <class T> T* alloc(size_t n) {
        void* ptr = nullptr;
        check(cudaMalloc(&ptr, n * sizeof(T)), "allocate drafter device storage");
        try { allocations.push_back(ptr); }
        catch (...) { (void)cudaFree(ptr); throw; }
        return static_cast<T*>(ptr);
    }
    Matrix load(const GgufFile& f, const std::string& name, std::initializer_list<uint64_t> shape,
                std::initializer_list<uint32_t> types) {
        const auto* t = f.find(name);
        if (!t || t->shape != std::vector<uint64_t>(shape) || std::find(types.begin(), types.end(), t->type) == types.end())
            throw std::runtime_error(f.path() + ": missing or incompatible tensor " + name);
        const int cols = int(*shape.begin()), rows = int(t->elements() / cols);
        int elems = 0, bytes = 0;
        if (!block_geometry(t->type, elems, bytes) || uint64_t(cols) % elems)
            throw std::runtime_error(name + ": unsupported quantization geometry");
        const size_t stride = cols / elems * bytes, len = stride * rows;
        const size_t avail = f.file_size() - f.data_start();
        if (t->offset > avail || len > avail - t->offset)
            throw std::runtime_error(f.path() + ": truncated tensor " + name);
        auto* p = alloc<uint8_t>(len);
        check(cudaMemcpy(p, f.tensor_data(*t), len, cudaMemcpyHostToDevice), name.c_str());
        return {p, stride, cols, rows, t->type};
    }
    Matrix mat(const GgufFile& f, const std::string& n, int in, int out) {
        return load(f, n, {uint64_t(in), uint64_t(out)}, {F32, F16, BF16, Q4_0, Q8_0});
    }
    const float* param(const GgufFile& f, const std::string& n, std::initializer_list<uint64_t> shape) {
        const auto w = load(f, n, shape, {F32, F16, BF16});
        return expand(w);
    }
    const float* expand(Matrix w) {
        if (w.type == F32) return reinterpret_cast<const float*>(w.data);
        const int n = w.in * w.out;
        float* p = alloc<float>(n);
        const int g = ceil_div(n, 256);
        switch (w.type) {
            case F16: expand_kernel<F16><<<g, 256, 0, st>>>(w, p, n); break;
            case BF16: expand_kernel<BF16><<<g, 256, 0, st>>>(w, p, n); break;
            case Q4_0: expand_kernel<Q4_0><<<g, 256, 0, st>>>(w, p, n); break;
            case Q8_0: expand_kernel<Q8_0><<<g, 256, 0, st>>>(w, p, n); break;
            default: throw std::logic_error("unsupported DFlash2 expansion type");
        }
        check(cudaGetLastError(), "expand drafter weights");
        return p;
    }
    void init(const std::string& draft, const std::string& target) {
        GgufFile f(draft), t(target);
        validate_draft(f);
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, device), "DFlash2 device properties");
        if (props.warpSize != 64 || std::string(props.gcnArchName).rfind("gfx906", 0) != 0)
            throw std::runtime_error("DFlash2 Qwen requires gfx906 wave64");
        check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "create DFlash2 stream");
        blas_check(cublasCreate(&blas), "create DFlash2 BLAS handle");
        blas_check(cublasSetStream(blas, st), "assign DFlash2 BLAS stream");
        for (auto& e : events) check(cudaEventCreate(&e), "create DFlash2 event");
        embedding = load(t, "token_embd.weight", {kWidth, kVocab}, {Q4_0});
        head = load(t, "output.weight", {kWidth, kVocab}, {Q6_K});
        fc = mat(f, "fc.weight", kTapWidth, kWidth);
        bulk_fc = expand(fc);
        cn = param(f, "enc.output_norm.weight", {kWidth});
        fin = param(f, "output_norm.weight", {kWidth});
        selector = mat(f, "selector_hidden.weight", kWidth, kRank);
        pred = mat(f, "selector_predecessor.weight", kRank, kVocab);
        succ = mat(f, "selector_successor.weight", kRank, kVocab);
        for (int i = 0; i < kLayers; ++i) {
            auto& l = layers[i];
            const auto name = [&](const char* suffix) { return "blk." + std::to_string(i) + '.' + suffix; };
            l.an = param(f, name("attn_norm.weight"), {kWidth});
            l.fn = param(f, name("ffn_norm.weight"), {kWidth});
            l.qn = param(f, name("attn_q_norm.weight"), {kH});
            l.kn = param(f, name("attn_k_norm.weight"), {kH});
            l.abase = param(f, name("attn_conv_base"), {kWidth, 2, 2});
            l.fbase = param(f, name("ffn_conv_base"), {kWidth, 2, 2});
            l.ap = mat(f, name("attn_conv_proj.weight"), kWidth, kDyn);
            l.fp = mat(f, name("ffn_conv_proj.weight"), kWidth, kDyn);
            l.q = mat(f, name("attn_q.weight"), kWidth, kQ);
            l.k = mat(f, name("attn_k.weight"), kWidth, kKV);
            l.v = mat(f, name("attn_v.weight"), kWidth, kKV);
            l.o = mat(f, name("attn_output.weight"), kQ, kWidth);
            l.gate = mat(f, name("ffn_gate.weight"), kWidth, kFF);
            l.up = mat(f, name("ffn_up.weight"), kWidth, kFF);
            l.down = mat(f, name("ffn_down.weight"), kFF, kWidth);
            l.bulk_k = expand(l.k);
            l.bulk_v = expand(l.v);
            l.kc = alloc<__half>(kWindow * kKV);
            l.vc = alloc<__half>(kWindow * kKV);
        }
        float f_cpu[64];
        for (int j = 0; j < 64; ++j) f_cpu[j] = 1.f / std::pow(10000.f, float(j) / 64.f);
        auto* f_gpu = alloc<float>(64);
        check(cudaMemcpy(f_gpu, f_cpu, sizeof(f_cpu), cudaMemcpyHostToDevice), "upload RoPE frequencies");
        freq = f_gpu;
        x = alloc<float>(kWidth * kContextBatch);
        h = alloc<float>(kWidth * kContextBatch);
        prepared = alloc<float>(kWidth * kBlock);
        dyn = alloc<float>(kDyn * kBlock);
        q = alloc<float>(kQ * kBlock);
        qr = alloc<float>(kQ * kBlock);
        k = alloc<float>(kKV * kContextBatch);
        kr = alloc<float>(kKV * kContextBatch);
        v = alloc<float>(kKV * kContextBatch);
        att = alloc<float>(kQ * kBlock);
        projected = alloc<float>(kWidth * kBlock);
        gate = alloc<float>(kFF * kBlock);
        up = alloc<float>(kFF * kBlock);
        gated = alloc<float>(kFF * kBlock);
        hidden = alloc<float>(kWidth * kBlock);
        logits = alloc<float>(kVocab * kSteps);
        sel = alloc<float>(kRank * kSteps);
        edges = alloc<float>(kTop * kTop * kSteps);
        partial = alloc<float>(kKVHeads * kMaxParts * (kBlock * 4) * (kH + 2));
        parts = alloc<Candidate>(kTopParts * kTop * kSteps);
        result = alloc<DeviceResult>(1);
        check(cudaMallocHost(reinterpret_cast<void**>(&readback), sizeof(DeviceResult)), "allocate DFlash2 readback");
        check(cudaStreamSynchronize(st), "initialize DFlash2 weights");
    }

    void linear(Matrix w, const float* in, float* out, int rows) {
        switch (w.type) {
            case Q8_0: small_product<Q8_0>(w, in, out, rows, st); break;
            case Q4_0: small_product<Q4_0>(w, in, out, rows, st); break;
            case Q6_K: small_product<Q6_K>(w, in, out, rows, st); break;
            case F32: small_product<F32>(w, in, out, rows, st); break;
            case F16: small_product<F16>(w, in, out, rows, st); break;
            case BF16: small_product<BF16>(w, in, out, rows, st); break;
            default: throw std::logic_error("unsupported DFlash2 matrix type");
        }
    }
    void context_product(Matrix w, const float* dense, const float* in, float* out, int n) {
        if (n <= kBlock) { linear(w, in, out, n); return; }
        const float alpha = 1.f, beta = 0.f;
        blas_check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, w.out, n, w.in, &alpha,
                               dense, w.in, in, w.in, &beta, out, w.out), "DFlash2 context GEMM");
    }
    void append(const float* taps, int64_t p, int n) {
        context_product(fc, bulk_fc, taps, x, n);
        rms_kernel<<<n, 256, 0, st>>>(x, cn, h);
        for (auto& l : layers) {
            context_product(l.k, l.bulk_k, h, k, n);
            context_product(l.v, l.bulk_v, h, v, n);
            norm_rope_kernel<<<dim3(kKVHeads, n), 64, 0, st>>>(k, l.kn, freq, kr, p);
            store_kernel<<<dim3(n, 2), 256, 0, st>>>(kr, v, l.kc, l.vc, p);
        }
        check(cudaGetLastError(), "append DFlash2 context");
    }
    void block_layers(int32_t anchor, int64_t p, int n) {
        embed_kernel<<<dim3(kWidth / 256, n), 256, 0, st>>>(embedding, anchor, x);
        const int64_t first = empty ? p : std::max(lo, p - (kWindow - n));
        const int visible = int(p - first), np = ceil_div(visible + n, kChunk);
        const dim3 conv_grid(kWidth / 256, n);
        for (const auto& l : layers) {
            rms_kernel<<<n, 256, 0, st>>>(x, l.an, h);
            linear(l.ap, h, dyn, n);
            conv_kernel<<<conv_grid, 256, 0, st>>>(h, dyn, l.abase, 0, prepared);
            linear(l.q, prepared, q, n);
            linear(l.k, prepared, k, n);
            linear(l.v, prepared, v, n);
            norm_rope_kernel<<<dim3(kQHeads, n), 64, 0, st>>>(q, l.qn, freq, qr, p);
            norm_rope_kernel<<<dim3(kKVHeads, n), 64, 0, st>>>(k, l.kn, freq, kr, p);
            attention_parts<<<dim3(kKVHeads, np), 256, 0, st>>>(qr, kr, v, l.kc, l.vc, first, visible, n, partial);
            attention_merge<<<dim3(kKVHeads, n * 4), kH, 0, st>>>(partial, np, n * 4, att);
            linear(l.o, att, projected, n);
            conv_kernel<<<conv_grid, 256, 0, st>>>(projected, dyn, l.abase, 1, x);
            rms_kernel<<<n, 256, 0, st>>>(x, l.fn, h);
            linear(l.fp, h, dyn, n);
            conv_kernel<<<conv_grid, 256, 0, st>>>(h, dyn, l.fbase, 0, prepared);
            linear(l.gate, prepared, gate, n);
            linear(l.up, prepared, up, n);
            swiglu_kernel<<<ceil_div(n * kFF, 256), 256, 0, st>>>(gate, up, gated, n * kFF);
            linear(l.down, gated, projected, n);
            conv_kernel<<<conv_grid, 256, 0, st>>>(projected, dyn, l.fbase, 1, x);
        }
        rms_kernel<<<n, 256, 0, st>>>(x, fin, hidden);
    }
};

Drafter::Drafter(const std::string& draft, const std::string& target, int device) : impl_(std::make_unique<Impl>(device)) {
    const OnDevice on(device);
    impl_->init(draft, target);
}
Drafter::~Drafter() = default;
cudaStream_t Drafter::stream() const { return impl_->st; }
void Drafter::reset() { impl_->lo = impl_->hi = 0; impl_->empty = true; }

void Drafter::context(const float* taps, int64_t p, int rows) {
    auto& d = *impl_;
    if (!rows) return;
    if (!taps || rows < 0 || p < 0 || p > INT64_MAX - rows || (!d.empty && p != d.hi))
        throw std::invalid_argument("DFlash2 context: expected a contiguous committed append");
    const OnDevice on(d.device);
    // Skip obsolete rows; each prefix row's context encoding is independent.
    if (rows > kWindow) {
        const int skip = rows - kWindow;
        taps += size_t(skip) * kTapWidth;
        p += skip;
        rows = kWindow;
    }
    for (int i = 0; i < rows; i += kContextBatch)
        d.append(taps + size_t(i) * kTapWidth, p + i, std::min(rows - i, kContextBatch));
    d.hi = p + rows;
    d.lo = std::max(d.empty ? p : d.lo, d.hi - kWindow);
    d.empty = false;
}

Result Drafter::ask(int32_t anchor, int64_t p, int n, Timing* timing, float* host_hidden) {
    auto& d = *impl_;
    if (anchor < 0 || anchor >= kVocab || n < 2 || n > kBlock || p < 0 || p > INT64_MAX - n || (!d.empty && p != d.hi))
        throw std::invalid_argument("DFlash2 ask: invalid anchor, width or position");
    const OnDevice on(d.device);
    const auto mark = [&](int i) { if (timing) check(cudaEventRecord(d.events[i], d.st), "record ask timing"); };
    const int steps = n - 1;
    mark(0);
    d.block_layers(anchor, p, n);
    mark(1);
    d.linear(d.head, d.hidden + kWidth, d.logits, steps);
    top_parts<<<dim3(kTopParts, steps), 64, 0, d.st>>>(d.logits, d.parts);
    top_merge<<<steps, 256, 0, d.st>>>(d.parts, d.result);
    mark(2);
    d.linear(d.selector, d.hidden + kWidth, d.sel, steps);
    edges_kernel<<<steps, 256, 0, d.st>>>(d.result, d.sel, d.pred, d.succ, anchor, d.edges);
    walk_kernel<<<1, 1, 0, d.st>>>(d.result, d.edges, steps);
    mark(3);
    check(cudaGetLastError(), "launch DFlash2 ask");
    check(cudaMemcpyAsync(d.readback, d.result, sizeof(DeviceResult), cudaMemcpyDeviceToHost, d.st), "read DFlash2 candidates");
    if (host_hidden)
        check(cudaMemcpyAsync(host_hidden, d.hidden, n * kWidth * sizeof(float), cudaMemcpyDeviceToHost, d.st), "read DFlash2 hidden");
    check(cudaStreamSynchronize(d.st), "complete DFlash2 ask");
    if (timing) {
        check(cudaEventElapsedTime(&timing->layers_ms, d.events[0], d.events[1]), "time ask layers");
        check(cudaEventElapsedTime(&timing->head_ms, d.events[1], d.events[2]), "time ask head");
        check(cudaEventElapsedTime(&timing->selector_ms, d.events[2], d.events[3]), "time ask selector");
        check(cudaEventElapsedTime(&timing->total_ms, d.events[0], d.events[3]), "time ask total");
    }
    Result r;
    r.steps = steps;
    std::copy_n(d.readback->tokens, steps, r.tokens.begin());
    std::copy_n(d.readback->prob, steps, r.prob.begin());
    std::copy_n(d.readback->ids, steps * kTop, r.top_ids.begin());
    std::copy_n(d.readback->unary, steps * kTop, r.unary.begin());
    return r;
}

}  // namespace strata::core::dflash2_qwen
