// gfx906: each TP2 owner runs two local GQA groups and lends its other two to a spare GPU.
#include "inkling_kv_helpers.hpp"

#if defined(STRATA_HIP_GFX906)
#include "strata/kernels/gfx_arch.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strata::poly::inkling::kv_helpers {
namespace {

constexpr int kPreseedEnd = 32768, kOffloadPos = 65536, kPromptRows = 32, kSeedRows = 2048;
constexpr int kThreads = 128;
using Word = uint32_t;   // transport the bit patterns of FP32, Q8 codes and FP16 scales
constexpr int kHeads = kHalfKvHeads * kQueriesPerKv;
constexpr int kQueryWords = kHeads * kHeadWidth;
constexpr int kCodeWords = kHalfKvHeads * kHeadWidth / sizeof(Word);
constexpr int kScaleWords = kHalfKvHeads * (kHeadWidth / kQ8BlockValues) * sizeof(uint16_t) / sizeof(Word);
constexpr int kQ81Words = kQueryWords / kQ8BlockValues * kQ81BlockBytes / sizeof(Word);
constexpr int kAnswerWords = kQueryWords + kQ81Words;
constexpr int kTableWords = kTableRows * kRelativeWidth;

enum Field { Q, Rel, Gate, Sink, K, V, KScale, VScale, Fields };
__host__ __device__ constexpr int width(int field) {
    switch (field) {
        case Q: case Gate: return kQueryWords;
        case Rel: return kHeads * kRelativeWidth;
        case Sink: return kHeads;
        case K: case V: return kCodeWords;
        default: return kScaleWords;
    }
}
// A packet has one plane per field, with compact consecutive rows within each plane.
__host__ __device__ constexpr int offset(int field) {
    int words = 0;
    for (int i = 0; i < field; ++i) words += width(i);
    return words;
}
constexpr int kKvWords = offset(Fields) - offset(K);
constexpr int kPacketWords = offset(Fields) * kMaxRows;
constexpr int blocks(int words) { return (words + kThreads - 1) / kThreads; }
static_assert(kOwnerKvHeads == 2 * kHalfKvHeads);

struct OwnerPlanes { const Word* field[Fields]; };
struct CompactKv {
    Word* field[Fields - K];
    __host__ __device__ Word*& at(int f) { return field[f - K]; }
    Word* at(int f) const { return field[f - K]; }
};

__global__ void gather_upper(OwnerPlanes src, Word* packet, int rows, int first_field, int first_kv_row) {
    const int f = blockIdx.y + first_field, w = width(f), i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * w) return;
    const size_t row = i / w + (f >= K ? first_kv_row : 0);
    packet[(offset(f) - offset(first_field)) * rows + i] = src.field[f][(row * 2 + 1) * w + i % w];
}

__global__ void append_upper(CompactKv dst, const Word* packet, int rows, int pos) {
    const int f = blockIdx.y + K, w = width(f), i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * w) dst.at(f)[size_t(pos) * w + i] = packet[offset(f) * rows + i];
}

__global__ void unpack_answer(const Word* packet, Word* out, Word* q8, int rows) {
    const bool quantized = blockIdx.y != 0;
    const int w = quantized ? kQ81Words : kQueryWords, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * w)
        (quantized ? q8 : out)[(i / w * 2 + 1) * w + i % w] = packet[(quantized ? kQueryWords * rows : 0) + i];
}

void check(hipError_t rc, const char* operation) {
    if (rc != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(rc));
}

struct RestoreDevice {
    int previous;
    explicit RestoreDevice(int device) : previous(device) {}
    ~RestoreDevice() { (void) hipSetDevice(previous); }
};

// One fixed workspace and one ordered auxiliary stream on each GPU. Allocation is separate from construction so
// partially initialized HIP resources are also released if one of the two links cannot be set up.
struct Resources {
    explicit Resources(int device) : device(device) {}
    const int device;
    hipStream_t stream = nullptr;
    hipEvent_t ready = nullptr, sent = nullptr;
    Word* memory = nullptr;

    void init(size_t words, bool owner) {
        check(hipSetDevice(device), "select KV helper workspace device");
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create KV helper stream");
        check(hipEventCreateWithFlags(&ready, hipEventDisableTiming), "create KV helper ready event");
        if (owner) check(hipEventCreateWithFlags(&sent, hipEventDisableTiming), "create KV helper transfer event");
        check(hipMalloc(reinterpret_cast<void**>(&memory), words * sizeof(Word)), "allocate KV helper workspace");
    }

    void wait(hipStream_t st) const noexcept {
        if (!st) return;
        int previous = 0;
        if (hipGetDevice(&previous) != hipSuccess) return;
        RestoreDevice restore(previous);
        if (hipSetDevice(device) == hipSuccess) (void) hipStreamSynchronize(st);
    }

    ~Resources() {
        if (!stream && !memory) return;
        int previous = 0;
        if (hipGetDevice(&previous) != hipSuccess) return;
        RestoreDevice restore(previous);
        if (hipSetDevice(device) != hipSuccess) return;
        if (stream) (void) hipStreamDestroy(stream);
        if (ready) (void) hipEventDestroy(ready);
        if (sent) (void) hipEventDestroy(sent);
        if (memory) (void) hipFree(memory);
    }
};

void enable_link(int from, int to) {
    int accessible = 0;
    check(hipDeviceCanAccessPeer(&accessible, from, to), "query KV helper peer access");
    if (!accessible)
        throw std::runtime_error("missing P2P link " + std::to_string(from) + " -> " + std::to_string(to));
    check(hipSetDevice(from), "select KV helper P2P device");
    const auto rc = hipDeviceEnablePeerAccess(to, 0);
    // Peer access is process-wide and may be shared with another engine; do not disable it on teardown.
    if (rc == hipErrorPeerAccessAlreadyEnabled) (void) hipGetLastError();
    else check(rc, "enable KV helper peer access");
}

}  // namespace

struct Mirror {
    Mirror(int rank, int capacity, hipStream_t model_stream)
        : owner(rank), helper(rank + 2), capacity(capacity), model_stream(model_stream) {}

    Resources owner, helper;
    const int capacity;
    const hipStream_t model_stream;
    bool used = false, long_answer = false;
    std::array<int, kFullVisits> valid{};
    std::array<bool, kFullVisits> table_ready{};
    Word *send = nullptr, *seed = nullptr, *receive = nullptr, *packet = nullptr, *answer = nullptr, *kv = nullptr;
    float *tables = nullptr, *partial = nullptr, *scores = nullptr;

    void init() {
        constexpr size_t seed_words = kKvWords * kSeedRows, answer_words = kAnswerWords * kMaxRows;
        const size_t kv_words = size_t(capacity) * kKvWords * kFullVisits;
        const size_t score_words = size_t(capacity) * kHeads * kMaxRows;
        const size_t partial_words = half_decode_floats();
        constexpr size_t table_words = kTableWords * kFullVisits;
        static_assert(sizeof(Word) == sizeof(float));
        owner.init(kPacketWords + seed_words + answer_words, true);
        send = owner.memory;
        seed = send + kPacketWords;
        receive = seed + seed_words;
        helper.init(kPacketWords + answer_words + kv_words + table_words + partial_words + score_words, false);
        packet = helper.memory;
        answer = packet + kPacketWords;
        kv = answer + answer_words;
        tables = reinterpret_cast<float*>(kv + kv_words);
        partial = tables + table_words;
        scores = partial + partial_words;
    }

    CompactKv planes(int slot) const {
        Word* base = kv + size_t(slot) * capacity * kKvWords;
        CompactKv out{};
        for (int f = K; f < Fields; ++f) out.at(f) = base + size_t(capacity) * (offset(f) - offset(K));
        return out;
    }

    ~Mirror() {
        // All three streams may still reference one of the two allocations. Drain before destroying either bundle.
        if (!used) return;
        owner.wait(owner.stream);
        helper.wait(helper.stream);
        owner.wait(model_stream);
    }
};

void FreeMirror::operator()(Mirror* mirror) const noexcept { delete mirror; }

std::array<MirrorPtr, 2> create_pair(int capacity, std::array<cudaStream_t, 2> owner_streams) {
    if (capacity <= kOffloadPos) throw std::runtime_error("max context must be greater than 65536");
    int previous = 0, devices = 0;
    check(hipGetDevice(&previous), "get KV helper setup device");
    RestoreDevice restore(previous);
    check(hipGetDeviceCount(&devices), "count KV helper devices");
    if (devices != 4) throw std::runtime_error("requires exactly four visible gfx906 GPUs");
    for (int device = 0; device < devices; ++device) {
        hipDeviceProp_t props{};
        check(hipGetDeviceProperties(&props, device), "query KV helper device properties");
        if (props.warpSize != 64 || !kernels::gfx_arch_is(props.gcnArchName, "gfx906"))
            throw std::runtime_error("requires gfx906 wave64 on all four devices");
    }
    for (int rank : {0, 1}) {
        enable_link(rank, rank + 2);
        enable_link(rank + 2, rank);
    }
    std::array<MirrorPtr, 2> mirrors;
    for (int rank : {0, 1}) {
        mirrors[rank].reset(new Mirror(rank, capacity, owner_streams[rank]));
        mirrors[rank]->init();
    }
    return mirrors;
}

void invalidate(Mirror& mirror, int64_t first) {
    for (int& end : mirror.valid) end = static_cast<int>(std::min<int64_t>(end, first));
}

void set_budget(Mirror& mirror, int64_t prompt_rows, int64_t max_new) {
    // A 64K/256 generation amortized the peer seed; a sampled /16 request did not.
    mirror.long_answer = prompt_rows >= kOffloadPos && max_new >= 256;
}

Visit try_visit(Mirror& m, const Attend& a, int slot, float* partial, float* scores, size_t score_floats, void* q8) {
    if (a.kv_heads != kOwnerKvHeads || a.window || a.ring || !a.k.scales || !a.v.scales ||
        slot < 0 || slot >= kFullVisits || a.n <= 0 || a.pos < 0 || a.pos > m.capacity - a.n)
        return Visit::Local;
    const int pos = static_cast<int>(a.pos), rows = a.n, end = pos + rows;
    int& valid = m.valid[slot];
    // keep() may accept only a prefix, or replay the tail after a snap. New owner stores supersede any provisional
    // suffix, even on a call that will run locally. Catch-up always starts from the remaining contiguous prefix.
    valid = std::min(valid, pos);
    // A short request still trims provisional rows above. Do not seed its prefix, or recopy a long missing
    // prefix after restore. A continuous long read reaches the preseed crossing with at most 32K rows missing.
    if (!m.long_answer || pos - valid > kPreseedEnd) return Visit::Local;
    const bool preseed = rows >= kPromptRows && end >= kPreseedEnd;
    if (!preseed && (pos < kOffloadPos || rows > kMaxRows || !partial || !scores || !q8 ||
                     size_t(end) > score_floats / (rows * kHeads * 2)))
        return Visit::Local;

    int current = 0;
    check(hipGetDevice(&current), "get KV helper visit device");
    if (current != m.owner.device) throw std::logic_error("Inkling KV helper called on the wrong TP2 owner");
    RestoreDevice restore(current);
    m.used = true;
    try {
        const OwnerPlanes src{{reinterpret_cast<const Word*>(a.q), reinterpret_cast<const Word*>(a.rel),
                               reinterpret_cast<const Word*>(a.gate), reinterpret_cast<const Word*>(a.sink),
                               reinterpret_cast<const Word*>(a.k.data), reinterpret_cast<const Word*>(a.v.data),
                               reinterpret_cast<const Word*>(a.k.scales), reinterpret_cast<const Word*>(a.v.scales)}};
        const CompactKv kv = m.planes(slot);
        const auto comm = m.owner.stream, model = m.model_stream, work = m.helper.stream;
        // This edge also protects the helper's packet, mirrors and owner receive scratch from the previous visit:
        // the model stream has already waited for its reply and unpacked it (or joined its bulk seed).
        check(hipEventRecord(m.owner.ready, model), "record KV helper inputs ready");
        check(hipStreamWaitEvent(comm, m.owner.ready, 0), "wait for KV helper inputs");
        const int seed_to = preseed ? end : pos;
        for (int at = valid; at < seed_to;) {
            const int n = std::min(kSeedRows, seed_to - at);
            gather_upper<<<dim3(blocks(n * kCodeWords), Fields - K), kThreads, 0, comm>>>(src, m.seed, n, K, at);
            check(hipGetLastError(), "gather KV helper seed");
            for (int f = K; f < Fields; ++f) {
                const int w = width(f);
                check(hipMemcpyPeerAsync(kv.at(f) + size_t(at) * w, m.helper.device,
                                         m.seed + (offset(f) - offset(K)) * n, m.owner.device,
                                         size_t(n) * w * sizeof(Word), comm), "send KV helper seed");
            }
            at += n;
        }
        if (preseed) {
            check(hipEventRecord(m.owner.sent, comm), "record KV helper seed sent");
            valid = seed_to;
            return Visit::Seeding;
        }

        gather_upper<<<dim3(blocks(rows * kQueryWords), Fields), kThreads, 0, comm>>>(src, m.send, rows, Q, pos);
        check(hipGetLastError(), "gather KV helper packet");
        check(hipMemcpyPeerAsync(m.packet, m.helper.device, m.send, m.owner.device,
                                 size_t(rows) * offset(Fields) * sizeof(Word), comm), "send KV helper packet");
        float* table = m.tables + size_t(slot) * kTableWords;
        if (!m.table_ready[slot])
            check(hipMemcpyPeerAsync(table, m.helper.device, a.table, m.owner.device,
                                     kTableWords * sizeof(float), comm), "send KV helper bias table");
        check(hipEventRecord(m.owner.sent, comm), "record KV helper packet sent");
        attend_decode_half(a, partial, scores, q8, model);

        check(hipSetDevice(m.helper.device), "select KV helper device");
        check(hipStreamWaitEvent(work, m.owner.sent, 0), "wait for KV helper packet");
        append_upper<<<dim3(blocks(rows * kCodeWords), Fields - K), kThreads, 0, work>>>(kv, m.packet, rows, pos);
        check(hipGetLastError(), "append KV helper rows");
        Attend b = a;
        b.q = reinterpret_cast<const float*>(m.packet);
        b.rel = reinterpret_cast<const float*>(m.packet + offset(Rel) * rows);
        b.gate = reinterpret_cast<const float*>(m.packet + offset(Gate) * rows);
        b.sink = reinterpret_cast<const float*>(m.packet + offset(Sink) * rows);
        b.table = table;
        b.k = {reinterpret_cast<const uint8_t*>(kv.at(K)), reinterpret_cast<const uint16_t*>(kv.at(KScale))};
        b.v = {reinterpret_cast<const uint8_t*>(kv.at(V)), reinterpret_cast<const uint16_t*>(kv.at(VScale))};
        b.kv_heads = kHalfKvHeads;
        b.out = reinterpret_cast<float*>(m.answer);
        attend_decode_half(b, m.partial, m.scores, m.answer + rows * kQueryWords, work);
        check(hipMemcpyPeerAsync(m.receive, m.owner.device, m.answer, m.helper.device,
                                 size_t(rows) * kAnswerWords * sizeof(Word), work), "return KV helper answer");
        check(hipEventRecord(m.helper.ready, work), "record KV helper answer ready");
        check(hipSetDevice(m.owner.device), "return to KV helper owner");
        check(hipStreamWaitEvent(model, m.helper.ready, 0), "wait for KV helper answer");
        unpack_answer<<<dim3(blocks(rows * kQueryWords), 2), kThreads, 0, model>>>(
            m.receive, reinterpret_cast<Word*>(a.out), static_cast<Word*>(q8), rows);
        check(hipGetLastError(), "unpack KV helper answer");
        m.table_ready[slot] = true;
        valid = end;
        return Visit::Offloaded;
    } catch (...) {
        // Never silently fall back after launching half an attention pass. Drain any in-flight transfers before
        // the owner's buffers can be reused and let the native TP2 error handling report the failed call.
        m.owner.wait(m.owner.stream);
        m.helper.wait(m.helper.stream);
        m.valid.fill(0);
        m.table_ready.fill(false);
        throw;
    }
}

void join_seed(Mirror& m) {
    const auto rc = hipStreamWaitEvent(m.model_stream, m.owner.sent, 0);
    if (rc != hipSuccess) {
        m.owner.wait(m.owner.stream);
        m.valid.fill(0);
        check(rc, "join KV helper seed");
    }
}

}  // namespace strata::poly::inkling::kv_helpers
#endif
