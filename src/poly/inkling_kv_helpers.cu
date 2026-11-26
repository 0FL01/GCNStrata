// gfx906: each TP2 owner runs two local GQA groups and lends its other two to a spare GPU.
#include "inkling_kv_helpers.hpp"

#if defined(STRATA_HIP_GFX906)
#include "inkling_attend_lean.hpp"
#include "strata/kernels/gfx_arch.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strata::poly::inkling::kv_helpers {
namespace {

constexpr int kPreseedEnd = 32768, kOffloadPos = 65536, kPromptRows = 32, kSeedRows = 2048;
int kv_helper_min_pos() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_INKLING_KV_HELPERS_MIN_POS");
        if (!e || !*e) return kOffloadPos;
        char* end = nullptr;
        long x = std::strtol(e, &end, 10);
        if (end == e || *end != '\0' || x < 4096 || x > 262144) return kOffloadPos;
        return int(x);
    }();
    return v;
}
int kv_helper_min_new() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_INKLING_KV_HELPERS_MIN_NEW");
        if (!e || !*e) return 256;
        char* end = nullptr;
        long x = std::strtol(e, &end, 10);
        if (end == e || *end != '\0' || x < 1 || x > 65536) return 256;
        return int(x);
    }();
    return v;
}
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

__global__ void unpack_answer_out(const Word* packet, Word* out, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * kQueryWords)
        out[(i / kQueryWords * 2 + 1) * kQueryWords + i % kQueryWords] = packet[i];
}

__global__ void gather_upper_kv_direct(OwnerPlanes src, CompactKv dst, int rows, int first_row) {
    const int f = blockIdx.y + K, w = width(f), i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * w) return;
    const size_t row = i / w + first_row;
    dst.at(f)[row * w + i % w] = src.field[f][(row * 2 + 1) * w + i % w];
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
    hipEvent_t ready = nullptr, sent = nullptr, bulk_input = nullptr;
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
        if (bulk_input) (void) hipEventDestroy(bulk_input);
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
    Mirror(int rank, int capacity, hipStream_t model_stream, int decode_max_rows)
        : owner(rank), helper(rank + 2), capacity(capacity), decode_max_rows(decode_max_rows), model_stream(model_stream) {}

    Resources owner, helper;
    std::unique_ptr<Resources> bulk;
    bool balance_flash = false;
    bool mirror_flash = false;
    Word *mirror_owner_mem = nullptr, *mirror_helper_mem = nullptr;
    Word *mirror_send = nullptr, *mirror_recv = nullptr, *mirror_packet = nullptr, *mirror_answer = nullptr;
    const int capacity, decode_max_rows;
    const hipStream_t model_stream;
    bool used = false, long_answer = false;
    std::array<int, kFullVisits> valid{};
    std::array<bool, kFullVisits> table_ready{};
    Word *send = nullptr, *seed = nullptr, *receive = nullptr, *packet = nullptr, *answer = nullptr, *kv = nullptr;
    float *tables = nullptr, *partial = nullptr, *scores = nullptr;

    void init() {
        constexpr size_t seed_words = kKvWords * kSeedRows;
        const size_t packet_words = size_t(offset(Fields)) * decode_max_rows;
        const size_t answer_words = size_t(kAnswerWords) * decode_max_rows;
        const size_t kv_words = size_t(capacity) * kKvWords * kFullVisits;
        const size_t score_words = size_t(capacity) * kHeads * decode_max_rows;
        const size_t partial_words = half_decode_floats(decode_max_rows);
        constexpr size_t table_words = kTableWords * kFullVisits;
        static_assert(sizeof(Word) == sizeof(float));
        owner.init(packet_words + seed_words + answer_words, true);
        send = owner.memory;
        seed = send + packet_words;
        receive = seed + seed_words;
        helper.init(packet_words + answer_words + kv_words + table_words + partial_words + score_words, false);
        packet = helper.memory;
        answer = packet + packet_words;
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
        if (used) {
            if (bulk) bulk->wait(bulk->stream);
            owner.wait(owner.stream);
            helper.wait(helper.stream);
            owner.wait(model_stream);
        }
        int previous = 0;
        if (hipGetDevice(&previous) == hipSuccess) {
            RestoreDevice restore(previous);
            if (mirror_owner_mem && hipSetDevice(owner.device) == hipSuccess) (void) hipFree(mirror_owner_mem);
            if (mirror_helper_mem && hipSetDevice(helper.device) == hipSuccess) (void) hipFree(mirror_helper_mem);
        }
    }
};

void FreeMirror::operator()(Mirror* mirror) const noexcept { delete mirror; }

std::array<MirrorPtr, 2> create_pair(int capacity, std::array<cudaStream_t, 2> owner_streams,
                                    bool wide_decode) {
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
        mirrors[rank].reset(new Mirror(rank, capacity, owner_streams[rank],
                                      wide_decode ? kMaxDecodeRows : kDefaultDecodeRows));
        mirrors[rank]->init();
    }
    return mirrors;
}

void invalidate(Mirror& mirror, int64_t first) {
    for (int& end : mirror.valid) end = static_cast<int>(std::min<int64_t>(end, first));
}

namespace {
constexpr int kBulkHalf = 1024, kBulkQ = kOwnerKvHeads * kQueriesPerKv * kHeadWidth;
constexpr int kBulkHeads = kOwnerKvHeads * kQueriesPerKv, kBulkRel = kBulkHeads * kRelativeWidth;
constexpr int kBulkCode = kOwnerKvHeads * kHeadWidth, kBulkScale = kBulkCode / kQ8BlockValues;
}

void enable_flash_rows(std::array<MirrorPtr, 2>& mirrors) {
    int device;
    check(hipGetDevice(&device), "get bulk Flash setup device");
    RestoreDevice restore(device);
    std::array<std::unique_ptr<Resources>, 2> scratch;
    const char* opt = std::getenv("STRATA_INKLING_FLASH_BALANCE");
    bool balance = opt && !std::strcmp(opt, "1");
    for (int r : {0, 1}) {
        if (balance) {
            for (int dev : {mirrors[r]->owner.device, mirrors[r]->helper.device}) {
                hipDeviceProp_t prop{};
                check(hipGetDeviceProperties(&prop, dev), "query bulk Flash CU count");
                balance &= prop.multiProcessorCount == 60;
            }
        }
        scratch[r] = std::make_unique<Resources>(mirrors[r]->helper.device);
        const size_t kv_bytes = size_t(mirrors[r]->capacity) * 2 * (kBulkCode + kBulkScale * sizeof(uint16_t));
        scratch[r]->init(kv_bytes / sizeof(Word) + kTableWords + kBulkHalf * (3 * kBulkQ + kBulkRel + kBulkHeads), false);
        check(hipSetDevice(mirrors[r]->owner.device), "select bulk Flash input device");
        check(hipEventCreateWithFlags(&mirrors[r]->owner.bulk_input, hipEventDisableTiming), "create bulk Flash input event");
    }
    for (int r : {0, 1}) {
        mirrors[r]->bulk = std::move(scratch[r]);
        mirrors[r]->balance_flash = balance;
    }
    if (balance) std::fputs("strata-poly: STRATA_INKLING_FLASH_BALANCE=1: 60 bulk query tiles per helper\n", stderr);
}

namespace {
constexpr int kMirrorMaxRows = 2048;
constexpr size_t kMirrorPacketWords = size_t(kMirrorMaxRows) * offset(Fields);
constexpr size_t kMirrorAnswerWords = size_t(kMirrorMaxRows) * kQueryWords;
constexpr size_t kMirrorTotalWords = kMirrorPacketWords + kMirrorAnswerWords;
}  // namespace

void enable_flash_mirror(std::array<MirrorPtr, 2>& mirrors) {
    int device = 0;
    check(hipGetDevice(&device), "get mirror Flash setup device");
    RestoreDevice restore(device);
    for (int r : {0, 1}) {
        Mirror& m = *mirrors[r];
        check(hipSetDevice(m.owner.device), "select mirror Flash owner device");
        check(hipMalloc(reinterpret_cast<void**>(&m.mirror_owner_mem), kMirrorTotalWords * sizeof(Word)),
              "allocate mirror Flash owner buffer");
        m.mirror_send = m.mirror_owner_mem;
        m.mirror_recv = m.mirror_owner_mem + kMirrorPacketWords;
        if (m.bulk && m.bulk->memory) {
            m.mirror_packet = m.bulk->memory;
            m.mirror_answer = m.bulk->memory + kMirrorPacketWords;
        } else {
            check(hipSetDevice(m.helper.device), "select mirror Flash helper device");
            check(hipMalloc(reinterpret_cast<void**>(&m.mirror_helper_mem), kMirrorTotalWords * sizeof(Word)),
                  "allocate mirror Flash helper buffer");
            m.mirror_packet = m.mirror_helper_mem;
            m.mirror_answer = m.mirror_helper_mem + kMirrorPacketWords;
        }
        m.mirror_flash = true;
    }
    std::fputs("strata-poly: STRATA_INKLING_FLASH_MIRROR=1: 4-GPU persistent-mirror head-split Flash (T=64..2048, pos>=8K)\n",
               stderr);
}

void seed_slot(Mirror& m, int slot, KvView k, KvView v, const float* table, int src_device, int end_row) {
    if (!m.mirror_flash || slot < 0 || slot >= kFullVisits || end_row <= 0 || end_row > m.capacity ||
        !k.scales || !v.scales) return;
    const int start = m.valid[slot];
    if (start >= end_row && m.table_ready[slot]) return;
    int prev = 0;
    check(hipGetDevice(&prev), "seed_slot getDevice");
    RestoreDevice restore(prev);
    m.used = true;
    const OwnerPlanes src{{nullptr, nullptr, nullptr, nullptr,
                           reinterpret_cast<const Word*>(k.data), reinterpret_cast<const Word*>(v.data),
                           reinterpret_cast<const Word*>(k.scales), reinterpret_cast<const Word*>(v.scales)}};
    const CompactKv kv = m.planes(slot);
    float* table_dst = m.tables + size_t(slot) * kTableWords;
    if (src_device == m.helper.device) {
        check(hipSetDevice(m.helper.device), "seed_slot helper dev");
        if (start < end_row) {
            const int rows = end_row - start;
            gather_upper_kv_direct<<<dim3(blocks(rows * kCodeWords), Fields - K), kThreads, 0, m.helper.stream>>>(
                src, kv, rows, start);
            check(hipGetLastError(), "seed_slot local gather");
        }
        if (!m.table_ready[slot] && table) {
            check(hipMemcpyAsync(table_dst, table, kTableWords * sizeof(float), hipMemcpyDeviceToDevice, m.helper.stream),
                  "seed_slot local table");
            m.table_ready[slot] = true;
        }
        m.valid[slot] = std::max(m.valid[slot], end_row);
    } else if (src_device == m.owner.device) {
        check(hipSetDevice(m.owner.device), "seed_slot owner dev");
        for (int at = start; at < end_row;) {
            const int n = std::min(kSeedRows, end_row - at);
            gather_upper<<<dim3(blocks(n * kCodeWords), Fields - K), kThreads, 0, m.owner.stream>>>(
                src, m.seed, n, K, at);
            check(hipGetLastError(), "seed_slot owner gather");
            for (int f = K; f < Fields; ++f) {
                const int w = width(f);
                check(hipMemcpyPeerAsync(kv.at(f) + size_t(at) * w, m.helper.device,
                                         m.seed + (offset(f) - offset(K)) * n, m.owner.device,
                                         size_t(n) * w * sizeof(Word), m.owner.stream),
                      "seed_slot owner P2P");
            }
            at += n;
        }
        if (!m.table_ready[slot] && table) {
            check(hipMemcpyPeerAsync(table_dst, m.helper.device, table, m.owner.device,
                                     kTableWords * sizeof(float), m.owner.stream),
                  "seed_slot owner table");
            m.table_ready[slot] = true;
        }
        m.valid[slot] = std::max(m.valid[slot], end_row);
    }
}

void sync_seed(Mirror& m) {
    if (!m.used) return;
    m.owner.wait(m.owner.stream);
    m.helper.wait(m.helper.stream);
}

bool try_flash_mirror(Mirror& m, const Attend& a, int slot) {
    if (!m.mirror_flash || a.kv_heads != kOwnerKvHeads || a.window || a.ring || !a.k.scales || !a.v.scales ||
        slot < 0 || slot >= kFullVisits || a.n < 64 || a.n > kMirrorMaxRows || a.pos < 8192 ||
        a.pos > m.capacity - a.n)
        return false;
    const int pos = static_cast<int>(a.pos), rows = a.n, end = pos + rows;
    int& valid = m.valid[slot];
    valid = std::min(valid, pos);
    const int max_catchup = (rows >= 1024) ? 32768 : 4096;
    if (pos - valid > max_catchup) return false;

    int current = 0;
    check(hipGetDevice(&current), "flash mirror getDevice");
    if (current != m.owner.device) throw std::logic_error("Inkling flash mirror called on wrong TP2 owner");
    RestoreDevice restore(current);
    m.used = true;
    const bool use_h1 = (rows <= 256);
    try {
        const OwnerPlanes src{{reinterpret_cast<const Word*>(a.q), reinterpret_cast<const Word*>(a.rel),
                               reinterpret_cast<const Word*>(a.gate), reinterpret_cast<const Word*>(a.sink),
                               reinterpret_cast<const Word*>(a.k.data), reinterpret_cast<const Word*>(a.v.data),
                               reinterpret_cast<const Word*>(a.k.scales), reinterpret_cast<const Word*>(a.v.scales)}};
        const CompactKv kv = m.planes(slot);
        const auto comm = m.owner.stream, model = m.model_stream, work = m.helper.stream;
        check(hipEventRecord(m.owner.ready, model), "mirror record inputs ready");
        check(hipStreamWaitEvent(comm, m.owner.ready, 0), "mirror wait inputs");
        for (int at = valid; at < pos;) {
            const int n = std::min(kSeedRows, pos - at);
            gather_upper<<<dim3(blocks(n * kCodeWords), Fields - K), kThreads, 0, comm>>>(src, m.seed, n, K, at);
            check(hipGetLastError(), "mirror gather catchup");
            for (int f = K; f < Fields; ++f) {
                const int w = width(f);
                check(hipMemcpyPeerAsync(kv.at(f) + size_t(at) * w, m.helper.device,
                                         m.seed + (offset(f) - offset(K)) * n, m.owner.device,
                                         size_t(n) * w * sizeof(Word), comm),
                      "mirror send catchup");
            }
            at += n;
        }
        gather_upper<<<dim3(blocks(rows * kQueryWords), Fields), kThreads, 0, comm>>>(src, m.mirror_send, rows, Q, pos);
        check(hipGetLastError(), "mirror gather packet");
        check(hipMemcpyPeerAsync(m.mirror_packet, m.helper.device, m.mirror_send, m.owner.device,
                                 size_t(rows) * offset(Fields) * sizeof(Word), comm),
              "mirror send packet");
        float* table = m.tables + size_t(slot) * kTableWords;
        if (!m.table_ready[slot])
            check(hipMemcpyPeerAsync(table, m.helper.device, a.table, m.owner.device,
                                     kTableWords * sizeof(float), comm),
                  "mirror send bias table");
        check(hipEventRecord(m.owner.sent, comm), "mirror record packet sent");

        // Owner computes lower 2 KV heads (query heads 0..5) in parallel with helper.
        attend_flash_lean_half(a, use_h1, kOwnerKvHeads, model);

        // Helper appends new K/V rows and computes upper 2 KV heads (query heads 6..11).
        check(hipSetDevice(m.helper.device), "mirror select helper");
        check(hipStreamWaitEvent(work, m.owner.sent, 0), "mirror wait packet");
        append_upper<<<dim3(blocks(rows * kCodeWords), Fields - K), kThreads, 0, work>>>(kv, m.mirror_packet, rows, pos);
        check(hipGetLastError(), "mirror append rows");
        Attend b = a;
        b.q = reinterpret_cast<const float*>(m.mirror_packet);
        b.rel = reinterpret_cast<const float*>(m.mirror_packet + offset(Rel) * rows);
        b.gate = reinterpret_cast<const float*>(m.mirror_packet + offset(Gate) * rows);
        b.sink = reinterpret_cast<const float*>(m.mirror_packet + offset(Sink) * rows);
        b.table = table;
        b.k = {reinterpret_cast<const uint8_t*>(kv.at(K)), reinterpret_cast<const uint16_t*>(kv.at(KScale))};
        b.v = {reinterpret_cast<const uint8_t*>(kv.at(V)), reinterpret_cast<const uint16_t*>(kv.at(VScale))};
        b.kv_heads = kHalfKvHeads;
        b.out = reinterpret_cast<float*>(m.mirror_answer);
        attend_flash_lean_half(b, use_h1, kHalfKvHeads, work);
        check(hipMemcpyPeerAsync(m.mirror_recv, m.owner.device, m.mirror_answer, m.helper.device,
                                 size_t(rows) * kQueryWords * sizeof(Word), work),
              "mirror return answer");
        check(hipEventRecord(m.helper.ready, work), "mirror record answer ready");

        check(hipSetDevice(m.owner.device), "mirror return owner");
        check(hipStreamWaitEvent(model, m.helper.ready, 0), "mirror wait answer");
        unpack_answer_out<<<blocks(rows * kQueryWords), kThreads, 0, model>>>(
            m.mirror_recv, reinterpret_cast<Word*>(a.out), rows);
        check(hipGetLastError(), "mirror unpack answer");
        m.table_ready[slot] = true;
        valid = end;
        return true;
    } catch (...) {
        m.owner.wait(m.owner.stream);
        m.helper.wait(m.helper.stream);
        m.valid.fill(0);
        m.table_ready.fill(false);
        throw;
    }
}

bool try_flash_rows(Mirror& m, const Attend& a) {
    // Stay on the owners for small prompts and short appends: the full-prefix DMA would not pay back.
    if (!m.bulk || a.n < 2047 || a.n > 2048 || a.pos < 8192 || a.pos > m.capacity - a.n ||
        a.window || a.ring || a.kv_heads != kOwnerKvHeads || !a.k.scales || !a.v.scales) return false;
    // Four KV groups per tile. Use 240 helper CTAs instead of 252/256 on the measured 60-CU parts;
    // the DMA and response belong to the helper's critical path, not the owner's tail.
    const int cut = m.balance_flash ? 60 * 16 : (a.n / 32) * 16;
    auto& w = *m.bulk;
    RestoreDevice restore(m.owner.device);
    m.used = true;
    // The previous reply follows its input event and all transfers. Never re-record a pending event;
    // prompt preseed on the other stream owns a distinct pair of events, including on this same visit.
    check(hipSetDevice(w.device), "select bulk Flash workspace");
    check(hipEventSynchronize(w.ready), "release previous bulk Flash packet");
    check(hipSetDevice(m.owner.device), "select bulk Flash producer");
    check(hipEventRecord(m.owner.bulk_input, m.model_stream), "publish bulk Flash inputs");
    Attend tail = a;
    tail.q += cut * kBulkQ; tail.gate += cut * kBulkQ; tail.out += cut * kBulkQ;
    tail.rel += cut * kBulkRel; tail.sink += cut * kBulkHeads; tail.pos += cut; tail.n -= cut;
    attend_flash_lean(tail, m.model_stream);

    check(hipSetDevice(w.device), "select bulk Flash helper");
    try {
        check(hipStreamWaitEvent(w.stream, m.owner.bulk_input, 0), "wait for bulk Flash inputs");
        // Only one visit is staged. Never read beyond this tile range's final visible key. Recopy from
        // canonical storage also after restore, with no persistent mirror cursor or rollback protocol.
        auto* p = reinterpret_cast<uint8_t*>(w.memory);
        const auto copy = [&](auto* src, size_t count) {
            using T = std::remove_pointer_t<decltype(src)>;
            auto* dst = reinterpret_cast<T*>(p);
            const size_t bytes = count * sizeof(T);
            check(hipMemcpyPeerAsync(p, w.device, src, m.owner.device, bytes, w.stream), "stage bulk Flash inputs");
            p += bytes;
            return dst;
        };
        Attend h = a;
        h.n = cut;
        const size_t keys = a.pos + cut;
        h.k.data = copy(a.k.data, keys * kBulkCode); h.k.scales = copy(a.k.scales, keys * kBulkScale);
        h.v.data = copy(a.v.data, keys * kBulkCode); h.v.scales = copy(a.v.scales, keys * kBulkScale);
        h.q = copy(a.q, cut * kBulkQ); h.gate = copy(a.gate, cut * kBulkQ);
        h.rel = copy(a.rel, cut * kBulkRel); h.sink = copy(a.sink, cut * kBulkHeads);
        h.table = copy(a.table, kTableWords); h.out = reinterpret_cast<float*>(p);
        attend_flash_lean(h, w.stream);
        check(hipMemcpyPeerAsync(a.out, m.owner.device, h.out, w.device, size_t(cut) * kBulkQ * sizeof(float),
                                 w.stream), "return bulk Flash rows");
        check(hipEventRecord(w.ready, w.stream), "publish bulk Flash result");
        check(hipSetDevice(m.owner.device), "select bulk Flash owner");
        check(hipStreamWaitEvent(m.model_stream, w.ready, 0), "join bulk Flash results");
    } catch (...) {
        w.wait(w.stream);
        throw;
    }
    return true;
}

void set_budget(Mirror& mirror, int64_t prompt_rows, int64_t max_new) {
    // A 64K/256 generation amortized the peer seed; a sampled /16 request did not.
    mirror.long_answer = prompt_rows >= kv_helper_min_pos() && max_new >= kv_helper_min_new();
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
    const int min_pos = kv_helper_min_pos();
    const int preseed_end = std::min(kPreseedEnd, min_pos);
    if (!m.long_answer || pos - valid > kPreseedEnd) return Visit::Local;
    const bool preseed = rows >= kPromptRows && end >= preseed_end;
    int score_rows = 0;
    if (!preseed) {
        if (pos < min_pos || rows > m.decode_max_rows || !partial || !scores || !q8)
            return Visit::Local;
        // The owner's score slab has the full 12-head stride despite computing only six heads.
        // T<=4 keeps its existing eligibility. Wider calls may reuse that slab across groups of rows.
        const size_t fit = score_floats / (size_t(end) * kHeads * 2);
        if (!fit || (rows <= kDefaultDecodeRows && fit < size_t(rows))) return Visit::Local;
        score_rows = static_cast<int>(std::min(size_t(rows), fit));
    }

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
        attend_decode_half(a, partial, scores, q8, model, score_rows);

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
