// g++ -O2 -std=c++20 tools/conversations_suffix_test.cpp src/poly/spill.cpp -o conversations_suffix_test
#include "../src/poly/conversations.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using strata::poly::CacheOptions;
using strata::poly::Conversations;

namespace {

void check(bool ok, const char* why) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        std::exit(1);
    }
}

// Distinct rank/plane strides, packed plane-by-plane as save_card_rows() does. Each byte depends
// on the token, its entire causal prefix, position, plane and byte offset.
struct ByteEngine {
    static constexpr int64_t context = 32768;
    static constexpr size_t strides[] = {16, 2, 16, 2, 16, 2, 16, 2};
    static constexpr size_t bytes_per_row = 72;
    std::array<std::vector<uint8_t>, std::size(strides)> kv;
    uint64_t hash = 1, snapshot = 0, generation = 1, anchor_generation = 0;
    int64_t pos = 0, valid = 0, dirty = 0, stage_b_valid = 0;
    size_t h2d = 0, hits = 0, queries = 0, b_clears = 0, state_loads = 0;
    uint64_t last_snapshot = 0;
    bool enabled = true, fail_load = false;

    explicit ByteEngine(bool enable) : enabled(enable) {
        for (size_t p = 0; p < kv.size(); ++p) kv[p].resize(context * strides[p], 0xCC);
    }
    static uint64_t mix(uint64_t h, uint64_t x) { return (h ^ x) * 0x9e3779b97f4a7c15ULL; }
    static uint8_t byte(uint64_t h, int64_t i, size_t p, size_t j) {
        const auto x = mix(h, (uint64_t(i) << 16) | (p << 8) | j);
        return uint8_t((x >> 40) ^ (x >> 23) ^ x);
    }
    void reset() { pos = 0; hash = 1; stage_b_valid = 0; } // whole full KV is not zeroed
    void write(const int32_t* ids, int64_t n, int64_t keep) {
        check(0 <= keep && keep <= n && pos + n <= context, "mock write bounds");
        dirty = std::max(dirty, pos + n); // before any rank writes; independent of keep
        uint64_t h = hash;
        for (int64_t i = 0; i < n; ++i) {
            h = mix(h, uint32_t(ids[i]));
            for (size_t p = 0; p < kv.size(); ++p)
                for (size_t j = 0; j < strides[p]; ++j)
                    kv[p][(pos + i) * strides[p] + j] = byte(h, pos + i, p, j);
            if (i < keep) hash = h;
        }
        pos += keep;
    }
    size_t row_bytes() const { return bytes_per_row; }
    size_t state_bytes() const { return sizeof(hash); }
    void save_state(uint8_t* dst) const { std::memcpy(dst, &hash, sizeof(hash)); }
    void load_state(const uint8_t* src, int64_t at) {
        check(src != nullptr, "restore must still load the checkpoint state");
        std::memcpy(&hash, src, sizeof(hash));
        pos = at;
        ++state_loads;
    }
    void save_rows(int64_t first, int64_t n, uint8_t* dst) const {
        check(first >= 0 && first + n <= pos, "never save outside the committed live sequence");
        for (size_t p = 0; p < kv.size(); ++p) {
            std::memcpy(dst, kv[p].data() + first * strides[p], n * strides[p]);
            dst += n * strides[p];
        }
    }
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
        check(from <= first && first + n <= from + count, "load respects the original blob offsets");
        dirty = std::max(dirty, first + n);
        for (size_t p = 0; p < kv.size(); ++p) {
            std::memcpy(kv[p].data() + first * strides[p], src + (first - from) * strides[p], n * strides[p]);
            if (fail_load) throw std::runtime_error("injected partial row load failure");
            src += count * strides[p];
        }
        h2d += n * bytes_per_row;
    }
    bool kv_prefix_tags() const { return false; }
    bool kv_canonical_suffix_supported() const { return enabled; }
    void arm_canonical_suffix(uint64_t id, int64_t hi) {
        check(id > last_snapshot && hi > 0 && hi <= pos, "arm only a unique, live snapshot");
        last_snapshot = id;
        if (!snapshot || hi > valid - dirty) {
            snapshot = id;
            valid = hi;
            dirty = 0;
            anchor_generation = generation;
        }
    }
    int64_t canonical_suffix_skip_from(uint64_t id, int64_t need) {
        ++queries;
        if (id && id == snapshot && generation == anchor_generation && need <= valid && dirty < need) {
            ++hits;
            return dirty;
        }
        return need;
    }
    void invalidate_canonical_suffix(uint64_t id = 0) {
        if (!id || id == snapshot) snapshot = 0;
    }
    void clear_stage_b_kv_valid() { stage_b_valid = 0; ++b_clears; }

    void exact(const std::vector<int32_t>& ids, int64_t n) const {
        check(pos == n && n <= (int64_t) ids.size(), "restored position");
        uint64_t h = 1;
        for (int64_t i = 0; i < n; ++i) {
            h = mix(h, uint32_t(ids[i]));
            for (size_t p = 0; p < kv.size(); ++p)
                for (size_t j = 0; j < strides[p]; ++j) {
                    if (kv[p][i * strides[p] + j] != byte(h, i, p, j)) {
                        std::fprintf(stderr, "FAIL: KV row %lld, plane %zu, byte %zu\n", (long long) i, p, j);
                        std::exit(1);
                    }
                }
        }
        check(hash == h, "recurrent state is exact");
    }
};

std::vector<int32_t> tokens(int32_t base, int64_t n) {
    std::vector<int32_t> ids(n);
    for (int64_t i = 0; i < n; ++i) ids[i] = base + int32_t(i);
    return ids;
}

struct Chat {
    ByteEngine e;
    Conversations<ByteEngine> kept;
    explicit Chat(int slots = 3, bool enable = true)
        : e(enable), kept(e, CacheOptions{.every = 0, .park_mib = 16, .park_slots = slots,
                                        .admit = {}, .log = nullptr}) {}
    void finish(const std::vector<int32_t>& ids, bool checkpoint = true) {
        if (checkpoint && e.pos < (int64_t) ids.size()) {
            append(ids, (int64_t) ids.size() - 1);
            kept.before(e.pos, true);
        }
        append(ids, (int64_t) ids.size());
        kept.prompt_was((int64_t) ids.size());
        e.exact(ids, (int64_t) ids.size());
    }
    void append(const std::vector<int32_t>& ids, int64_t end) {
        const int64_t n = end - e.pos;
        check(n >= 0, "append range");
        if (!n) return;
        const int32_t* t = ids.data() + e.pos;
        e.write(t, n, n);
        kept.read(t, n);
    }
    void resume(const std::vector<int32_t>& ids, int64_t expected) {
        check(kept.resume(ids) == expected, "resume reaches the intended checkpoint");
        e.exact(ids, expected);
    }
};

const auto a = tokens(100'000, 23'284);
const auto b = tokens(200'000, 1'290);
const auto c = tokens(300'000, 1'290);
constexpr int64_t a_mark = 23'283, b_mark = 1'289;

void detour(bool park_b, bool speculate, bool enable = true) {
    Chat t(3, enable);
    t.resume(a, 0);
    t.finish(a);
    t.resume(b, 0);
    const auto a_snap = t.e.snapshot;
    if (enable) check(a_snap != 0, "A is anchored after a clean park");
    if (speculate) {
        t.append(b, b_mark);
        if (park_b) t.kept.before(t.e.pos, true);
        const auto draft = tokens(b.back(), 8);
        t.e.write(draft.data(), 8, 1);
        t.kept.read(draft.data(), 1);
        t.e.exact(b, (int64_t) b.size());
    } else t.finish(b, park_b);
    const auto calls = t.e.state_loads;
    t.e.stage_b_valid = b_mark;
    t.resume(a, a_mark);
    const int64_t rows = enable ? int64_t(b.size()) + (speculate ? 7 : 0) : a_mark;
    check(t.e.h2d == rows * ByteEngine::bytes_per_row, "reload exactly the dirty prefix");
    check(t.e.state_loads == calls + 1, "suffix skip does not elide state load");
    check(t.e.stage_b_valid == 0 && t.e.b_clears == 1, "cross-slot invalidates Stage B");
    check(t.e.hits == size_t(enable) && t.e.queries == size_t(enable), "query exactly once before H2D");
    if (enable && park_b) check(t.e.last_snapshot > a_snap, "B tried to arm without replacing A");
    if (enable) check(rows * 10 < a_mark, "skip more than 90% of restore bytes");
    t.finish(a);
    std::printf("PASS %s: reloaded %lld/%lld rows (%zu/%zu bytes), skipped %.2f%%, every KV byte exact\n",
                !enable ? "default-off" : speculate ? "T=8 keep=1" : park_b ? "parkable detour" : "short detour",
                (long long) rows, (long long) a_mark, t.e.h2d, a_mark * ByteEngine::bytes_per_row,
                100.0 * (a_mark - rows) / a_mark);
}

void zero_copy() {
    Chat t;
    t.resume(a, 0);
    t.finish(a);
    t.resume(b, 0); // park/reset, but don't write B
    t.e.stage_b_valid = a_mark;
    t.resume(a, a_mark);
    check(t.e.hits == 1 && t.e.h2d == 0 && t.e.b_clears == 1 && t.e.stage_b_valid == 0,
          "zero-H2D restore still clears Stage B");
    t.finish(a);
    std::puts("PASS zero-H2D cross-slot restore clears Stage B and restores exact state");
}

void evict_branch_cancel_drop() {
    Chat t(1);
    t.resume(a, 0);
    t.finish(a);
    t.resume(b, 0);
    const auto a_snap = t.e.snapshot;
    t.finish(b);
    t.resume(c, 0); // parking B to take C evicts A
    check(t.kept.evictions() == 1 && t.e.snapshot != a_snap, "eviction retires A's anchor");
    t.finish(c, false);
    t.resume(a, 0);
    t.append(a, 1'100);
    t.kept.before(t.e.pos, true);
    t.finish(a);
    auto fork = a;
    fork[1'100] = 400'000;
    t.resume(fork, 1'100); // in-slot branch; original is parked, but neither slot inherits an anchor
    check(t.e.snapshot == 0, "branch doesn't publish a suffix");
    const auto draft = tokens(fork[1'100] + 1, 8);
    t.e.write(draft.data(), 8, 0); // cancelled read
    check(t.kept.rewind() == 1'100, "cancel rewinds to fork checkpoint");
    t.e.exact(fork, 1'100);
    t.kept.drop(); // error path
    check(t.e.snapshot == 0, "drop invalidates suffix proof");
    t.e.reset();
    t.resume(a, a_mark);
    check(t.e.hits == 0 && t.e.h2d == a_mark * ByteEngine::bytes_per_row, "zero stale hits after evict/branch/cancel/drop");
    t.finish(a);
    std::puts("PASS evict/branch/cancel/rewind/drop: zero stale suffix hits, every KV byte exact");
}

void partial_enter_branch() {
    constexpr int64_t fork_at = 2'048;
    Chat t;
    auto fork = a;
    fork[fork_at] = 400'000;
    t.resume(a, 0);
    t.append(a, fork_at);
    t.kept.before(fork_at, true);
    t.finish(a);
    t.resume(b, 0);
    // Invalidate A's canonical suffix, without making its host copy unavailable.
    t.e.invalidate_canonical_suffix();
    t.finish(b, false);
    t.resume(fork, fork_at); // only the early checkpoint was loaded, but A can still be parked from its blobs
    check(t.e.last_snapshot == 1, "enter->branch did not re-arm the unloaded suffix");
    t.append(fork, fork_at + 1);
    const auto before = t.e.h2d;
    t.resume(a, a_mark);
    check(t.e.hits == 0 && t.e.h2d - before == (a_mark - fork_at) * ByteEngine::bytes_per_row,
          "branch restore uses the shared live prefix and original blob strides, not unloaded rows");
    t.finish(a);
    std::puts("PASS partial enter->branch: no stale anchor, every KV byte exact");
}

void reload_tail_blob(bool rebind) {
    Chat t;
    auto short_a = a;
    short_a.resize(1'025);
    auto long_b = tokens(200'000, 1'600);
    t.resume(short_a, 0);
    t.finish(short_a);
    t.resume(b, 0);
    t.resume(a, 1'024);
    t.finish(a); // next park saves the tail into a second blob
    t.resume(long_b, 0);
    t.finish(long_b, false);
    const size_t before = t.e.h2d;
    if (rebind) {
        ++t.e.generation;
        for (auto& plane : t.e.kv) std::fill(plane.begin(), plane.end(), 0xEE);
    }
    t.resume(a, a_mark);
    check(t.e.h2d - before == (rebind ? a_mark : 1'600) * ByteEngine::bytes_per_row,
          "reload spans both saved blobs, and alloc generation mismatch forces a full reload");
    t.finish(a);
    std::puts(rebind ? "PASS allocation generation change: full reload, every KV byte exact" :
                       "PASS multi-blob restore: partial second-blob H2D, every KV byte exact");
}

void load_error() {
    Chat t;
    t.resume(a, 0);
    t.finish(a);
    t.resume(b, 0);
    t.finish(b, false);
    t.e.fail_load = true;
    try {
        t.kept.resume(a);
        check(false, "injected load failure should propagate");
    } catch (const std::runtime_error&) {
        check(t.e.snapshot == 0, "failed load retires the anchor even before caller drops");
    }
    t.e.fail_load = false;
    t.kept.drop();
    t.e.reset();
    const size_t hits = t.e.hits;
    t.resume(a, a_mark);
    check(t.e.hits == hits && t.e.h2d == a_mark * ByteEngine::bytes_per_row, "retry after error reloads every row");
    t.finish(a);
    std::puts("PASS partial restore error and retry: zero stale suffix hits, every KV byte exact");
}

}  // namespace

int main() {
    detour(false, false);
    detour(true, false);
    detour(true, true);
    detour(false, false, false);
    zero_copy();
    evict_branch_cancel_drop();
    partial_enter_branch();
    reload_tail_blob(false);
    reload_tail_blob(true);
    load_error();
}
