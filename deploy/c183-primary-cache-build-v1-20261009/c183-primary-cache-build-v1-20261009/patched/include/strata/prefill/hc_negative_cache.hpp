// Experimental host-only, primary public-run HC W-negative cache. No GPU dependencies.
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

namespace strata::prefill::hc_negative {
inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HC_WNEG_PRIMARY");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}
// Versioned policy: W * 4096, X * 512, exact round-trip guard, finite/range guard.
constexpr unsigned exact_scaled_policy = 1;
constexpr int capacity = 256;
class Request;
inline thread_local Request* current = nullptr;

class Request {
public:
    struct Context {
        const void *owner, *table, *model, *session, *stream;
        int device;
    };
    struct Entry {
        const void* w = nullptr;
        int64_t n = 0, k = 0;
        unsigned policy = 0, bad = 0;
        uint64_t hits = 0;
        bool occupied = false, poisoned = false;
    };
    struct Counters { uint64_t candidates = 0, hits = 0, learned = 0, mismatches = 0; } counts;
    // Admission lasts through normal drain, or clears on primary exception unwind.
    // Peer threads never access this TLS state; existing pipeline exception semantics stay intact.
    Request(std::atomic_flag& gate, Context ctx, bool requested, bool primary = true) : ctx_(ctx) {
        if (!requested) return;
        if (current || gate.test_and_set(std::memory_order_acquire)) {
            rejected_ = true;
            return;
        }
        gate_ = &gate;
        const char* trace = std::getenv("STRATA_HC_WNEG_QUALIFY");
        trace_ = trace && trace[0] == '1' && trace[1] == '\0';
        live_ = primary && ctx.owner && ctx.table && ctx.model && ctx.session && ctx.stream && ctx.device >= 0;
        current = this;
    }
    ~Request() {
        if (gate_) {
            // Same binary and counters in qualification and timing; no per-call env read or clock.
            if (trace_) {
            std::fprintf(stderr, "strata hc-wneg: device=%d candidates=%llu hits=%llu learned=%llu mismatches=%llu\n",
                         ctx_.device, (unsigned long long)counts.candidates, (unsigned long long)counts.hits,
                         (unsigned long long)counts.learned, (unsigned long long)counts.mismatches);
            for (int i = 0; i < capacity; ++i) if (entries_[i].bad || entries_[i].hits)
                std::fprintf(stderr, "strata hc-wneg-key: ordinal=%d N=%lld K=%lld mask=%u hits=%llu\n",
                             i, (long long)entries_[i].n, (long long)entries_[i].k,
                             entries_[i].bad, (unsigned long long)entries_[i].hits);
            }
            invalidate();
            current = nullptr;
            gate_->clear(std::memory_order_release);
        }
    }
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    bool rejected() const { return rejected_; }
    bool admitted() const { return gate_ != nullptr; }
    const Context& context() const { return ctx_; }
    void invalidate() { live_ = false; ordinal_ = -1; entries_ = {}; }
    bool owns(const void* owner) const { return ctx_.owner == owner; }
    void declare(const void* owner, int64_t ordinal) {
        ordinal_ = live_ && owns(owner) && ordinal >= 0 && ordinal < capacity ? (int)ordinal : -1;
    }
    void undeclare() { ordinal_ = -1; }
    // Called only AFTER baseline eligibility, scratch capacity and overlap validation.
    bool negative(const void* owner, const void* stream, const void* w,
                  int64_t n, int64_t k, unsigned policy, bool diagnostics) {
        Entry* e = entry(owner, stream, w, n, k, policy, diagnostics);
        if (!e) return false;
        ++counts.candidates;
        if (!e->bad) return false;
        ++counts.hits;
        ++e->hits;
        return true;
    }
    // Caller certifies successful existing guard D2H AND stream sync; never passes X bits.
    void observe(const void* owner, const void* stream, const void* w,
                 int64_t n, int64_t k, unsigned policy, bool diagnostics,
                 const unsigned (&guard)[10]) {
        Entry* e = entry(owner, stream, w, n, k, policy, diagnostics);
        if (!e) return;
        const unsigned bad = (guard[1] ? 1u : 0u) | (guard[2] ? 2u : 0u) | (guard[3] ? 4u : 0u);
        if (bad && !e->bad) { e->bad = bad; ++counts.learned; }
    }
    const std::array<Entry, capacity>& entries() const { return entries_; }
private:
    Entry* entry(const void* owner, const void* stream, const void* w,
                 int64_t n, int64_t k, unsigned policy, bool diagnostics) {
        if (!live_ || !owns(owner) || stream != ctx_.stream || ordinal_ < 0 || !w ||
            policy != exact_scaled_policy || diagnostics) return nullptr;
        Entry& e = entries_[(unsigned)ordinal_];
        if (e.poisoned) return nullptr;
        if (e.occupied && (e.w != w || e.n != n || e.k != k || e.policy != policy)) {
            e.bad = 0; e.poisoned = true; ++counts.mismatches; return nullptr;
        }
        if (!e.occupied) { e.w = w; e.n = n; e.k = k; e.policy = policy; e.occupied = true; }
        return &e;
    }
    Context ctx_;
    std::atomic_flag* gate_ = nullptr;
    bool live_ = false, rejected_ = false, trace_ = false;
    int ordinal_ = -1;
    std::array<Entry, capacity> entries_{};
};
class Declaration {
public:
    Declaration(const void* owner, int64_t ordinal) : request_(current) {
        if (request_) request_->declare(owner, ordinal);
    }
    ~Declaration() { if (request_) request_->undeclare(); }
    Declaration(const Declaration&) = delete;
    Declaration& operator=(const Declaration&) = delete;
private:
    Request* request_;
};
inline void invalidate(const void* owner) {
    if (current && current->owns(owner)) current->invalidate();
}
} // namespace strata::prefill::hc_negative
