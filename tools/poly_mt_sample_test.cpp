// Host-only parity test and warm-row benchmark. From the repository root:
/*
g++ -std=c++20 -O3 -mavx2 -pthread -Iinclude -ffunction-sections -fdata-sections \
    tools/poly_mt_sample_test.cpp src/spec/draft_policy.cpp -Wl,--gc-sections -o poly_mt_sample_test
STRATA_POLY_MT_SAMPLE=1 ./poly_mt_sample_test
STRATA_POLY_MT_FILTER_SAMPLE=0 ./poly_mt_sample_test --filtered-parity 0
STRATA_POLY_MT_FILTER_SAMPLE=1 ./poly_mt_sample_test --filtered-parity 1
STRATA_POLY_MT_FILTER_SAMPLE=1 STRATA_POLY_MT_SAMPLE=4 ./poly_mt_sample_test --filtered-bench
./poly_mt_sample_test --prefilter-bench
*/
// Compile the actual sampler code; the linker drops the unused GPU runtime.
#define main strata_poly_unused_main
#include "../src/poly/main.cpp"
#undef main

#include <cassert>
#include <numeric>
#include <limits>

// Host stand-in for Engine's greedy fallback: the same max (NaNs ignored), then the first equal id.
namespace strata::poly {
int most_likely(const float* logits, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = std::max(mx, logits[i]);
    const auto at = std::find(logits, logits + n, mx);
    return at == logits + n ? 0 : (int) (at - logits);
}
}  // namespace strata::poly

namespace {
struct SamplerHostTest {
    using Pool = Sampler::SamplePool;
    static constexpr int kVocab = 200058;
    static constexpr int kRows = 500;

    static int configured_threads() { return Sampler::mt_sample_threads(); }

    static Sampling sampling(float temperature) {
        Sampling s;
        s.temperature = temperature;
        s.top_k = 0;
        return s;
    }

    template <class T>
    static void equal(const std::vector<T>& a, const std::vector<T>& b) {
        assert(a.size() == b.size());
        assert(a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
    }

    static void parity(Pool& p4, Pool& p8) {
        Sampler serial, four, eight, via_pick;
        const char* avx2 = std::getenv("STRATA_POLY_AVX2_SAMPLE");
        const bool pick_avx2 = configured_threads() > 0 || (avx2 && std::strcmp(avx2, "1") == 0);
        std::mt19937_64 srng(123), frng(123), erng(123), prng(123);
        std::mt19937 gen(20261126);
        std::normal_distribution<float> dist(0.f, 6.f);
        std::vector<float> logits(kVocab);
        const auto check = [&](int n, float temp = 1.f) {
            const int nr = Sampler::n_runs(n);
            const auto s = sampling(temp);
            const float mx = Sampler::max_avx2(logits.data(), n);
            for (Pool* pool : {&p4, &p8}) {
                const float mt_mx = pool->max(logits.data(), n, nr);
                assert(std::memcmp(&mx, &mt_mx, sizeof(mx)) == 0);
            }
            const int ref = serial.draw_all<true>(logits.data(), n, s, srng);
            assert(four.draw_all<true>(logits.data(), n, s, frng, &p4) == ref);
            assert(eight.draw_all<true>(logits.data(), n, s, erng, &p8) == ref);
            if (pick_avx2) {
                assert(via_pick.pick(logits.data(), n, s, {}, prng) == ref);
                const bool pooled = configured_threads() > 0 && nr >= configured_threads();
                assert(bool(via_pick.sample_pool_) == pooled);
                equal(serial.q_, via_pick.q_);
                equal(serial.runs_, via_pick.runs_);
                assert(srng == prng);
            }
            equal(serial.q_, four.q_);
            equal(serial.q_, eight.q_);
            equal(serial.runs_, four.runs_);
            equal(serial.runs_, eight.runs_);
            assert(srng == frng && srng == erng);
        };
        for (int i = 0; i < kRows; ++i) {
            for (float& l : logits) l = dist(gen);
            check(kVocab);
        }
        std::printf("Random parity: %d/%d rows (max, every weight and run, token ID, RNG state)\n", kRows, kRows);

        constexpr float nan = std::numeric_limits<float>::quiet_NaN();
        std::fill(logits.begin(), logits.end(), -1.f);
        logits[7] = -0.f;
        logits[8] = 0.f;
        logits[25 * Sampler::kSumRun] = 0.f;
        check(kVocab);  // first zero is not in the first AVX2 lane
        logits[7] = 0.f;
        logits[8] = -0.f;
        check(kVocab);
        std::fill(logits.begin(), logits.end(), -INFINITY);
        logits[Sampler::kSumRun * 19 + 7] = -0.f;
        logits[kVocab - 1] = 0.f;
        logits[0] = logits[Sampler::kSumRun * 7] = logits[kVocab - 2] = nan;
        logits[kVocab - 3] = std::numeric_limits<float>::signaling_NaN();
        check(kVocab);
        logits[kVocab - 1] = INFINITY;
        check(kVocab);  // no draw, buffers and RNG unchanged
        std::fill(logits.begin(), logits.end(), -INFINITY);
        check(kVocab);
        std::fill(logits.begin(), logits.end(), nan);
        check(kVocab);
        std::fill(logits.begin(), logits.end(), -2.f);
        logits[kVocab - 1] = -1.f;
        check(kVocab, std::numeric_limits<float>::denorm_min());  // reciprocal overflow
        check(kVocab, INFINITY);  // zero inverse temperature
        check(8 * Sampler::kSumRun);
        for (int tail = 1; tail <= 8; ++tail) check(8 * Sampler::kSumRun + tail, 0.7f);
        const unsigned csr = _mm_getcsr();
        _mm_setcsr((csr & ~_MM_ROUND_MASK) | _MM_ROUND_DOWN);
        check(kVocab, 0.7f);  // change the caller's rounding mode after creating the workers
        _mm_setcsr(csr);
        std::puts("Edge parity: 18/18 cases");
    }

    struct FourShards {
        std::array<std::vector<float>, 4> storage;
        LogitShards view;

        explicit FourShards(const std::vector<float>& row) {
            view.len = {50015, 50014, 50015, 50014};
            int at = 0;
            for (size_t i = 0; i < storage.size(); ++i) {
                storage[i].assign(row.begin() + at, row.begin() + at + view.len[i]);
                view.ptr[i] = storage[i].data();
                at += view.len[i];
            }
            assert((size_t) at == row.size());
        }
    };

    static int stock_pick(Sampler& sampler, LogitShards logits, const Sampling& s,
                          const std::vector<int32_t>& history, std::mt19937_64& rng) {
        if (s.top_k == 0 && s.top_p >= 1.f && s.min_p <= 0.f)
            return sampler.pick(logits, kVocab, s, history, rng);
        if (Sampler::penalizes(s)) {
            sampler.pen_.resize(kVocab);
            logits.copy(0, kVocab, sampler.pen_.data());
            sampler.penalize(s, history);
            logits = {sampler.pen_.data(), kVocab};
        }
        sampler.stock_candidates(logits, kVocab, s.top_k == 0 ? kVocab : std::min(s.top_k, kVocab));
        return sampler.draw_filtered(s, rng);
    }

    static void equal_candidates(const Sampler& a, const Sampler& b) {
        assert(a.cand_.size() == b.cand_.size());
        for (size_t i = 0; i < a.cand_.size(); ++i) {
            assert(a.cand_[i].id == b.cand_[i].id);
            assert(std::bit_cast<uint32_t>(a.cand_[i].logit) == std::bit_cast<uint32_t>(b.cand_[i].logit));
        }
    }

    static std::vector<float> normal_logits() {
        std::mt19937 gen(17);
        std::normal_distribution<float> dist(0.f, 6.f);
        std::vector<float> row(kVocab);
        for (float& v : row) v = dist(gen);
        return row;
    }

    static void filtered_parity(bool enabled) {
        // pick() is exercised with the process-wide opt-in both off and on (separate invocations).
        assert(Sampler::filtered_sample_enabled() == enabled);
        const auto normal = normal_logits();
        const std::vector<int32_t> history{0, 50014, 42, 42, 50015, 100029, 50015, 150044, kVocab - 1};
        const std::vector<int32_t> no_history;
        std::mt19937_64 seeds(0xbb1226);
        Sampler stock, via_pick;
        int counts[4] = {};
        int layouts = 0;
        auto matrix = [&](const char* name, const std::vector<float>& row, bool penalties = false) {
            const FourShards shards(row);
            for (auto logits : {LogitShards(row.data(), kVocab), shards.view}) {
                ++layouts;
                for (int k : {0, 1, 5, 20, 64, 256})
                    for (float p : {0.5f, 0.95f, 1.f})
                        for (float min : {0.f, 0.05f, 0.5f})
                            for (float temp : {0.1f, 0.6f, 0.7f, 0.8f, 1.f, 1.5f, 1e-35f}) {
                                Sampling s = sampling(temp);
                                s.top_k = k;
                                s.top_p = p;
                                s.min_p = min;
                                if (penalties) {
                                    s.penalty_last_n = temp < 0.8f ? -1 : 6;
                                    s.penalty_repeat = 1.1f;
                                    s.penalty_freq = 0.19f;
                                    s.penalty_present = 0.2f;
                                }
                                const auto& h = penalties ? history : no_history;
                                const auto seed = seeds();
                                std::mt19937_64 a(seed), b(seed);
                                const int ref = stock_pick(stock, logits, s, h, a);
                                const int got = via_pick.pick(logits, kVocab, s, h, b);
                                if (ref != got || a != b || a() != b()) {
                                    std::fprintf(stderr, "%s (%s): k=%d p=%g min=%g temp=%g seed=%llu: %d != %d or RNG mismatch\n",
                                                 name, logits.split() ? "shards" : "contiguous", k, p, min, temp,
                                                 (unsigned long long) seed, ref, got);
                                    std::abort();
                                }
                                const int path = k > 0 ? 0 : p < 1.f ? 2 : min > 0.f ? 1 : 3;
                                if (path != 3) {
                                    if (stock.cand_.size() != via_pick.cand_.size())
                                        std::fprintf(stderr, "%s: k=%d p=%g min=%g temp=%g, candidate count %zu != %zu\n",
                                                     name, k, p, min, temp, stock.cand_.size(), via_pick.cand_.size());
                                    equal_candidates(stock, via_pick);
                                    equal(stock.p_, via_pick.p_);
                                }
                                ++counts[path];
                            }
            }
        };
        matrix("normal", normal);
        std::vector<float> row(kVocab, -120.f);
        row[100029] = 40.f;
        matrix("sharp peak", row);
        std::fill(row.begin(), row.end(), 1.f);
        matrix("uniform", row);
        for (int cut : {50015, 100029, 150044}) std::fill_n(row.begin() + cut - 47, 94, 10.f);
        matrix("shard boundary ties", row);
        for (int i = 0; i < kVocab; ++i) row[i] = i % 2 ? 0.f : -0.f;
        matrix("signed zeros", row);
        row = normal;
        for (int i = 1; i < kVocab; i += 47) row[i] = -INFINITY;
        matrix("negative infinity", row);
        row[50014] = row[50015] = row.back() = INFINITY;
        matrix("positive infinity", row);
        row = normal;
        row[7] = std::numeric_limits<float>::quiet_NaN();
        row[100028] = std::numeric_limits<float>::signaling_NaN();
        row.back() = std::numeric_limits<float>::quiet_NaN();
        matrix("NaN fallback", row);
        for (int i = 0; i < kVocab; ++i) {
            const float floor = 3.f + std::log(i % 8 < 4 ? 0.05f : 0.5f);
            row[i] = std::nextafter(floor, i % 4 == 0 ? -INFINITY : i % 4 == 1 ? INFINITY : floor);
        }
        row[150043] = row[150044] = 3.f;
        matrix("exact min-p floors", row);
        matrix("penalty history", normal, true);
        std::printf("Filtered flag %d: %d/%d matrix checks (%d layout/fixture pairs; A=%d B=%d C=%d unfiltered=%d), token, RNG, candidate bits and probabilities\n",
                    enabled, std::accumulate(counts, counts + 4, 0), std::accumulate(counts, counts + 4, 0),
                    layouts, counts[0], counts[1], counts[2], counts[3]);
    }

    static void candidate_edges() {
        Sampler ref, test;
        Pool four(4), eight(8);
        std::vector<float> row(kVocab, -20.f);
        const float tiny = std::numeric_limits<float>::denorm_min();
        row[50014] = -0.f;
        row[50015] = tiny;
        row[100029] = 0.f;
        row[150044] = -tiny;
        row.back() = std::numeric_limits<float>::min();
        const unsigned csr = _mm_getcsr();
        for (unsigned mode : {0u, (unsigned) _MM_DENORMALS_ZERO_MASK, (unsigned) _MM_ROUND_DOWN}) {
            _mm_setcsr((csr & ~(_MM_DENORMALS_ZERO_MASK | _MM_ROUND_MASK)) | mode);
            FourShards shards(row);
            for (auto logits : {LogitShards(row.data(), kVocab), shards.view}) {
                ref.stock_candidates(logits, kVocab, kVocab);
                assert(test.radix_candidates(logits, kVocab));
                equal_candidates(ref, test);
                for (int k : {1, 20, 64, 256, 16385, kVocab - 1}) {
                    ref.stock_candidates(logits, kVocab, k);
                    for (Pool* pool : {static_cast<Pool*>(nullptr), &four, &eight}) {
                        assert(test.top_k_candidates(logits, kVocab, k, pool));
                        equal_candidates(ref, test);
                    }
                }
            }
        }
        _mm_setcsr(csr);
        // NaNs must be detected in scalar and vector lanes, even outside the selected top-k.
        for (int i : {0, 256, 50014, 50015, 150043, kVocab - 1}) {
            const float old = row[i];
            row[i] = std::numeric_limits<float>::quiet_NaN();
            FourShards shards(row);
            for (auto logits : {LogitShards(row.data(), kVocab), shards.view}) {
                assert(!test.radix_candidates(logits, kVocab));
                assert(!test.min_p_candidates(logits, kVocab, 0.05f));
                for (Pool* pool : {static_cast<Pool*>(nullptr), &four, &eight})
                    assert(!test.top_k_candidates(logits, kVocab, 256, pool));
            }
            row[i] = old;
        }
        // A frozen threshold can let many tokens through; retry the suffix with the moving threshold.
        std::fill(row.begin() + 16384, row.end(), 4.f);
        const FourShards dense(row);
        for (auto logits : {LogitShards(row.data(), kVocab), dense.view}) {
            ref.stock_candidates(logits, kVocab, 64);
            for (Pool* pool : {&four, &eight}) {
                assert(test.top_k_candidates(logits, kVocab, 64, pool));
                equal_candidates(ref, test);
            }
        }
        row.back() = std::numeric_limits<float>::quiet_NaN();
        for (Pool* pool : {&four, &eight})
            assert(!test.top_k_candidates(LogitShards(row.data(), kVocab), kVocab, 64, pool));
        // Nonstandard min-p values can leave no survivor by comparison; the first candidate still stays.
        for (float logit : {-INFINITY, -0.f, INFINITY}) {
            std::fill(row.begin(), row.end(), logit);
            for (float min : {0.05f, 1.f, 2.f, INFINITY}) {
                Sampling s = sampling(0.8f);
                s.min_p = min;
                std::mt19937_64 a(17), b(17);
                LogitShards logits(row.data(), kVocab);
                const int id = stock_pick(ref, logits, s, {}, a);
                assert(test.min_p_candidates(logits, kVocab, min));
                assert(test.draw_filtered(s, b) == id && a == b);
                equal_candidates(ref, test);
            }
        }
        std::puts("Candidate edge checks: 192/192 (4/8-way prefilter, dense suffix, DAZ, NaN and infinite floors)");
    }

    template <class F>
    static void timed(const char* label, int iterations, F f) {
        std::vector<double> times;
        times.reserve(iterations);
        int64_t checksum = 0;
        for (int i = -16; i < iterations; ++i) {
            const auto start = Clock::now();
            checksum += f();
            if (i >= 0) times.push_back(ms_since(start) * 1000.0);
        }
        std::sort(times.begin(), times.end());
        std::printf("  %-11s median %9.2f us, p95 %9.2f us (checksum %lld)\n", label,
                    times[times.size() / 2], times[times.size() * 95 / 100 - 1], (long long) checksum);
    }

    static void filtered_bench() {
        assert(Sampler::filtered_sample_enabled());
        const auto row = normal_logits();
        const FourShards shards(row);
        for (auto logits : {LogitShards(row.data(), kVocab), shards.view}) {
            for (int k : {20, 64, 0, -1}) {
                Sampling s = sampling(0.8f);
                s.top_k = std::max(k, 0);
                s.top_p = k < 0 ? 1.f : 0.95f;
                s.min_p = k < 0 ? 0.05f : 0.f;
                std::printf("%s k=%d p=%.2f min_p=%.2f\n", logits.split() ? "sharded" : "contiguous",
                            s.top_k, s.top_p, s.min_p);
                Sampler stock, fast;
                std::mt19937_64 a(99), b(99);
                const int iterations = k > 0 ? 2000 : 150;
                timed("stock", iterations, [&] { return stock_pick(stock, logits, s, {}, a); });
                timed("fast", iterations, [&] { return fast.pick(logits, kVocab, s, {}, b); });
                assert(a == b);
            }
        }
    }

    static void prefilter_bench() {
        const auto row = normal_logits();
        const FourShards shards(row);
        for (auto logits : {LogitShards(row.data(), kVocab), shards.view}) {
            for (int k : {20, 64, 65, 128, 256, 1024}) {
                std::printf("%s k=%d (candidate prep)\n", logits.split() ? "sharded" : "contiguous", k);
                Sampler s;
                const auto run = [&](Pool* pool, const char* label) {
                    timed(label, 1500, [&] {
                        if (!s.top_k_candidates(logits, kVocab, k, pool)) std::abort();
                        return s.cand_.back().id;
                    });
                };
                run(nullptr, "1 thread");
                { Pool pool(4); run(&pool, "4 threads"); }
                { Pool pool(8); run(&pool, "8 threads"); }
            }
        }
    }

    static void bench(Pool* pool, int threads) {
        Sampler sampler;
        auto s = sampling(1.f);
        std::mt19937 rng_in(17);
        std::mt19937_64 rng_out(99);
        std::normal_distribution<float> d(0.f, 6.f);
        std::vector<float> logits(kVocab);
        for (float& l : logits) l = d(rng_in);
        double rows[kRows];
        int64_t ids = 0;
        for (int i = -16; i < kRows; ++i) {
            const auto t = Clock::now();
            ids += sampler.draw_all<true>(logits.data(), kVocab, s, rng_out, pool);
            if (i >= 0) rows[i] = ms_since(t) * 1000.0;
        }
        double sum = 0.0;
        for (double us : rows) sum += us;
        std::sort(rows, rows + kRows);
        std::printf("%d thread(s): median %.2f, mean %.2f, p95 %.2f us/row (checksum %lld)\n",
                    threads, (rows[kRows / 2 - 1] + rows[kRows / 2]) / 2, sum / kRows,
                    rows[kRows * 95 / 100 - 1], (long long) ids);
    }
};
}  // namespace

int main(int argc, char** argv) {
    if (!__builtin_cpu_supports("avx2")) return 77;
    if (argc > 1 && std::strcmp(argv[1], "--filtered-parity") == 0) {
        assert(argc == 3);
        SamplerHostTest::filtered_parity(std::strcmp(argv[2], "1") == 0);
        SamplerHostTest::candidate_edges();
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--filtered-bench") == 0) {
        SamplerHostTest::filtered_bench();
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--prefilter-bench") == 0) {
        SamplerHostTest::prefilter_bench();
        return 0;
    }
    // Also exercise the public pick() opt-in without enabling the old AVX2 flag separately.
    const int expected_threads = argc == 2 ? std::atoi(argv[1]) : 4;
    assert(SamplerHostTest::configured_threads() == expected_threads);
    {
        SamplerHostTest::Pool four(4), eight(8);
        SamplerHostTest::parity(four, eight);
    }
    SamplerHostTest::bench(nullptr, 1);
    {
        SamplerHostTest::Pool pool(4);
        SamplerHostTest::bench(&pool, 4);
    }
    {
        SamplerHostTest::Pool pool(8);
        SamplerHostTest::bench(&pool, 8);
    }
}
