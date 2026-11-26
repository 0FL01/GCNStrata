// Host-only parity test and warm-row benchmark. From the repository root:
/*
g++ -std=c++20 -O3 -mavx2 -pthread -Iinclude -ffunction-sections -fdata-sections \
    tools/poly_mt_sample_test.cpp src/spec/draft_policy.cpp -Wl,--gc-sections -o poly_mt_sample_test
STRATA_POLY_MT_SAMPLE=1 ./poly_mt_sample_test
*/
// Compile the actual sampler code; the linker drops the unused GPU runtime.
#define main strata_poly_unused_main
#include "../src/poly/main.cpp"
#undef main

#include <cassert>
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
