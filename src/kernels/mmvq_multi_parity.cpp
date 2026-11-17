// src/kernels/mmvq_multi_parity.cpp - the multi_exact contract of native_mmvq, tested bitwise.
//
//     build/mmvq_multi_parity
//
// WHY IT EXISTS.  `include/strata/kernels/native_mmvq.hpp` says of `native_mmvq_set_multi_exact`:
//
//     "true (default): the ncols == 1 layout, every column bitwise equal to a single-column call"
//
// and `include/strata/core/verify.hpp` counts "multi-column MMVQ in exact mode" among the reasons the verify window
// produces what greedy decode would, bit for bit.  Nothing tested that sentence: `iq_parity` calls `native_mmvq`
// with two columns and compares against a float64 reference at 2e-2, which cannot see a difference between one
// column and T columns.
//
// WHAT IT COMPARES.  The SAME Q8_1 bytes: one T-column call against T single-column calls, each reading column j
// from the same buffer (the header: "its Q8_1 blocks at j * n_in / 32").  The quantization is shared, so the only
// thing under test is how the columns are laid out.
//
// THE DATA.  The weights are random bytes, except each block's fp16 scales (`d`, and `dmin` where the format has
// one), which are rewritten as normal fp16 values in +-[2^-10, 2^-5): random bytes there would put inf and NaN in
// the outputs, and on the GPU every NaN comes out as the same canonical bit pattern, so a NaN output compares equal
// whatever operations produced it and proves nothing.  With finite scales and finite normal activations (the
// precondition the header puts on the input) every output is finite, and a non-finite one fails the test.
//
// THE NEGATIVE CONTROL.  The same comparison with `multi_exact` off must find differences, or the test has no power
// and its pass means nothing.  Where it has power depends on T: in `native_mmvq.cu` the generic layout uses
// `NW = NCOLS <= 4 ? 4 : 2` warps, and with `WARPS = 4` and `BPI = F::BPI * NW / WARPS` it coincides with the
// exact layout at T <= 4.  So the control is ASSERTED on T > 4 for every case, and at T <= 4 its silence is
// reported as that coincidence rather than skipped.  The control needs a row long enough for the two layouts to
// group blocks differently: IQ4_XS takes 16 blocks per iteration with 4 warps and 8 with 2, so at n_in = 2048 (8
// blocks of 256) each thread holds at most one block in both layouts and they coincide; its case uses n_in = 4096.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

struct Case {
    const char* name;
    int type;          // stable GGML type id, from the list in native_mmvq.hpp
    int n_in;
    int n_out;
    int block_elems;   // elements per weight block
    int block_bytes;
    int scale_at[2];   // byte offsets of the block's fp16 scales in native_mmvq.cu's block structs; -1 = none
};

// the types a head or a projection can have in this model
const Case CASES[] = {
    {"Q4_0", 2, 2048, 512, 32, 18, {0, -1}},          // Q40Block: half d
    {"Q8_0", 8, 2048, 512, 32, 34, {0, -1}},          // Q80Block: half d
    {"Q4_K", 12, 2048, 512, 256, 144, {0, 2}},        // Q4KBlock: half2 dm
    {"Q5_K", 13, 2048, 512, 256, 176, {0, 2}},        // Q5KBlock: half2 dm
    {"Q6_K", 14, 2048, 512, 256, 210, {208, -1}},     // Q6KBlock: half d after ql, qh, scales
    {"IQ4_XS", 23, 4096, 512, 256, 136, {0, -1}},     // IQ4XSBlock: half d; n_in 4096, see THE NEGATIVE CONTROL
    {"Q5_K wide", 13, 4096, 2048, 256, 176, {0, 2}},
};

// a normal fp16 in +-[2^-10, 2^-5): exponent field 5..9 (bias 15), any mantissa, either sign
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

// T = 4 is the last width where the two layouts coincide, T = 5 the first where they differ
const int WIDTHS[] = {1, 2, 3, 4, 5, 6, 8};
constexpr int COINCIDE_MAX_T = 4;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// Independent host FP16 decoding; unlike a multi-vs-single comparison this oracle does not use a GPU dot kernel.
double half_value(uint16_t h) {
    const int exp = (h >> 10) & 31, mant = h & 1023;
    if (exp == 31) throw std::runtime_error("non-finite half in oracle input");
    const double v = std::ldexp((double) (exp ? 1024 + mant : mant), exp ? exp - 25 : -24);
    return h & 0x8000 ? -v : v;
}

struct Q81Host { uint16_t d, sum; int8_t qs[32]; };
static_assert(sizeof(Q81Host) == 36);

bool small_oracle(int type, int T, cudaStream_t s) {
    constexpr int N = 2560, OUT = 64, NB = N / 32;
    const int bytes = type == 2 ? 18 : 22, bias = type == 2 ? 8 : 16;
    std::mt19937 rng(0x87654u + (unsigned) type * 13u + (unsigned) T);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<uint8_t> w((size_t) OUT * NB * bytes);
    for (auto& v : w) v = (uint8_t) rng();
    for (size_t b = 0; b < (size_t) OUT * NB; ++b) {
        uint16_t d = sane_half(rng); std::memcpy(w.data() + b * bytes, &d, sizeof d);
    }
    std::vector<float> x((size_t) T * N);
    for (auto& v : x) v = normal(rng);
    if (T > 1) std::fill(x.end() - N, x.end(), 0.f);
    const size_t xq_bytes = strata::kernels::native_q8_1_bytes(N, T);
    if (xq_bytes != (size_t) T * NB * sizeof(Q81Host)) return false;
    void *dw = nullptr, *dq = nullptr; float *dx = nullptr, *dy = nullptr;
    if (!ck(cudaMalloc(&dw, w.size()), "oracle weights") || !ck(cudaMalloc(&dq, xq_bytes), "oracle q8") ||
        !ck(cudaMalloc(&dx, x.size() * sizeof(float)), "oracle inputs") ||
        !ck(cudaMalloc(&dy, (size_t) T * OUT * sizeof(float)), "oracle outputs")) return false;
    if (!ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "oracle weight copy") ||
        !ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "oracle input copy")) return false;
    strata::kernels::quantize_q8_1_rows(dx, T, N, dq, s);
    strata::kernels::native_mmvq(type, dw, dq, dy, N, OUT, T, s);
    if (!ck(cudaStreamSynchronize(s), "oracle launch")) return false;
    std::vector<Q81Host> q8((size_t) T * NB);
    std::vector<float> got((size_t) T * OUT);
    if (!ck(cudaMemcpy(q8.data(), dq, xq_bytes, cudaMemcpyDeviceToHost), "oracle q8 readback") ||
        !ck(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "oracle result readback")) return false;
    double err = 0, refnorm = 0, neg = 0, ferr = 0, fnorm = 0, zero = 0;
    for (int t = 0; t < T; ++t) for (int r = 0; r < OUT; ++r) {
        double expected = 0, centered = 0, f64 = 0;
        for (int b = 0; b < NB; ++b) {
            const uint8_t* block = w.data() + ((size_t) r * NB + b) * bytes;
            uint16_t d; std::memcpy(&d, block, sizeof d);
            const double dw4 = half_value(d), da = half_value(q8[(size_t) t * NB + b].d);
            const double sum = half_value(q8[(size_t) t * NB + b].sum);
            int dot = 0, sumq = 0;
            for (int k = 0; k < 32; ++k) {
                const int ql = (block[(type == 2 ? 2 : 6) + k % 16] >> (k < 16 ? 0 : 4)) & 15;
                const int code = ql | (type == 6 ? (((block[2 + k / 8] >> (k % 8)) & 1) << 4) : 0);
                const int q = q8[(size_t) t * NB + b].qs[k];
                dot += code * q; sumq += q;
                f64 += dw4 * (code - bias) * x[(size_t) t * N + b * 32 + k];
            }
            expected += dw4 * (da * dot - bias * sum);
            centered += dw4 * da * (dot - bias * sumq);
        }
        const double y = got[(size_t) t * OUT + r];
        if (!std::isfinite(y)) return false;
        err += (y - expected) * (y - expected); refnorm += expected * expected;
        neg += (centered - expected) * (centered - expected);
        ferr += (y - f64) * (y - f64); fnorm += f64 * f64;
        if (T > 1 && t == T - 1) zero = std::max(zero, std::fabs(y));
    }
    cudaFree(dw); cudaFree(dq); cudaFree(dx); cudaFree(dy);
    const double same_rel = std::sqrt(err / (refnorm + 1e-30));
    const double neg_rel = std::sqrt(neg / (refnorm + 1e-30));
    const double float_rel = std::sqrt(ferr / (fnorm + 1e-30));
    std::printf("Q%d_0 T=%d same-code %.9g centered-control %.9g float64 %.9g zero %.9g\n",
                type == 2 ? 4 : 5, T, same_rel, neg_rel, float_rel, zero);
    return same_rel < 1e-4 && neg_rel > 1e-5 && float_rel < 0.04 && zero == 0;
}

// Returns the number of outputs that differ bit for bit; counts the non-finite outputs and the differences on
// outputs that are finite in both paths.
long long compare(const Case& c, int T, cudaStream_t s, long long& nonfinite, long long& diff_finite, bool& ran) {
    ran = false;
    if (!strata::kernels::native_mmvq_supported(c.type)) return -1;

    const std::size_t wbytes = strata::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    const std::size_t qcol = strata::kernels::native_q8_1_bytes(c.n_in, 1);   // Q8_1 bytes of ONE column
    const std::size_t qall = strata::kernels::native_q8_1_bytes(c.n_in, T);
    if (qall != qcol * (std::size_t) T) {
        std::printf("%-12s T=%d: the Q8_1 bytes of T columns (%zu) are not T times those of one (%zu): the test "
                    "assumes contiguous columns and they are not\n", c.name, T, qall, qcol);
        return -1;
    }

    std::mt19937 rng(1234u + (unsigned) c.type * 97u + (unsigned) T);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%-12s T=%d: %zu blocks of %d bytes are not the %zu weight bytes native_mmvq expects\n", c.name,
                    T, n_blocks, c.block_bytes, wbytes);
        return -1;
    }
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    std::vector<float> x((std::size_t) T * c.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);

    void* dw = nullptr;
    void* xq = nullptr;
    float* dx = nullptr;
    float* dmulti = nullptr;
    float* dsingle = nullptr;
    const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
    if (!ck(cudaMalloc(&dw, wbytes), "malloc w") || !ck(cudaMalloc(&xq, qall), "malloc xq") ||
        !ck(cudaMalloc(&dx, x.size() * 4), "malloc x") || !ck(cudaMalloc(&dmulti, ybytes), "malloc multi") ||
        !ck(cudaMalloc(&dsingle, ybytes), "malloc single"))
        return -1;
    if (!ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") ||
        !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
        return -1;

    // ONE quantization, shared by both paths: the quantization is not under test
    strata::kernels::quantize_q8_1_rows(dx, T, c.n_in, xq, s);
    if (!ck(cudaStreamSynchronize(s), "quantize")) return -1;

    try {
        strata::kernels::native_mmvq(c.type, dw, xq, dmulti, c.n_in, c.n_out, T, s);
        for (int j = 0; j < T; ++j)
            strata::kernels::native_mmvq(c.type, dw, (const uint8_t*) xq + (std::size_t) j * qcol,
                                         dsingle + (std::size_t) j * c.n_out, c.n_in, c.n_out, 1, s);
    } catch (const std::exception& e) {
        std::printf("%-12s T=%d: %s\n", c.name, T, e.what());
        return -1;
    }
    if (!ck(cudaStreamSynchronize(s), "mmvq")) return -1;

    std::vector<uint32_t> a((std::size_t) T * c.n_out), b(a.size());
    if (!ck(cudaMemcpy(a.data(), dmulti, ybytes, cudaMemcpyDeviceToHost), "read multi") ||
        !ck(cudaMemcpy(b.data(), dsingle, ybytes, cudaMemcpyDeviceToHost), "read single"))
        return -1;

    long long diff = 0;
    nonfinite = 0;
    diff_finite = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float fa, fb;
        std::memcpy(&fa, &a[i], 4);
        std::memcpy(&fb, &b[i], 4);
        const bool both_finite = std::isfinite(fa) && std::isfinite(fb);
        if (!both_finite) ++nonfinite;
        if (a[i] != b[i]) {
            ++diff;
            if (both_finite) ++diff_finite;
        }
    }
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(dmulti); cudaFree(dsingle);
    ran = true;
    return diff;
}

struct Totals {
    long long checked = 0, diff = 0, diff_finite = 0, nonfinite = 0;   // all widths
    long long diff_finite_wide = 0;                                     // T > COINCIDE_MAX_T only
    int powerless_cases = 0;   // cases where the control found no difference at any T > COINCIDE_MAX_T
};

int sweep(bool exact, cudaStream_t s, Totals& t) {
    strata::kernels::native_mmvq_set_multi_exact(exact);
    if (strata::kernels::native_mmvq_multi_exact() != exact) {
        std::printf("native_mmvq_multi_exact() does not reflect what was set\n");
        return 1;
    }
    std::printf("\n=== multi_exact %s ===\n", exact ? "ON (the default)" : "off (negative control)");
    std::printf("%-12s %3s %10s %10s %14s %10s\n", "type", "T", "outputs", "bits diff", "finite diff", "nonfinite");
    int bad = 0;
    for (const Case& c : CASES) {
        long long case_wide = 0;
        for (int T : WIDTHS) {
            long long nf = 0, dfin = 0;
            bool ran = false;
            const long long diff = compare(c, T, s, nf, dfin, ran);
            if (!ran) { ++bad; continue; }
            const long long n = (long long) T * c.n_out;
            t.checked += n;
            t.diff += diff;
            t.diff_finite += dfin;
            t.nonfinite += nf;
            if (T > COINCIDE_MAX_T) { t.diff_finite_wide += dfin; case_wide += dfin; }
            const char* verdict;
            if (exact) verdict = diff == 0 ? "ok" : "FAIL";
            else if (dfin > 0) verdict = "control sees it";
            else verdict = T <= COINCIDE_MAX_T ? "layouts coincide here (NW = 4)" : "no power at this T";
            std::printf("%-12s %3d %10lld %10lld %14lld %10lld  %s\n", c.name, T, n, diff, dfin, nf, verdict);
            if (exact && diff != 0) ++bad;
        }
        if (!exact && case_wide == 0) ++t.powerless_cases;
    }
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream create")) return 1;

    if (argc == 2 && std::strcmp(argv[1], "--q4-q5-oracle") == 0) {
        bool ok = true;
        for (bool exact : {true, false}) {
            strata::kernels::native_mmvq_set_multi_exact(exact);
            std::printf("small-format oracle multi_exact=%d\n", (int) exact);
            for (int type : {2, 6}) for (int T : {1, 3, 6}) ok = small_oracle(type, T, s) && ok;
        }
        strata::kernels::native_mmvq_set_multi_exact(true);
        cudaStreamDestroy(s);
        return ok ? 0 : 1;
    }
    if (argc != 1) { std::printf("expected --q4-q5-oracle or no arguments\n"); cudaStreamDestroy(s); return 1; }

    Totals on, off;
    const int bad = sweep(true, s, on);
    const int bad_off = sweep(false, s, off);
    strata::kernels::native_mmvq_set_multi_exact(true);   // back to the default

    std::printf("\nmulti_exact on:   %lld outputs compared, %lld differ bitwise, %lld non-finite\n",
                on.checked, on.diff, on.nonfinite);
    std::printf("multi_exact off:  %lld outputs compared, %lld differ bitwise (%lld on finite outputs, %lld at T > %d)\n",
                off.checked, off.diff, off.diff_finite, off.diff_finite_wide, COINCIDE_MAX_T);

    if (on.nonfinite != 0 || off.nonfinite != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %lld non-finite outputs from finite scales and activations: the "
                    "bit comparison is only meaningful on finite outputs.\n", on.nonfinite + off.nonfinite);
        return 1;
    }
    if (bad != 0 || bad_off != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d case(s) with multi_exact on differ from single-column calls "
                    "or did not run, %d case(s) did not run with it off. If they differ, the sentence in "
                    "native_mmvq.hpp does not hold, and neither does the bit-exactness verify.hpp claims.\n",
                    bad, bad_off);
        return 1;
    }
    // THE NEGATIVE CONTROL, asserted where it can see something: past T = 4 the generic layout is a different one,
    // so if it still finds no difference the comparison cannot tell the layouts apart and the pass proves nothing.
#if defined(STRATA_HIP_GFX906)
    // gfx906: the wave64 layout (one wavefront per row) is the same kernel for every column count, so multi_exact
    // has nothing to switch and the control cannot see a difference.  The control with power is the CUDA layout:
    // ctest's mmvq_multi_parity_cuda_layout runs this with STRATA_MMVQ_WAVE=0.
    if (off.powerless_cases != 0 && !(std::getenv("STRATA_MMVQ_WAVE") && std::string(std::getenv("STRATA_MMVQ_WAVE")) == "0")) {
        std::printf("\nmmvq_multi_parity: ok (gfx906 wave64 layout: every column count runs the same exact kernel, "
                    "so the negative control has no other layout to find; mmvq_multi_parity_cuda_layout checks it).\n");
        return 0;
    }
#endif
    if (off.powerless_cases != 0) {
        std::printf("\nmmvq_multi_parity: FAILED (no power). In %d case(s) no output differs with multi_exact off at "
                    "any T > %d, where the generic layout is not the exact one: for those cases this comparison "
                    "cannot tell the two layouts apart, so their pass above proves nothing.\n", off.powerless_cases,
                    COINCIDE_MAX_T);
        return 1;
    }
    std::printf("\nmmvq_multi_parity: ok. With multi_exact on every column is bitwise equal to a single-column "
                "call; with it off the control finds differences at T > %d in every case, so the comparison has "
                "power.\n",
                COINCIDE_MAX_T);
    return 0;
}
