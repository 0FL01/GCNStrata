// SPDX-License-Identifier: MIT
// Adapted from the N=32 scalar math/{expf.c,exp2f_data.c} in Arm Optimized Routines,
// https://github.com/ARM-software/optimized-routines/tree/20859d3fd9e797dde592c489c8ef7d383f778620/math
// Copyright (c) 2017-2026, Arm Limited.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <cstdint>

namespace strata::poly::inkling_route {

// Permit a host comparison with the actual std::exp(float) as well as device use.
#if defined(__HIPCC__)
#define INKLING_EXPF_HD __host__ __device__
#else
#define INKLING_EXPF_HD
#endif

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
static __device__ __constant__ const uint64_t exp2_table[32] = {
#else
inline constexpr uint64_t exp2_table[32] = {
#endif
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL, 0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL, 0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL, 0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};

inline constexpr float kExpMin = -0x1.5d589ep+6f, kExpMax = 0x1.62e42ep+6f;

// Normal-result expf, round-to-nearest/even. The caller handles exceptional inputs.
// The host x86 FMA variant fuses range reduction too: do not reuse a rounded x*invln2.
INKLING_EXPF_HD static inline float expf_normal(float x) {
    constexpr double shift = 0x1.8p+52, invln2 = 0x1.71547652b82fep+0 * 32;
    constexpr double c0 = 0x1.c6af84b912394p-5 / 32 / 32 / 32;
    constexpr double c1 = 0x1.ebfce50fac4f3p-3 / 32 / 32;
    constexpr double c2 = 0x1.62e42ff0c52d6p-1 / 32;
    const double encoded = __builtin_fma(invln2, (double) x, shift);
    const double r = __builtin_fma(invln2, (double) x, -(encoded - shift));
    const auto ki = __builtin_bit_cast(uint64_t, encoded);
    const double s = __builtin_bit_cast(double, exp2_table[ki & 31u] + (ki << 47));
    const double z = __builtin_fma(c0, r, c1), y = __builtin_fma(c2, r, 1.0);
    return (float) (__builtin_fma(z, r * r, y) * s);
}

#undef INKLING_EXPF_HD
}  // namespace strata::poly::inkling_route
