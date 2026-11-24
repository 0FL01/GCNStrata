#pragma once

namespace strata::poly::inkling {
namespace {

#if defined(__gfx906__)
// All exchanges stay inside each of the eight 8-lane groups in a wave64.
template <int X> __device__ __forceinline__ float tp2_lean_xor(float v) {
    static_assert(X == 1 || X == 2 || X == 4);
    const auto bits = __builtin_bit_cast(int, v);
    if constexpr (X == 1)
        return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(bits, 0xb1, 15, 15, false));
    else if constexpr (X == 2)
        return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(bits, 0x4e, 15, 15, false));
    else
        return __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(bits, (X << 10) | 31));
}

template <int O> __device__ __forceinline__ float tp2_lean_owner(float v) {
    static_assert(O >= 0 && O < 8);
    return __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 24 | (O << 5)));
}

__device__ __forceinline__ float tp2_lean_broadcast(float v, int owner) {
    switch (owner) {
        case 0: return tp2_lean_owner<0>(v);
        case 1: return tp2_lean_owner<1>(v);
        case 2: return tp2_lean_owner<2>(v);
        case 3: return tp2_lean_owner<3>(v);
        case 4: return tp2_lean_owner<4>(v);
        case 5: return tp2_lean_owner<5>(v);
        case 6: return tp2_lean_owner<6>(v);
        default: return tp2_lean_owner<7>(v);
    }
}
#endif

}  // namespace
}  // namespace strata::poly::inkling
