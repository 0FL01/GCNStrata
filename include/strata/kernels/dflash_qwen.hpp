// include/strata/kernels/dflash_qwen.hpp - Qwen3.8-Flash-Next DFlash2 (incoai/Qwen3.8-Flash-Next-DFlash2) target taps.
//
// The drafter reads five target features per committed position: the residual after zero-based physical layers
// 5, 15, 25, 35 and 45 (GGUF dflash.target_layers = 6,16,26,36,46: the next layer's input boundary), each the
// unweighted mean of the four hyper-connection streams (capture patch: hidden.unflatten(-1, (hc, n)).mean(-2)).
// In the decode verifier the FFN write of layer l is folded into layer l+1's first HC read, so the full residual of
// layer l exists right after that read (R_out = R in place); bulk prefill materializes it after layer l's FFN write.
#pragma once

#include <cstdint>

namespace strata::kernels::dflash_qwen {

constexpr int kTaps = 5;
constexpr int kTapLayers[kTaps] = {5, 15, 25, 35, 45};

/// The tap index of zero-based layer l (its full residual), or -1.
inline int tap_of_layer(int64_t l) {
    for (int k = 0; k < kTaps; ++k)
        if (kTapLayers[k] == l) return k;
    return -1;
}

/// out[r * out_ld + c] = (R[r][0][c] + R[r][1][c] + R[r][2][c] + R[r][3][c]) * 0.25 (fixed order, no FMA), for r < rows,
/// c < n; R rows are hc * n floats apart.  hc must be 4.
void hc_stream_mean(const float* R, int rows, int hc, int n, float* out, int64_t out_ld, void* stream);

/// STRATA_DFLASH2_TAPS_DUMP=<file> (measurement only): append records {int64 pos, int32 tap, int32 n, float[n]} for
/// rows [0, rows) of `dev` ([rows][ld] floats on the current device, row r at position pos0 + r).  Synchronizes `stream`.
bool taps_dump_enabled();
void taps_dump_rows(const float* dev, int rows, int64_t ld, int n, int tap, int64_t pos0, void* stream);
/// The same from a hyper-connection residual (rows of hc * n floats): the stream mean first (scratch of its own).
void taps_dump_residual(const float* R, int rows, int hc, int n, int tap, int64_t pos0, void* stream);

}  // namespace strata::kernels::dflash_qwen
