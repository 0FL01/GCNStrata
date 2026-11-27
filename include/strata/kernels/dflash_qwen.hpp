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

/// STRATA_DFLASH2=<draft gguf> (serve, opt-in): the committed rows' taps go to a pinned host ring that the native
/// drafter's context is fed from.  The producers are the bulk prompt path (after a tap layer's FFN write) and every
/// verifier stage's commit (its own taps, the kept rows only).  A row is complete when all five taps were written
/// for the same position.  Only rows at or after the sequence's `lo` are kept (the drafter sees 2048 rows).
constexpr int kSinkRows = 2560;
bool sink_enabled();
void sink_begin(int64_t lo);   ///< a new sequence: every row is invalidated, rows < lo are not kept
/// rows [0, rows) of `dev` (row r at position pos0 + r, ld floats apart, n of them): copied, `stream` synchronized
void sink_rows(const float* dev, int rows, int64_t ld, int n, int tap, int64_t pos0, void* stream);
void sink_residual(const float* R, int rows, int hc, int n, int tap, int64_t pos0, void* stream);
/// the smallest s in [a, b] such that every row of [s, b) is complete (b: none is)
int64_t sink_complete_from(int64_t a, int64_t b);
/// a complete row's [5][n] floats (host, pinned); rows of consecutive positions are adjacent until the ring wraps
const float* sink_row(int64_t pos);
int64_t sink_wrap_rows(int64_t pos);   ///< rows from pos to the ring's end
/// the verifier's split of sink_rows: the window's rows are copied as soon as its stage is done (enqueued on `stream`,
/// no wait; positions not yet committed are not complete), and the kept ones are marked after that copy has landed
void sink_stage_rows(const float* dev, int rows, int64_t ld, int n, int tap, int64_t pos0, void* stream);
void sink_tag_rows(int tap, int64_t pos0, int rows);

}  // namespace strata::kernels::dflash_qwen
