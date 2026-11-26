// The DFlash2 drafter of Inkling-2-Mini (incoai/Inkling-2-Mini-DFlash2, a `dflash` GGUF with selector_hidden): five
// non-causal sliding-window layers with two-tap dynamic convolutions around each sublayer, fed by the target's rows
// at visits 5, 23, 41 and its final norm (concatenated, fc, hidden_norm) as a K/V prefix, a block [anchor, MASK x 7]
// in one pass, the target's head on the seven guessing rows, and a candidate selector that walks the top-16 lattice
// (z-lab/dflash model.py DFlash2DraftModel / CandidateSelector; llama.cpp renoirvieir4@0280b55a).  These are the
// launches the engine has no kernel for, including optional planar matrix products.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::poly::dflash2 {

constexpr int kWidth = 2304, kHeads = 32, kKv = 8, kHead = 128, kQuery = kHeads * kHead, kKey = kKv * kHead;
constexpr int kBlock = 8, kGuess = kBlock - 1, kWindow = 2048, kTop = 16, kRank = 256, kGroups = kWidth / 16;
constexpr int kDyn = 4 * kGroups;   // [stage][tap][group]
constexpr int kChunk = 64;          // attention keys a CTA
constexpr int kMaxChunks = (kWindow + kBlock + 32 - 1) / 32;   // attend3's chunks (32 keys) bound the others'
static_assert((kWindow & (kWindow - 1)) == 0, "ring slots are positions & (kWindow - 1)");

/// out[r] = [tap0[first + r], tap1[..], tap2[..], tap3[..]] (4 x kWidth floats a row) for r < n; tap rows kWidth apart.
struct Taps {
    const float* p[4];
};
void gather_taps(const Taps& taps, int first, int n, float* out, cudaStream_t st);

/// The two-tap grouped dynamic convolution of `stage` (0 prepare, 1 finish) over the block's rows:
/// y[r][c] = (base[s][0][c] + dyn[r][s][0][c/16]) x[r][c] + (r ? (base[s][1][c] + dyn[r][s][1][c/16]) x[r-1][c] : 0).
/// add: out += y (finish into the residual), else out = y.
void conv(const float* x, const float* dyn, const float* base, int stage, float* out, int rows, bool add,
          cudaStream_t st);

/// K/V rows (normed and rotated K) of positions pos .. pos + n - 1 into the half rings of kWindow rows ([slot][kKey]).
void store(const float* k, const float* v, __half* kc, __half* vc, int64_t pos, int n, cudaStream_t st);

/// Copy only accepted FP16 rows from [layers][kBlock][kKey] to [layers][kWindow][kKey].
void commit_prov_kv(const __half* k, const __half* v, __half* kc, __half* vc, int64_t pos, int n, int layers,
                    cudaStream_t st);

/// Non-causal attention of the block's queries at pos .. pos + 7 over the ring rows of positions lo .. pos - 1 (row i
/// sees those within kWindow - 1 of pos + i) and the block's own keys.  q [8][32][128], kb/vb [8][8][128]; out [8][4096].
size_t attend_floats();
void attend(const float* q, const float* kb, const float* vb, const __half* kc, const __half* vc, int64_t lo,
            int64_t pos, float* scratch, float* out, cudaStream_t st);

/// The top kTop of each of `rows` rows of logits (ld floats apart, the first `valid` count), by value then smaller id.
size_t topk_floats(int rows);
void topk(const float* logits, int rows, int64_t ld, int valid, float* scratch, int32_t* ids, float* vals,
          cudaStream_t st);

/// The two vocabulary halves' top kTop (ids already absolute): ids/vals a [rows][kTop] each -> the rows' top kTop.
void merge2(const int32_t* ids0, const float* vals0, const int32_t* ids1, const float* vals1, int rows, int32_t* ids,
            float* vals, cudaStream_t st);
/// [rows][2 * kTop]: kTop values, then kTop ids (+ offset) as float bits, for one gather.
void pack_top(const int32_t* ids, const float* vals, int rows, int offset, float* out, cudaStream_t st);
void unpack_top(const float* in, int rows, int32_t* ids, float* vals, cudaStream_t st);

/// The selector: edges S_t(a, b) = cap tanh((U_t(b) + sum_r P[a][r] H_t[r] Q[b][r]) / cap) from the anchor (t = 0) or
/// the candidates of position t - 1, then the greedy walk.  out[i] = the i-th guess and the softmax probability (over
/// its 16 candidates, temperature 1) of it.  pred/succ: [vocab][256]; h: [7][256]; edges: 7 x 256 floats.
struct Guess {
    int32_t token;
    float prob;
};
enum class BookKind { kF16, kQ8, kQ4 };
struct Book {
    const void* data = nullptr;
    BookKind kind = BookKind::kF16;
};
void select(const int32_t* ids, const float* vals, const float* h, Book pred, Book succ,
            int32_t anchor, float cap, float* edges, Guess* out, cudaStream_t st);

/// Planar GGML blocks: 32 int8 codes and one float scale (Q8), or 16 split-half nibble bytes and one half/float
/// scale (Q4). Each plane is row-major, with blocks of 32 weights; all code blocks are 16-byte aligned.
enum class MatKind { kQ8, kQ4 };
struct Mat {
    const uint8_t* q = nullptr;
    const void* d = nullptr;
    int n_in = 0, n_out = 0;
    MatKind kind = MatKind::kQ8;
    bool scale_f32 = true;

    explicit operator bool() const { return q != nullptr; }
    bool q4() const { return kind == MatKind::kQ4; }
    int code_bytes() const { return q4() ? 16 : 32; }
    int scale_bytes() const { return scale_f32 ? sizeof(float) : sizeof(__half); }
    Mat slice(int first, int rows) const;   // contiguous output-row view; no copy
};
void pack_q8_0(const void* src, int n_in, int n_out, uint8_t* q, float* d, cudaStream_t st);
void pack_q4_0(const void* src, int n_in, int n_out, uint8_t* q, void* d, bool scale_f32, cudaStream_t st);

/// Activations: int8 codes, a float scale, and (when xsum is nonnull) the integer sum of codes per 32 values.
void quantize(const float* x, int n_in, int rows, int8_t* q, float* d, int32_t* xsum, cudaStream_t st);
void gemv(const Mat& w, const int8_t* xq, const float* xd, const int32_t* xsum, float* y, int rows, cudaStream_t st);
/// Output rows [0, n0) go to y0, the next n1 to y1, and the next n2 to y2 ([rows][n_i] each).
void gemv_split(const Mat& w, const int8_t* xq, const float* xd, const int32_t* xsum, float* y0, int n0, float* y1,
                int n1, float* y2, int n2, int rows, cudaStream_t st);

/// Private vocabulary half, exactly kGuess activation rows. Fused IDs are local, like topk()'s.
size_t head_top_bytes(int n_out);
void head_top(const Mat& w, const int8_t* xq, const float* xd, const int32_t* xsum, void* scratch, int32_t* ids,
              float* vals, cudaStream_t st);

/// attend() with 32 keys a CTA, halves in LDS and dot2 (two CTAs a CU).
void attend3(const float* q, const float* kb, const float* vb, const __half* kc, const __half* vc, int64_t lo,
             int64_t pos, float* scratch, float* out, cudaStream_t st);

/// Fused drafter helpers (STRATA_DFLASH2_FAST=1): eliminate intermediate FP32 round-trips and quantize launches.
void rms_norm_quantize(const float* x, const float* w, float* y_out, int8_t* q, float* d, int32_t* xsum, int n_in,
                       int rows, float eps, cudaStream_t st);
void conv_quantize(const float* x, const float* dyn, const float* base, int stage, int rows, int8_t* q, float* d,
                   int32_t* xsum, cudaStream_t st);
void swiglu_quantize(const float* g, const float* u, int n_in, int rows, int8_t* q, float* d, int32_t* xsum,
                     cudaStream_t st);
}  // namespace strata::poly::dflash2
