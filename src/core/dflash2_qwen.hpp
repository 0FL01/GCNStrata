// Standalone Qwen3.8-Flash-Next DFlash2 forward for gfx906 (CUDA-source HIP).
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::core::dflash2_qwen {

inline constexpr int kWidth = 2560, kTaps = 5, kTapWidth = kWidth * kTaps;
inline constexpr int kBlock = 8, kWindow = 2048, kTop = 16, kVocab = 248320, kMask = 248077;

struct Result {
    int steps = 0;
    std::array<int32_t, kBlock - 1> tokens{};
    std::array<float, kBlock - 1> prob{};  // realized q of the selected token (1 in greedy mode)
    std::array<int32_t, (kBlock - 1) * kTop> top_ids{};
    std::array<float, (kBlock - 1) * kTop> unary{};  // raw logits, no softcap
};

struct Timing {
    float layers_ms = 0;    // includes embeddings and final RMSNorm
    float head_ms = 0;      // vocabulary projection + top-16
    float selector_ms = 0;  // projection, edges, walk; excludes host readback
    float total_ms = 0;     // GPU events, from embeddings through walk
};

// One request/cache per instance, not thread-safe. Owns a nonblocking stream and all scratch.
// Model files are read-only; only the target embedding and output tensors are uploaded.
class Drafter {
public:
    Drafter(const std::string& draft_gguf, const std::string& target_shard, int device = 0);
    ~Drafter();
    Drafter(const Drafter&) = delete;
    Drafter& operator=(const Drafter&) = delete;

    cudaStream_t stream() const;
    void reset();

    // Taps are contiguous device float32 [rows,5,2560] after each *committed* token,
    // at absolute positions first_pos .. first_pos+rows-1. The first append after
    // construction/reset may start at any nonnegative position; later appends must
    // abut the preceding ones. Enqueues work on stream(), does not synchronize.
    // The caller orders writes/reads and keeps the input alive until consumed.
    void context(const float* device_taps, int64_t first_pos, int rows);

    // Positions p .. p+B-1 belong to [anchor, mask, ...]. Requires p at the end
    // of the committed prefix (or an empty prefix). Does not mutate the cache.
    // Synchronous result; optional host_hidden has room for B*kWidth floats.
    Result ask(int32_t anchor_id, int64_t p, int B, Timing* timing = nullptr, float* host_hidden = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::core::dflash2_qwen
