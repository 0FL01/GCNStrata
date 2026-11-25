#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::poly::inkling {

// All target Flash readers tile absolute keys in groups of 16.
inline constexpr int kSwaKeyTile = 16;
inline constexpr std::size_t kSwaCopyAlign = 16;

struct SwaPlan {
    int ring_rows, stage_rows = 0;
    std::size_t saved_bytes = 0;
};

// From PolyStrata 4ec4bb35, but only borrow existing, aligned rank-local scratch. If it cannot
// hold the whole stage, keep the stock rings rather than reserving another allocation.
constexpr SwaPlan plan_swa(int context, int window, int rows, int verify, int visits, std::size_t row_bytes,
                           std::size_t input_bytes, bool enabled) {
    const int stock = std::min(context, window - 1 + rows);
    if (!enabled) return {stock};
    // keep() can accept only a prefix of the provisionally written verifier rows.
    const int ring = std::min(context, window - 1 + verify);
    if (ring >= stock) return {stock};
    const int stage = stock + kSwaKeyTile - 1;
    if (std::size_t(stage) * row_bytes > input_bytes) return {stock};
    return {ring, stage, std::size_t(visits) * (stock - ring) * row_bytes};
}

struct SwaStageLayout {
    int64_t first = 0;
    int history = 0, head = 0;
    constexpr int append() const { return head + history; }
    constexpr bool fits_ring(int n, int ring) const { return history + n <= ring; }
};

constexpr SwaStageLayout swa_stage_layout(int64_t pos, int window) {
    const int history = (int) std::min<int64_t>(pos, window - 1);
    const int64_t first = pos - history;
    return {first, history, (int) (first % kSwaKeyTile)};
}

struct SwaRingSpan { int row, count; };

constexpr SwaRingSpan swa_ring_span(int64_t first, int64_t end, int ring) {
    const int row = (int) (first % ring);
    return {row, (int) std::min<int64_t>(end - first, ring - row)};
}

}  // namespace strata::poly::inkling
