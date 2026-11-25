// Scratch and launch contract for the fully resident Inkling TP2 prompt path.
#pragma once

#include <cstdint>

namespace strata::poly::inkling_prefill {

constexpr int experts = 128, picks = 8, max_tokens = 2048, classes = 3;
constexpr int width = 2304, ff = 576;
constexpr int slice = 1024, waves = slice / 64, max_parts = max_tokens * picks / slice;
constexpr int tiles[classes] = {32, 64, 64};
constexpr int cutoffs[classes - 1] = {32, 128};

constexpr int min(int a, int b) { return a < b ? a : b; }
constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// For m active experts of this class, sum ceil(rows[e]/J) is at most
// floor((tokens*picks + m*(J-1))/J) and at most m*ceil(largest_possible_rows/J).
// One token never picks the same expert twice. No count readback for the compact grid.
constexpr int task_cap(int tokens, int c) {
    const int lo = c == 0 ? 1 : cutoffs[c - 1] + 1;
    const int hi = c == classes - 1 ? tokens : min(tokens, cutoffs[c]);
    if (hi < lo) return 0;
    const int rows = tokens * picks, m = min(experts, rows / lo), j = tiles[c];
    return min(m * ceil_div(hi, j), (rows + m * (j - 1)) / j);
}

struct SlotOrder { int32_t expert[experts]; };  // inverse physical slot permutation, passed by value
struct Summary {
    int32_t invalid;   // cleared by the caller; nonzero on selector or partition failure
    int32_t count[experts];   // by physical slot, for the existing use/profile counters
    int32_t most[classes];    // stock first-best J and stock fallback, not compact grid sizing
};
static_assert(sizeof(Summary) == 528);
struct Task { int32_t slot, col, remaining; };
struct Scratch {
    Summary summary;
    uint64_t mask[experts][max_parts][waves];
    int32_t count[experts][max_parts], first[experts][max_parts];
    int32_t begin[experts], used[classes];
    Task list[task_cap(max_tokens, 0) + task_cap(max_tokens, 1) + task_cap(max_tokens, 2)];

    static constexpr int list_offset(int c) {
        return c == 0 ? 0 : task_cap(max_tokens, 0) + (c == 1 ? 0 : task_cap(max_tokens, 1));
    }
};

// Invoked on one stream, with the exact selector and its status preceding this work.
// The plan writes stock class bounds and compact task lists; scatter builds stock gather/inverse and place.
void plan(Scratch* s, const int32_t* ids, int tokens, SlotOrder order, int32_t* cb, bool lists, void* stream);
void scatter(const Scratch* s, int tokens, bool inverse, int32_t* tok, int32_t* place, void* stream);

}  // namespace strata::poly::inkling_prefill
