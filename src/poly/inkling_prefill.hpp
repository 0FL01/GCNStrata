// Scratch and launch contract for the fully resident Inkling TP2 prompt path.
#pragma once

#include <cstdint>

namespace strata::poly::inkling_prefill {

constexpr int experts = 128, picks = 8, max_tokens = 2048, classes = 3;
constexpr int width = 2304, ff = 576;
constexpr int slice = 1024, waves = slice / 64, max_parts = max_tokens * picks / slice;
constexpr int cutoffs[classes - 1] = {32, 128};

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct SlotOrder { int32_t expert[experts]; };  // inverse physical slot permutation, passed by value
struct Summary {
    int32_t invalid;   // cleared by the caller; nonzero on selector or partition failure
    int32_t count[experts];   // by physical slot, for the existing use/profile counters
    int32_t most[classes];    // stock first-best J and stock fallback, not compact grid sizing
};
static_assert(sizeof(Summary) == 528);
struct Scratch {
    Summary summary;
    uint64_t mask[experts][max_parts][waves];
    int32_t count[experts][max_parts], first[experts][max_parts];
    int32_t begin[experts];

};

// Invoked on one stream, with the exact selector and its status preceding this work.
// The plan writes stock class bounds only; scatter builds stock gather/inverse and place.
void plan(Scratch* s, const int32_t* ids, int tokens, SlotOrder order, int32_t* cb, void* stream);
void scatter(const Scratch* s, int tokens, bool inverse, int32_t* tok, int32_t* place, void* stream);

}  // namespace strata::poly::inkling_prefill
