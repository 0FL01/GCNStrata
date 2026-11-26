// Internal gfx906 TP2 attention extension. Canonical K/V and the forward interface stay on the owners.
#pragma once

#if defined(STRATA_HIP_GFX906)
#include "inkling_attend.hpp"

#include <array>
#include <memory>

namespace strata::poly::inkling::kv_helpers {

inline constexpr int kFullVisits = 8, kOwnerKvHeads = 4, kHalfKvHeads = 2;
inline constexpr int kDefaultDecodeRows = 4, kMaxDecodeRows = 8;
inline constexpr int kHeadWidth = 128, kQueriesPerKv = 3, kRelativeWidth = 16, kTableRows = 1024;
inline constexpr int kQ8BlockValues = 32, kQ81BlockBytes = 36;

struct Mirror;
struct FreeMirror { void operator()(Mirror*) const noexcept; };
using MirrorPtr = std::unique_ptr<Mirror, FreeMirror>;

// Either both links are usable and both mirrors are allocated, or setup throws before either rank uses one.
std::array<MirrorPtr, 2> create_pair(int capacity, std::array<cudaStream_t, 2> owner_streams,
                                    bool wide_decode);
// Optional disposable packet per helper. No replica validity or checkpoint semantics for prompt rows.
void enable_flash_rows(std::array<MirrorPtr, 2>& mirrors);
bool try_flash_rows(Mirror& mirror, const Attend& a);
// Persistent-mirror head-split Flash attention (0-ULP exact, T in [64..2048], pos >= 8192).
void enable_flash_mirror(std::array<MirrorPtr, 2>& mirrors);
bool try_flash_mirror(Mirror& mirror, const Attend& a, int slot);
void seed_slot(Mirror& mirror, int slot, KvView k, KvView v, const float* table, int src_device, int end_row);
void sync_seed(Mirror& mirror);
void invalidate(Mirror& mirror, int64_t first);
void set_budget(Mirror& mirror, int64_t prompt_rows, int64_t max_new);

enum class Visit { Local, Seeding, Offloaded };
// Called after the canonical stores, even on full visits that will stay local (they can overwrite provisional rows).
Visit try_visit(Mirror& mirror, const Attend& a, int slot, float* partial, float* scores, size_t score_floats,
                void* q8);
// The caller must join Seeding after enqueuing ordinary attention, before reusing or overwriting its inputs.
void join_seed(Mirror& mirror);

// Shared factored kernels and stock combine: two groups, stride 4 in the owner's buffers or 2 in a compact mirror.
size_t half_decode_floats(int rows);
// score_rows=0 means the score slab fits every row. A smaller slab is reused in order on the same stream.
void attend_decode_half(const Attend& a, float* partial, float* scores, void* q8, cudaStream_t stream,
                        int score_rows = 0);

}  // namespace strata::poly::inkling::kv_helpers
#endif
