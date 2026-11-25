// Included by poly/mmq.cu, after the pinned ggml MMQ headers.
#pragma once

#include "inkling_prefill.hpp"

namespace strata::poly::inkling_prefill {
namespace {

constexpr int threads = 256, wave_size = 64, cta_waves = threads / wave_size;

__global__ void mask_picks(Scratch* s, const int32_t* ids, int r, SlotOrder order) {
    if (s->summary.invalid) return;
    const int e = blockIdx.x, part = blockIdx.y, lane = threadIdx.x % wave_size, wave = threadIdx.x / wave_size;
    const int expert = order.expert[e];
    __shared__ int totals[cta_waves];
    int n = 0;
    for (int w = wave; w < waves; w += cta_waves) {
        const int i = part * slice + w * wave_size + lane;
        const uint64_t bits = __ballot(i < r && ids[i] == expert);
        if (lane == 0) s->mask[e][part][w] = bits;
        n += __popcll(bits);
    }
    if (lane == 0) totals[wave] = n;
    __syncthreads();
    if (threadIdx.x == 0) s->count[e][part] = totals[0] + totals[1] + totals[2] + totals[3];
}

__global__ void make_plan(Scratch* s, int32_t* cb, int parts, int tokens) {
    if (s->summary.invalid) return;
    const int e = threadIdx.x;
    int n = 0;
    for (int p = 0; p < parts; ++p) {
        s->first[e][p] = n;
        n += s->count[e][p];
    }
    s->summary.count[e] = n;
    __shared__ int counts[experts];
    counts[e] = n;
    __syncthreads();
    if (e == 0) {
        int total = 0;
        for (int v = 0; v < experts; ++v) {
            total += counts[v];
            if (counts[v] > tokens) s->summary.invalid = 1;
        }
        if (total != tokens * picks) s->summary.invalid = 1;
        if (!s->summary.invalid) {
            int row = 0;
            for (int c = 0; c < classes; ++c) {
                int most = 0;
                int32_t* b = cb + c * (experts + 1);
                for (int v = 0; v < experts; ++v) {
                    b[v] = row;
                    const int nn = counts[v];
                    const int cls = nn <= cutoffs[0] ? 0 : nn <= cutoffs[1] ? 1 : 2;
                    if (nn == 0 || c != cls) continue;
                    s->begin[v] = row;
                    row += nn;
                    most = most < nn ? nn : most;
                }
                b[experts] = row;
                s->summary.most[c] = most;
            }
        }
    }

}

__global__ void scatter_picks(const Scratch* s, bool inverse, int32_t* tok, int32_t* place) {
    const int e = blockIdx.x, part = blockIdx.y, lane = threadIdx.x % wave_size, wave = threadIdx.x / wave_size;
    if (s->summary.invalid || !s->count[e][part]) return;
    __shared__ uint64_t masks[waves];
    if (threadIdx.x < waves) masks[threadIdx.x] = s->mask[e][part][threadIdx.x];
    __syncthreads();
    __shared__ int prior[waves];
    if (threadIdx.x < waves) {
        int n = 0;
        for (int w = 0; w < threadIdx.x; ++w) n += __popcll(masks[w]);
        prior[threadIdx.x] = n;
    }
    __syncthreads();
    const int begin = s->begin[e] + s->first[e][part];
    const uint64_t bit = uint64_t(1) << lane;
    for (int w = wave; w < waves; w += cta_waves) {
        const uint64_t mask = masks[w];
        if (mask & bit) {
            const int i = part * slice + w * wave_size + lane;
            const int row = begin + prior[w] + __popcll(mask & (bit - 1));
            if (inverse) tok[i] = row;
            else tok[row] = i / picks;
            place[row] = i;
        }
    }
}

}  // namespace
}  // namespace strata::poly::inkling_prefill
