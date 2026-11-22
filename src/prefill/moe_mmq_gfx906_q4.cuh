// Included in moe_mmq.cu's private namespace only for the gfx906 HIP build.
// The release's rectangular non-stream-k MoE grid, with an explicit per-expert weight address.
struct DownTabs { WeightTab body, last; };

template<ggml_type T, int J, typename Tabs>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(T, J, false), ggml_cuda_mmq_get_occupancy(T, J, false))
__global__ void gfx906_q4_tab(Tabs tab, const int* xq, const int32_t* bounds, const int32_t* ids,
                              float* dst, int bpr, int total, int rows, int ld) {
    constexpr int I = ggml_cuda_mmq_get_I(T, J, false);
    const int z = blockIdx.z, i = blockIdx.x * I, start = bounds[z] + (int) blockIdx.y * J;
    const int left = bounds[z + 1] - start;
    if (left <= 0) return;
    extern __shared__ int out_ids[];
    const int lane = threadIdx.y * 64 + threadIdx.x;
    if (lane < J) out_ids[lane] = lane < left ? ids[start + lane] : 0;
    __syncthreads();
    int offset = i * bpr;
    const void* weight;
    if constexpr (std::is_same_v<Tabs, DownTabs>) {
        const bool last = blockIdx.x + 1 == gridDim.x;
        weight = last ? tab.last.p[z] : tab.body.p[z];
        if (last) offset = 0;
    } else {
        weight = tab.p[z]; // original GU: no final-tile branch or second table
    }
    mul_mat_q_process_tile<T, J, false, false>((const char*) weight, offset,
        xq + start * (sizeof(block_q8_1_mmq) / sizeof(int)), out_ids, dst + i, nullptr, nullptr,
        bpr, total, ld, rows - i - 1, left - 1, 0, bpr);
}

// Check every selectable tile before the plan permits skipping gather. An unfamiliar device,
// shape or scheduler keeps the original contiguous path for the entire layer.
bool tab_has_type(int t) {
    if (t != GGML_TYPE_Q4_0 && t != GGML_TYPE_Q4_1) return false;
    const auto& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const auto& d = info.devices[id];
        if (d.cc != GGML_CUDA_CC_VEGA20 || d.warp_size != 64) return false;
        for (int j = 8; j <= 128; j += 8) {
            const auto c = ggml_cuda_mmq_get_config((ggml_type) t, j, false, d.cc);
            if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, d.cc) > d.smpbo) continue;
            if (j > 64 || j == 56 || c.J != j || c.I != 64 || c.K_vram != 256 ||
                c.nthreads != 256 || c.stream_k) return false;
        }
    }
    return info.device_count > 0;
}

template<ggml_type T, typename Tabs>
void run_q4_tab(const mmq_args& a, const Tabs& tab, cudaStream_t s) {
    const auto& d = ggml_cuda_info().devices[ggml_cuda_get_device()];
    // First minimum in ascending J, as in mul_mat_q_switch_J. The hint never limits the grid.
    int best = 0, ntiles = INT_MAX;
    size_t shared = 0;
    for (int j = 8; j <= 64 && ntiles > 1; j += 8) {
        const auto c = ggml_cuda_mmq_get_config(T, j, false, d.cc);
        if (c.type == GGML_TYPE_COUNT) continue;
        const size_t bytes = mmq_get_nbytes_shared(c, d.cc);
        const int n = (a.ncols_opt + j - 1) / j;
        if (bytes <= d.smpbo && n < ntiles) { best = j; ntiles = n; shared = bytes; }
    }
    GGML_ASSERT(best > 0);
    const dim3 grid(a.nrows_x / 64, (a.ncols_max + best - 1) / best, a.nchannels_y);
    const dim3 threads(64, 4);
    auto launch = [&]<int J>() {
        gfx906_q4_tab<T, J><<<grid, threads, shared, s>>>(tab, a.y, a.expert_bounds, a.ids_dst, a.dst,
            a.stride_row_x, a.ncols_y, a.nrows_x, a.nrows_dst);
    };
#define Q4_TAB(J) case J: launch.template operator()<J>(); break
    switch (best) {
        Q4_TAB(8); Q4_TAB(16); Q4_TAB(24); Q4_TAB(32); Q4_TAB(40); Q4_TAB(48); Q4_TAB(64);
        default: GGML_ABORT("unexpected gfx906 Q4 pointer-table tile");
    }
#undef Q4_TAB
}
