// src/poly/tp2_comm.hpp - 2-GPU Tensor Parallel P2P collective primitives for gfx906 (MI50).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace strata::poly {

class Tp2Comm {
public:
    static constexpr int kSmallMax = 8 * 2304;  // T <= 8 uses single-kernel uncached P2P ping-pong

    static std::shared_ptr<Tp2Comm> create(int dev0, int dev1, int max_floats);
    ~Tp2Comm();

    Tp2Comm(const Tp2Comm&) = delete;
    Tp2Comm& operator=(const Tp2Comm&) = delete;

    int device(int rank) const { return dev_[rank]; }

    /// In-place sum across rank 0 and rank 1: on return, `data[0..n)` on both GPUs holds data_0 + data_1.
    /// For n <= kSmallMax, uses a single HIP-graph-safe uncached P2P ping-pong kernel (~7.5 us at T=1).
    /// For n > kSmallMax, uses FP16 hipMemcpyPeerAsync (~523 us at T=512) with host event handoff.
    void all_reduce(int rank, float* data, int n, void* stream);

    /// Broadcast `data[0..n)` from rank 0 to rank 1 using peer copy + event handoff.
    void broadcast_from_0(int rank, float* data, int n, void* stream);

    /// Simple reusable 2-thread host spin barrier between rank 0 and rank 1.
    void host_barrier(int rank);
    void abort_barrier();

private:
    Tp2Comm() = default;

    int dev_[2] = {0, 1};
    int max_floats_ = 0;

    // Uncached P2P ping-pong buffers & generation flags (for T <= 8, HIP-graph safe)
    float* p2p_buf_[2][2] = {};
    uint32_t* p2p_flag_[2] = {};
    uint32_t* p2p_gen_[2] = {};
    uint32_t* abort_dev_ = nullptr;   // host-mapped pinned flag visible to both GPUs

    // FP16 staging buffers & cross-device events for large prefill all-reduce (T > 8)
    uint16_t* dma_send_[2] = {};
    uint16_t* dma_recv_[2] = {};
    void* ev_bcast_ = nullptr;
    void* ev_bcast_done_ = nullptr;

    // Shared pointer handoff for broadcast_from_0
    float* bcast_src_ = nullptr;

    // 2-thread host barrier state
    std::atomic<uint64_t> bar_arrived_[2] = {0, 0};
    std::atomic<bool> bar_aborted_{false};
    uint64_t bar_epoch_[2] = {0, 0};
};

}  // namespace strata::poly
