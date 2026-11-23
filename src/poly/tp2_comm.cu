// src/poly/tp2_comm.cu - 2-GPU Tensor Parallel P2P collective primitives.
#include "tp2_comm.hpp"
#include "runtime.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef STRATA_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

struct DeviceGuard {
    int prev = 0;
    bool ok = false;
    DeviceGuard() { ok = (cudaGetDevice(&prev) == cudaSuccess); }
    ~DeviceGuard() {
        if (ok) { (void) cudaSetDevice(prev); }
    }
};

// Single-block ping-pong P2P all-reduce for T <= 8 (n <= 18432 floats, n % 4 == 0).
// Uses uncached VRAM buffers & generation flags; 100% HIP-graph capture/replay safe.
__global__ void p2p_ar_pingpong_f32(float* __restrict__ data,
                                    float* __restrict__ self_b0, float* __restrict__ self_b1,
                                    const float* __restrict__ peer_b0, const float* __restrict__ peer_b1,
                                    volatile uint32_t* __restrict__ self_flag,
                                    volatile uint32_t* __restrict__ peer_flag,
                                    uint32_t* __restrict__ gen_slot,
                                    volatile uint32_t* __restrict__ abort_dev,
                                    int n) {
    __shared__ uint32_t s_slot;
    __shared__ uint32_t s_aborted;
    if (threadIdx.x == 0) {
        uint32_t g = *gen_slot + 1u;
        if (g == 0u) g = 1u;   // keep 0 reserved for initial state
        *gen_slot = g;
        s_slot = g & 1u;
        s_aborted = 0u;
    }
    __syncthreads();
    const uint32_t slot = s_slot;
    float* __restrict__ sb = slot ? self_b1 : self_b0;
    const float* __restrict__ pb = slot ? peer_b1 : peer_b0;

    const int n4 = n >> 2;
    float4* __restrict__ d4 = reinterpret_cast<float4*>(data);
    float4* __restrict__ sb4 = reinterpret_cast<float4*>(sb);
    const float4* __restrict__ pb4 = reinterpret_cast<const float4*>(pb);

    // Stage local partial sum into uncached P2P buffer
    for (int i = threadIdx.x; i < n4; i += blockDim.x) sb4[i] = d4[i];
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint32_t g = *gen_slot;
        __threadfence_system();
        peer_flag[slot] = g;
        __threadfence_system();
        while (self_flag[slot] != g) {
            if (*abort_dev != 0u) {
                s_aborted = 1u;
                break;
            }
#ifdef STRATA_USE_HIP
            __builtin_amdgcn_s_sleep(1);
#endif
        }
        __threadfence_system();
    }
    __syncthreads();
    if (s_aborted) return;

    // Sum local + peer into data
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 a = sb4[i];
        const float4 b = pb4[i];
        d4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Symmetric FP16 sum so both ranks compute the exact same float value (fp16(a) + fp16(b)).
__global__ void sum_f16_pair_to_f32_kernel(float* __restrict__ dst,
                                           const __half* __restrict__ a,
                                           const __half* __restrict__ b,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(a[i]) + __half2float(b[i]);
}

}  // namespace

std::shared_ptr<Tp2Comm> Tp2Comm::create(int dev0, int dev1, int max_floats) {
    DeviceGuard guard;
    int count = 0;
    ck(cudaGetDeviceCount(&count), "tp2 device count");
    if (dev0 < 0 || dev1 < 0 || dev0 >= count || dev1 >= count || dev0 == dev1)
        throw std::runtime_error("TP2 needs two distinct visible GPUs (found " + std::to_string(count) + ")");

    int can01 = 0, can10 = 0;
    ck(cudaDeviceCanAccessPeer(&can01, dev0, dev1), "tp2 peer check 0->1");
    ck(cudaDeviceCanAccessPeer(&can10, dev1, dev0), "tp2 peer check 1->0");
    if (!can01 || !can10)
        throw std::runtime_error("TP2 requires bidirectional PCIe P2P between GPUs " + std::to_string(dev0) + " and " +
                                 std::to_string(dev1));

    std::shared_ptr<Tp2Comm> c(new Tp2Comm());
    c->dev_[0] = dev0;
    c->dev_[1] = dev1;
    c->max_floats_ = std::max(max_floats, kSmallMax);

    ck(cudaSetDevice(dev0), "tp2 setDevice abort");
#ifdef STRATA_USE_HIP
    ck(hipHostMalloc((void**) &c->abort_dev_, sizeof(uint32_t), hipHostMallocMapped), "tp2 abort_dev");
#else
    ck(cudaHostAlloc((void**) &c->abort_dev_, sizeof(uint32_t), cudaHostAllocMapped), "tp2 abort_dev");
#endif
    *c->abort_dev_ = 0u;

    for (int r = 0; r < 2; ++r) {
        ck(cudaSetDevice(c->dev_[r]), "tp2 setDevice");
        cudaError_t pe = cudaDeviceEnablePeerAccess(c->dev_[1 - r], 0);
        if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled)
            ck(pe, "tp2 enablePeerAccess");
        (void) cudaGetLastError();

        const size_t small_bytes = (size_t) kSmallMax * sizeof(float);
        for (int b = 0; b < 2; ++b) {
#ifdef STRATA_USE_HIP
            ck(hipExtMallocWithFlags((void**) &c->p2p_buf_[r][b], small_bytes, hipDeviceMallocUncached),
               "tp2 p2p_buf");
#else
            ck(cudaMalloc((void**) &c->p2p_buf_[r][b], small_bytes), "tp2 p2p_buf");
#endif
            ck(cudaMemset(c->p2p_buf_[r][b], 0, small_bytes), "tp2 p2p_buf zero");
        }
#ifdef STRATA_USE_HIP
        ck(hipExtMallocWithFlags((void**) &c->p2p_flag_[r], 64, hipDeviceMallocUncached), "tp2 p2p_flag");
#else
        ck(cudaMalloc((void**) &c->p2p_flag_[r], 64), "tp2 p2p_flag");
#endif
        ck(cudaMemset(c->p2p_flag_[r], 0, 64), "tp2 p2p_flag zero");
        ck(cudaMalloc((void**) &c->p2p_gen_[r], 64), "tp2 p2p_gen");
        ck(cudaMemset(c->p2p_gen_[r], 0, 64), "tp2 p2p_gen zero");

        const size_t dma_bytes = (size_t) c->max_floats_ * sizeof(uint16_t);
        ck(cudaMalloc((void**) &c->dma_send_[r], dma_bytes), "tp2 dma_send");
        ck(cudaMalloc((void**) &c->dma_recv_[r], dma_bytes), "tp2 dma_recv");
        if (r == 0) {
            cudaEvent_t eb = nullptr;
            ck(cudaEventCreateWithFlags(&eb, cudaEventDisableTiming), "tp2 ev_bcast");
            c->ev_bcast_ = eb;
        } else {
            cudaEvent_t ed = nullptr;
            ck(cudaEventCreateWithFlags(&ed, cudaEventDisableTiming), "tp2 ev_bcast_done");
            c->ev_bcast_done_ = ed;
        }
        ck(cudaDeviceSynchronize(), "tp2 init sync");
    }
    std::fprintf(stderr, "strata-poly: TP2 P2P enabled between GPU %d and GPU %d (uncached ping-pong %d floats, dma %d floats)\n",
                 dev0, dev1, kSmallMax, c->max_floats_);
    return c;
}

Tp2Comm::~Tp2Comm() {
    abort_barrier();
    DeviceGuard guard;
    for (int r = 0; r < 2; ++r) {
        if (cudaSetDevice(dev_[r]) != cudaSuccess) continue;
        for (int b = 0; b < 2; ++b)
            if (p2p_buf_[r][b]) (void) cudaFree(p2p_buf_[r][b]);
        if (p2p_flag_[r]) (void) cudaFree(p2p_flag_[r]);
        if (p2p_gen_[r]) (void) cudaFree(p2p_gen_[r]);
        if (dma_send_[r]) (void) cudaFree(dma_send_[r]);
        if (dma_recv_[r]) (void) cudaFree(dma_recv_[r]);
        if (r == 0 && ev_bcast_) (void) cudaEventDestroy(static_cast<cudaEvent_t>(ev_bcast_));
        if (r == 1 && ev_bcast_done_) (void) cudaEventDestroy(static_cast<cudaEvent_t>(ev_bcast_done_));
    }
    if (abort_dev_) {
#ifdef STRATA_USE_HIP
        (void) hipHostFree(abort_dev_);
#else
        (void) cudaFreeHost(abort_dev_);
#endif
    }
}

void Tp2Comm::abort_barrier() {
    if (abort_dev_) __atomic_store_n(abort_dev_, 1u, __ATOMIC_RELEASE);
    bar_aborted_.store(true, std::memory_order_release);
}

void Tp2Comm::host_barrier(int rank) {
    if (bar_aborted_.load(std::memory_order_acquire))
        throw std::runtime_error("Tp2Comm: communicator aborted");
    const uint64_t next = ++bar_epoch_[rank];
    bar_arrived_[rank].store(next, std::memory_order_release);
    const int peer = 1 - rank;
    while (bar_arrived_[peer].load(std::memory_order_acquire) < next) {
        if (bar_aborted_.load(std::memory_order_acquire))
            throw std::runtime_error("Tp2Comm: peer aborted during barrier");
    }
}

void Tp2Comm::all_reduce(int rank, float* data, int n, void* stream) {
    if (bar_aborted_.load(std::memory_order_acquire))
        throw std::runtime_error("Tp2Comm::all_reduce: communicator aborted");
    const int peer = 1 - rank;
    if (n <= kSmallMax && (n & 3) == 0) {
        p2p_ar_pingpong_f32<<<1, 256, 0, cs(stream)>>>(
            data, p2p_buf_[rank][0], p2p_buf_[rank][1], p2p_buf_[peer][0], p2p_buf_[peer][1],
            p2p_flag_[rank], p2p_flag_[peer], p2p_gen_[rank], abort_dev_, n);
        return;
    }
    if (n > max_floats_)
        throw std::runtime_error("Tp2Comm::all_reduce: n=" + std::to_string(n) + " exceeds max_floats=" +
                                 std::to_string(max_floats_));
    // Large prefill path: symmetric FP16 peer copy (both ranks compute fp16(a) + fp16(b))
    const int blocks = (n + 255) / 256;
    f32_to_f16_kernel<<<blocks, 256, 0, cs(stream)>>>(data, reinterpret_cast<__half*>(dma_send_[rank]), n);
    ck(cudaStreamSynchronize(cs(stream)), "tp2 dma ready sync");
    host_barrier(rank);
    ck(cudaMemcpyPeerAsync(dma_recv_[rank], dev_[rank], dma_send_[peer], dev_[peer],
                           (size_t) n * sizeof(uint16_t), cs(stream)), "tp2 dma peer copy");
    sum_f16_pair_to_f32_kernel<<<blocks, 256, 0, cs(stream)>>>(
        data, reinterpret_cast<const __half*>(dma_send_[rank]), reinterpret_cast<const __half*>(dma_recv_[rank]), n);
    ck(cudaStreamSynchronize(cs(stream)), "tp2 dma done sync");
    host_barrier(rank);
}

void Tp2Comm::broadcast_from_0(int rank, float* data, int n, void* stream) {
    if (bar_aborted_.load(std::memory_order_acquire))
        throw std::runtime_error("Tp2Comm::broadcast_from_0: communicator aborted");
    if (rank == 0) {
        bcast_src_ = data;
        ck(cudaEventRecord(static_cast<cudaEvent_t>(ev_bcast_), cs(stream)), "tp2 bcast record");
        host_barrier(0);   // ev_bcast_ and bcast_src_ are published
        host_barrier(0);   // wait until rank 1 has recorded ev_bcast_done_
        ck(cudaStreamWaitEvent(cs(stream), static_cast<cudaEvent_t>(ev_bcast_done_), 0), "tp2 bcast done wait");
    } else {
        host_barrier(1);
        ck(cudaStreamWaitEvent(cs(stream), static_cast<cudaEvent_t>(ev_bcast_), 0), "tp2 bcast wait");
        ck(cudaMemcpyPeerAsync(data, dev_[1], bcast_src_, dev_[0], (size_t) n * sizeof(float), cs(stream)),
           "tp2 bcast peer copy");
        ck(cudaEventRecord(static_cast<cudaEvent_t>(ev_bcast_done_), cs(stream)), "tp2 bcast done record");
        host_barrier(1);
    }
}

}  // namespace strata::poly
