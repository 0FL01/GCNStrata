// src/poly/tp2_comm.cu - 2-GPU Tensor Parallel P2P collective primitives.
#include "tp2_comm.hpp"
#include "runtime.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef STRATA_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <cstdio>
#include <cstdlib>
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
// Pull reads the peer's outbox. Push writes the peer's inbox and publishes only after all lanes finish.
template<bool Push>
__global__ void p2p_ar_pingpong_f32(float* __restrict__ data,
                                    float* __restrict__ self_b0, float* __restrict__ self_b1,
                                    float* __restrict__ peer_b0, float* __restrict__ peer_b1,
                                    volatile uint32_t* __restrict__ self_flag,
                                    volatile uint32_t* __restrict__ peer_flag,
                                    uint32_t* __restrict__ gen_slot,
                                    volatile uint32_t* __restrict__ abort_dev,
                                    int n) {
    __shared__ uint32_t s_slot;
    __shared__ uint32_t s_aborted;
    if (threadIdx.x == 0) {
        uint32_t g = *gen_slot + 1u;
        if (!Push && g == 0u) g = 1u;   // Push must alternate inboxes even at uint32 wrap.
        *gen_slot = g;
        s_slot = g & 1u;
        s_aborted = 0u;
    }
    __syncthreads();
    const uint32_t slot = s_slot;
    float* __restrict__ sb = slot ? self_b1 : self_b0;
    float* __restrict__ pb = slot ? peer_b1 : peer_b0;

    const int n4 = n >> 2;
    float4* __restrict__ d4 = reinterpret_cast<float4*>(data);
    float4* __restrict__ send = reinterpret_cast<float4*>(Push ? pb : sb);
    const float4* __restrict__ recv = reinterpret_cast<const float4*>(Push ? sb : pb);

    for (int i = threadIdx.x; i < n4; i += blockDim.x) send[i] = d4[i];
    if constexpr (Push) __threadfence_system();   // lane 0 alone cannot publish the other lanes' PCIe writes
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint32_t g = *gen_slot;
#if defined(STRATA_HIP_GFX906)
        if constexpr (Push) {
            __hip_atomic_store(const_cast<uint32_t*>(peer_flag) + slot, g, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        } else
#endif
        {
            __threadfence_system();
            peer_flag[slot] = g;
            __threadfence_system();
        }
        for (;;) {
#if defined(STRATA_HIP_GFX906)
            if constexpr (Push) {
                if (__hip_atomic_load(const_cast<uint32_t*>(self_flag) + slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == g) break;
            } else
#endif
            if (self_flag[slot] == g) break;
            if (*abort_dev != 0u) {
                s_aborted = 1u;
                break;
            }
#ifdef STRATA_USE_HIP
            __builtin_amdgcn_s_sleep(1);
#endif
        }
        if constexpr (!Push) __threadfence_system();
    }
    __syncthreads();
    if (s_aborted) return;

    // The next inbox is distinct. Reaching this slot again requires both ranks to have started the next hop.
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 a = d4[i];
        const float4 b = recv[i];
        d4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

// Symmetric FP16 ping-pong P2P all-reduce for T = 2..8 (halves PCIe bytes, identical fp16(a)+fp16(b) on both ranks).
__global__ void p2p_ar_pingpong_f16(float* __restrict__ data,
                                    __half2* __restrict__ self_b0, __half2* __restrict__ self_b1,
                                    const __half2* __restrict__ peer_b0, const __half2* __restrict__ peer_b1,
                                    volatile uint32_t* __restrict__ self_flag,
                                    volatile uint32_t* __restrict__ peer_flag,
                                    uint32_t* __restrict__ gen_slot,
                                    volatile uint32_t* __restrict__ abort_dev,
                                    int n) {
    __shared__ uint32_t s_slot;
    __shared__ uint32_t s_aborted;
    if (threadIdx.x == 0) {
        uint32_t g = *gen_slot + 1u;
        if (g == 0u) g = 1u;
        *gen_slot = g;
        s_slot = g & 1u;
        s_aborted = 0u;
    }
    __syncthreads();
    const uint32_t slot = s_slot;
    __half2* __restrict__ sb = slot ? self_b1 : self_b0;
    const __half2* __restrict__ pb = slot ? peer_b1 : peer_b0;

    const int n4 = n >> 2;
    float4* __restrict__ d4 = reinterpret_cast<float4*>(data);
    uint2* __restrict__ sb2 = reinterpret_cast<uint2*>(sb);
    const uint2* __restrict__ pb2 = reinterpret_cast<const uint2*>(pb);

    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = d4[i];
        const __half2 lo = __floats2half2_rn(v.x, v.y);
        const __half2 hi = __floats2half2_rn(v.z, v.w);
        uint2 packed;
        packed.x = *reinterpret_cast<const unsigned int*>(&lo);
        packed.y = *reinterpret_cast<const unsigned int*>(&hi);
        sb2[i] = packed;
    }
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

    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = d4[i];
        const __half2 alo = __floats2half2_rn(v.x, v.y);
        const __half2 ahi = __floats2half2_rn(v.z, v.w);
        const uint2 bp = pb2[i];
        const __half2 blo = *reinterpret_cast<const __half2*>(&bp.x);
        const __half2 bhi = *reinterpret_cast<const __half2*>(&bp.y);
        d4[i] = make_float4(__low2float(alo) + __low2float(blo),
                            __high2float(alo) + __high2float(blo),
                            __low2float(ahi) + __low2float(bhi),
                            __high2float(ahi) + __high2float(bhi));
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

#if defined(STRATA_HIP_GFX906)
__device__ __forceinline__ uint32_t pack_half2_bits(float a, float b) {
    const __half2 h = __halves2half2(__float2half(a), __float2half(b));
    return *reinterpret_cast<const uint32_t*>(&h);
}

__global__ void f32_to_f16_push_uint4_kernel(const float* __restrict__ src,
                                             uint4* __restrict__ peer,
                                             unsigned vectors) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < vectors; i += blockDim.x * gridDim.x) {
        const float* x = src + 8 * i;
        peer[i] = make_uint4(pack_half2_bits(x[0], x[1]), pack_half2_bits(x[2], x[3]),
                             pack_half2_bits(x[4], x[5]), pack_half2_bits(x[6], x[7]));
    }
}

__global__ void sum_f16_direct_kernel(float* __restrict__ data,
                                      const uint4* __restrict__ peer,
                                      unsigned vectors) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vectors) return;
    const uint4 p = peer[i];
    const __half2 h0 = *reinterpret_cast<const __half2*>(&p.x);
    const __half2 h1 = *reinterpret_cast<const __half2*>(&p.y);
    const __half2 h2 = *reinterpret_cast<const __half2*>(&p.z);
    const __half2 h3 = *reinterpret_cast<const __half2*>(&p.w);
    float4* dst = reinterpret_cast<float4*>(data);
    float4 d0 = dst[2 * i];
    float4 d1 = dst[2 * i + 1];
    d0.x = __half2float(__float2half(d0.x)) + __low2float(h0);
    d0.y = __half2float(__float2half(d0.y)) + __high2float(h0);
    d0.z = __half2float(__float2half(d0.z)) + __low2float(h1);
    d0.w = __half2float(__float2half(d0.w)) + __high2float(h1);
    d1.x = __half2float(__float2half(d1.x)) + __low2float(h2);
    d1.y = __half2float(__float2half(d1.y)) + __high2float(h2);
    d1.z = __half2float(__float2half(d1.z)) + __low2float(h3);
    d1.w = __half2float(__float2half(d1.w)) + __high2float(h3);
    dst[2 * i] = d0;
    dst[2 * i + 1] = d1;
}
#endif

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
#if defined(STRATA_HIP_GFX906)
        static const bool want_direct = [] {
            const char* v = std::getenv("STRATA_TP2_PREFILL_DIRECT");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        c->direct_enabled_ = want_direct;
        if (want_direct) {
            for (int b = 0; b < 2; ++b) {
                ck(hipExtMallocWithFlags((void**) &c->direct_inbox_[r][b], dma_bytes, hipDeviceMallocUncached),
                   "tp2 direct_inbox");
                hipEvent_t ep = nullptr, ed = nullptr;
                ck(hipEventCreateWithFlags(&ep, hipEventDisableTiming | hipEventReleaseToSystem),
                   "tp2 ev_direct_pushed");
                ck(hipEventCreateWithFlags(&ed, hipEventDisableTiming), "tp2 ev_direct_done");
                c->ev_direct_pushed_[r][b] = ep;
                c->ev_direct_done_[r][b] = ed;
            }
        }
#endif
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
        for (int b = 0; b < 2; ++b) {
            if (direct_inbox_[r][b]) (void) cudaFree(direct_inbox_[r][b]);
            if (ev_direct_pushed_[r][b]) (void) cudaEventDestroy(static_cast<cudaEvent_t>(ev_direct_pushed_[r][b]));
            if (ev_direct_done_[r][b]) (void) cudaEventDestroy(static_cast<cudaEvent_t>(ev_direct_done_[r][b]));
        }
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

bool Tp2Comm::small_push_enabled() {
#if defined(STRATA_HIP_GFX906)
    static const bool enabled = [] {
        const char* push = std::getenv("STRATA_TP2_SMALL_PUSH");
        const char* half = std::getenv("STRATA_TP2_P2P_F16");
        return push && push[0] == '1' && !push[1] && !(half && half[0] == '1' && !half[1]);
    }();
    return enabled;
#else
    return false;
#endif
}

void Tp2Comm::all_reduce(int rank, float* data, int n, void* stream) {
    if (bar_aborted_.load(std::memory_order_acquire))
        throw std::runtime_error("Tp2Comm::all_reduce: communicator aborted");
    const int peer = 1 - rank;
    if (n <= kSmallMax && (n & 3) == 0) {
        static const bool use_f16 = [] {
            const char* v = std::getenv("STRATA_TP2_P2P_F16");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        if (use_f16 && n > 2304) {
            p2p_ar_pingpong_f16<<<1, 256, 0, cs(stream)>>>(
                data,
                reinterpret_cast<__half2*>(p2p_buf_[rank][0]), reinterpret_cast<__half2*>(p2p_buf_[rank][1]),
                reinterpret_cast<const __half2*>(p2p_buf_[peer][0]), reinterpret_cast<const __half2*>(p2p_buf_[peer][1]),
                p2p_flag_[rank], p2p_flag_[peer], p2p_gen_[rank], abort_dev_, n);
        } else if (small_push_enabled()) {
            p2p_ar_pingpong_f32<true><<<1, 256, 0, cs(stream)>>>(
                data, p2p_buf_[rank][0], p2p_buf_[rank][1], p2p_buf_[peer][0], p2p_buf_[peer][1],
                p2p_flag_[rank], p2p_flag_[peer], p2p_gen_[rank], abort_dev_, n);
        } else {
            p2p_ar_pingpong_f32<false><<<1, 256, 0, cs(stream)>>>(
                data, p2p_buf_[rank][0], p2p_buf_[rank][1], p2p_buf_[peer][0], p2p_buf_[peer][1],
                p2p_flag_[rank], p2p_flag_[peer], p2p_gen_[rank], abort_dev_, n);
        }
        return;
    }
    if (n > max_floats_)
        throw std::runtime_error("Tp2Comm::all_reduce: n=" + std::to_string(n) + " exceeds max_floats=" +
                                 std::to_string(max_floats_));
    // Large prefill path: symmetric FP16 peer copy (both ranks compute fp16(a) + fp16(b))
#if defined(STRATA_HIP_GFX906)
    if (direct_enabled_ && (n & 7) == 0 && n >= 32 * 2304) {
        const int slot = direct_slot_[rank];
        if (direct_used_[rank][slot])
            ck(cudaStreamWaitEvent(cs(stream), static_cast<cudaEvent_t>(ev_direct_done_[peer][slot]), 0),
               "tp2 direct wait done");
        const unsigned vecs = (unsigned) (n / 8);
        const unsigned pblocks = std::min(480u, (vecs + 255u) / 256u);
        f32_to_f16_push_uint4_kernel<<<pblocks, 256, 0, cs(stream)>>>(
            data, reinterpret_cast<uint4*>(direct_inbox_[peer][slot]), vecs);
        ck(cudaEventRecord(static_cast<cudaEvent_t>(ev_direct_pushed_[rank][slot]), cs(stream)),
           "tp2 direct record pushed");
        host_barrier(rank);
        ck(cudaStreamWaitEvent(cs(stream), static_cast<cudaEvent_t>(ev_direct_pushed_[peer][slot]), 0),
           "tp2 direct wait pushed");
        const unsigned sblocks = (vecs + 255u) / 256u;
        sum_f16_direct_kernel<<<sblocks, 256, 0, cs(stream)>>>(
            data, reinterpret_cast<const uint4*>(direct_inbox_[rank][slot]), vecs);
        ck(cudaEventRecord(static_cast<cudaEvent_t>(ev_direct_done_[rank][slot]), cs(stream)),
           "tp2 direct record done");
        direct_used_[rank][slot] = true;
        direct_slot_[rank] ^= 1;
        return;
    }
#endif
    const int blocks = (n + 255) / 256;
    f32_to_f16_kernel<<<blocks, 256, 0, cs(stream)>>>(data, reinterpret_cast<__half*>(dma_send_[rank]), n);
    ck(cudaStreamSynchronize(cs(stream)), "tp2 dma ready sync");
    host_barrier(rank);
#if defined(STRATA_HIP_GFX906)
    static const bool use_push = [] {
        const char* v = std::getenv("STRATA_TP2_PREFILL_PUSH");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (use_push && n >= 32 * 2304) {
        // This rank's stream pushes its own send buffer. Both writes must finish before either rank sums,
        // since our stream alone does not order the peer's write into our receive buffer.
        ck(cudaMemcpyPeerAsync(dma_recv_[peer], dev_[peer], dma_send_[rank], dev_[rank],
                               (size_t) n * sizeof(uint16_t), cs(stream)), "tp2 dma peer push");
        ck(cudaStreamSynchronize(cs(stream)), "tp2 dma push sync");
        host_barrier(rank);
    } else
#endif
    {
        ck(cudaMemcpyPeerAsync(dma_recv_[rank], dev_[rank], dma_send_[peer], dev_[peer],
                               (size_t) n * sizeof(uint16_t), cs(stream)), "tp2 dma peer copy");
    }
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

void Tp2Comm::gather_from_1(int rank, float* dst_on_0, const float* src_on_1, int n, void* stream) {
    if (bar_aborted_.load(std::memory_order_acquire))
        throw std::runtime_error("Tp2Comm::gather_from_1: communicator aborted");
    if (rank == 0) {
        bcast_src_ = dst_on_0;
        host_barrier(0);   // dst_on_0 published
        host_barrier(0);   // wait until rank 1 has recorded ev_bcast_done_
        ck(cudaStreamWaitEvent(cs(stream), static_cast<cudaEvent_t>(ev_bcast_done_), 0), "tp2 gather wait");
    } else {
        host_barrier(1);
        ck(cudaMemcpyPeerAsync(bcast_src_, dev_[0], src_on_1, dev_[1], (size_t) n * sizeof(float), cs(stream)),
           "tp2 gather peer copy");
        ck(cudaEventRecord(static_cast<cudaEvent_t>(ev_bcast_done_), cs(stream)), "tp2 gather done record");
        host_barrier(1);
    }
}

}  // namespace strata::poly
