// Packed gfx906 DFlash2 kernels, adapted from src/poly/inkling_dflash2.cu.
// Included after the Qwen baseline kernels (inside its private namespace).

struct Packed {
    const int4* codes = nullptr;
    const __half* scales = nullptr;
    int in = 0, out = 0;
    bool q4 = false;
    Packed slice(int first, int count) const {
        const size_t off = size_t(first) * (in / 32);
        return {codes + off * (q4 ? 1 : 2), scales + off, in, count, q4};
    }
};
struct Quant {
    int8_t* q;
    float* d;
    int32_t* sum;
    Quant slice(int cols, int first) const {
        const size_t at = size_t(cols) * first;
        return {q + at, d + at / 32, sum + at / 32};
    }
};
struct Outs { float* y[3]; int n[3]; };
constexpr int kTileK = 2560;
constexpr int kHeadCta = 64, kHeadLists = (kVocab + kHeadCta - 1) / kHeadCta;
constexpr int kMergeCta = 256, kMergeLists = (kHeadLists + kMergeCta - 1) / kMergeCta;

__global__ void pack_kernel(const uint8_t* __restrict__ src, uint8_t* q, __half* d, size_t total) {
    const size_t b = (size_t) blockIdx.x * 8 + threadIdx.x / 32;   // a block of 32 values a 32-thread group
    if (b >= total) return;
    const int j = threadIdx.x % 32;
    const uint8_t* blk = src + b * 34;
    q[b * 32 + j] = blk[2 + j];
    if (j == 0) d[b] = *reinterpret_cast<const __half*>(blk);
}
template <typename Scale>
__global__ void pack4_kernel(const uint8_t* __restrict__ src, uint8_t* q, Scale* d, size_t total) {
    const size_t b = (size_t) blockIdx.x * 16 + threadIdx.x / 16;
    if (b >= total) return;
    const int j = threadIdx.x % 16;
    const uint8_t* blk = src + b * 18;
    q[b * 16 + j] = blk[2 + j];
    if (j == 0) {
        const __half scale = *reinterpret_cast<const __half*>(blk);
        if constexpr (sizeof(Scale) == 2) d[b] = scale;
        else d[b] = __half2float(scale);
    }
}

template <bool Sum>
__device__ __forceinline__ void quant_block(float v, int lane, int8_t* q, float* d, int32_t* xsum, size_t b) {
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
    const float s = a / 127.f;
    const auto code = (int8_t) (s > 0.f ? __float2int_rn(v / s) : 0);
    *q = code;
    if (lane == 0) d[b] = s;
    if constexpr (Sum) {
        int sum = code;
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
        if (lane == 0) xsum[b] = sum;
    }
}

template <bool Sum>
__global__ void quantize_kernel(const float* __restrict__ x, int n_in, int8_t* q, float* d, int32_t* xsum) {
    // a 32-value block a 32-lane group: d = amax / 127, q = round(x / d)
    const int r = blockIdx.y, b = blockIdx.x * 8 + threadIdx.x / 32, j = threadIdx.x % 32;
    if (b * 32 >= n_in) return;
    const size_t at = (size_t) r * n_in + b * 32 + j;
    quant_block<Sum>(x[at], j, q + at, d, xsum, (size_t) r * (n_in / 32) + b);
}
template <bool Sum>
__global__ void __launch_bounds__(256) rms_norm_quantize_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                                float* __restrict__ y_out, int8_t* __restrict__ q,
                                                                float* __restrict__ d, int32_t* xsum, int n_in,
                                                                float eps) {
    __shared__ float sw[4];
    const int r = int(blockIdx.x), tid = int(threadIdx.x);
    const float* xr = x + (size_t) r * n_in;
    float ss = 0.f;
    for (int i = tid; i < n_in; i += 256) {
        const float v = xr[i];
        ss += v * v;
    }
    for (int o = 32; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 64);
    if ((tid & 63) == 0) sw[tid >> 6] = ss;
    __syncthreads();
    const float total = sw[0] + sw[1] + sw[2] + sw[3];
    const float scale = rsqrtf(total / float(n_in) + eps);
    const int sub = tid >> 5, lane = tid & 31, blocks = n_in / 32;
    float* yr = y_out ? y_out + (size_t) r * n_in : nullptr;
    int8_t* qr = q + (size_t) r * n_in;
    for (int b = sub; b < blocks; b += 8) {
        const int c = b * 32 + lane;
        const float val = xr[c] * scale * w[c];
        if (yr) yr[c] = val;
        quant_block<Sum>(val, lane, qr + c, d, xsum, (size_t) r * blocks + b);
    }
}

template <bool Sum>
__global__ void __launch_bounds__(256) conv_quantize_kernel(const float* __restrict__ x, const float* __restrict__ dyn,
                                                            const float* __restrict__ base, int stage,
                                                            int8_t* __restrict__ q, float* __restrict__ d, int32_t* xsum) {
    const int r = int(blockIdx.y), b = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5), j = int(threadIdx.x) & 31;
    const int c = b * 32 + j;
    if (c >= kWidth) return;
    const int g = c / 16;
    const float* dn = dyn + (size_t) r * kDyn + stage * 2 * (kWidth / kGroup);
    float val = (base[(stage * 2 + 0) * kWidth + c] + dn[g]) * x[(size_t) r * kWidth + c];
    if (r) val += (base[(stage * 2 + 1) * kWidth + c] + dn[(kWidth / kGroup) + g]) * x[(size_t) (r - 1) * kWidth + c];
    quant_block<Sum>(val, j, q + (size_t) r * kWidth + c, d, xsum, (size_t) r * (kWidth / 32) + b);
}

template <bool Sum>
__global__ void __launch_bounds__(256) swiglu_quantize_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                                              int n_in, int8_t* __restrict__ q, float* __restrict__ d,
                                                              int32_t* xsum) {
    const int r = int(blockIdx.y), b = int(blockIdx.x) * 8 + (int(threadIdx.x) >> 5), j = int(threadIdx.x) & 31;
    const int c = b * 32 + j;
    if (c >= n_in) return;
    const size_t at = (size_t) r * n_in + c;
    const float gv = g[at];
    const float val = (gv / (1.0f + expf(-gv))) * u[at];
    quant_block<Sum>(val, j, q + at, d, xsum, (size_t) r * (n_in / 32) + b);
}

template <int M, int Rows>
__device__ __forceinline__ void local_top16(const float* stage, Candidate* parts, int n) {
    static_assert(Rows <= 64);
    const int t = threadIdx.x, wave = t / 64, lane = t % 64;
    const int out = int(blockIdx.x) * Rows + lane;
    for (int m = wave; m < M; m += 4) {
        Candidate a = kInvalid;
        if (lane < Rows && out < n) {
            a = {stage[m * Rows + lane], out};
            if (isnan(a.logit)) a.logit = -INFINITY;   // keep a valid id even if all scores are NaN
        }
#pragma unroll
        for (int len = 2; len <= Rows; len *= 2) {
#pragma unroll
            for (int hop = len / 2; hop > 0; hop /= 2) {
                const Candidate b{__shfl_xor(a.logit, hop, 64), __shfl_xor(a.id, hop, 64)};
                const bool want_better = ((lane & len) == 0) == ((lane & hop) == 0);
                if (want_better ? better(b, a) : better(a, b)) a = b;
            }
        }
        if (lane < kTop) {
            // Make rank 0 contiguous across CTAs for the merge; the remaining
            // 15 are contiguous within a CTA's list. Exactly 16 pairs/CTA/pos.
            const size_t i = lane == 0 ? blockIdx.x : gridDim.x + size_t(blockIdx.x) * (kTop - 1) + lane - 1;
            parts[size_t(m) * gridDim.x * kTop + i] = a;
        }
    }
}

template <int M, int L, int R, bool Q4, bool Fused = false>
__global__ void __launch_bounds__(256) planar_kernel(
        const int4* __restrict__ wq, const __half* __restrict__ wd,
        const int8_t* __restrict__ xq, const float* __restrict__ xd, const int32_t* __restrict__ xsum,
        Outs out, Candidate* __restrict__ parts, int k, int n) {
    constexpr int rows = (256 / L) * R, kb_tile = kTileK / 32;
    static_assert(!Fused || rows <= 64);
    __shared__ int4 xs[M * (kTileK / 16)];
    __shared__ float xds[M * kb_tile];
    __shared__ int xcs[Q4 ? M * kb_tile : 1];
    static_assert(!Fused || rows <= kb_tile);
    const int t = threadIdx.x, lane = t % L, local = (t / L) * R;
    const int first = int(blockIdx.x) * rows + local, kb = k / 32;
    const int batch_start = blockIdx.y * M;
    xq += size_t(batch_start) * k;
    xd += size_t(batch_start) * kb;
    if constexpr (Q4) xsum += size_t(batch_start) * kb;
    float acc[R][M] = {};
    for (int k0 = 0; k0 < kb; k0 += kb_tile) {
        const int blks = min(kb_tile, kb - k0), chunks = 2 * blks;
        __syncthreads();
        for (int i = t; i < M * chunks; i += 256) {
            const int m = i / chunks, c = i % chunks;
            xs[m * (kTileK / 16) + c] = reinterpret_cast<const int4*>(xq)[size_t(m) * 2 * kb + 2 * k0 + c];
        }
        for (int i = t; i < M * blks; i += 256) {
            const int m = i / blks, b = i % blks;
            const size_t xi = size_t(m) * kb + k0 + b;
            xds[m * kb_tile + b] = xd[xi];
            if constexpr (Q4) xcs[m * kb_tile + b] = -8 * xsum[xi];
        }
        __syncthreads();
        if (first >= n) continue;
        for (int b = lane; b < blks; b += L) {
            int4 a0[R], a1[R];
            float ws[R];
            constexpr int mask = 0x0f0f0f0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                a0[r] = a1[r] = make_int4(0, 0, 0, 0);
                ws[r] = 0.f;
                if (first + r < n) {
                    const size_t wi = size_t(first + r) * kb + k0 + b;
                    if constexpr (Q4) {
                        const int4 a = wq[wi];
                        a0[r] = make_int4(a.x & mask, a.y & mask, a.z & mask, a.w & mask);
                        a1[r] = make_int4((a.x >> 4) & mask, (a.y >> 4) & mask, (a.z >> 4) & mask, (a.w >> 4) & mask);
                    } else {
                        a0[r] = wq[2 * wi];
                        a1[r] = wq[2 * wi + 1];
                    }
                    ws[r] = __half2float(wd[wi]);
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const int4 x0 = xs[m * (kTileK / 16) + 2 * b], x1 = xs[m * (kTileK / 16) + 2 * b + 1];
                const float xd0 = xds[m * kb_tile + b];
                const int init = Q4 ? xcs[m * kb_tile + b] : 0;   // -8 * sum(xq) for biased nibbles
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int dot = __builtin_amdgcn_sdot4(a0[r].x, x0.x, init, false);
                    dot = __builtin_amdgcn_sdot4(a0[r].y, x0.y, dot, false);
                    dot = __builtin_amdgcn_sdot4(a0[r].z, x0.z, dot, false);
                    dot = __builtin_amdgcn_sdot4(a0[r].w, x0.w, dot, false);
                    dot = __builtin_amdgcn_sdot4(a1[r].x, x1.x, dot, false);
                    dot = __builtin_amdgcn_sdot4(a1[r].y, x1.y, dot, false);
                    dot = __builtin_amdgcn_sdot4(a1[r].z, x1.z, dot, false);
                    dot = __builtin_amdgcn_sdot4(a1[r].w, x1.w, dot, false);
                    acc[r][m] += float(dot) * (ws[r] * xd0);
                }
            }
        }
    }
    // Reuse the activation-scale tile after all its readers are done. A
    // separate M*64-float scratch tile would reduce Q4 head CTA residency
    // from three to two on gfx906 (64 KiB LDS/CU).
    if constexpr (Fused) __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (first + r >= n) break;
        int seg = 0, at = first + r;
        if constexpr (!Fused) {
            while (seg < 2 && at >= out.n[seg]) at -= out.n[seg++];
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = acc[r][m];
            for (int o = L / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, L);
            if (lane == 0) {
                if constexpr (Fused) xds[m * rows + local + r] = v;
                else out.y[seg][size_t(batch_start + m) * out.n[seg] + at] = v;
            }
        }
    }
    if constexpr (Fused) {
        __syncthreads();
        local_top16<M, rows>(xds, parts, n);
    }
}

__global__ void __launch_bounds__(256) attend3_part_kernel(const float* __restrict__ q, const float* __restrict__ kb,
                                                            const float* __restrict__ vb, const __half* __restrict__ kc,
                                                            const __half* __restrict__ vc, int64_t lo0,
                                                            int n_ring, int rows, float* part) {
    __shared__ __half2 qs[kBlock * 4][kH / 2 + 1];
    __shared__ __half2 ks[kChunk][kH / 2 + 1];
    __shared__ __half2 vs[kChunk][kH / 2 + 1];
    __shared__ float p[kBlock * 4][kChunk + 1];
    const int g = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
    const int nq = rows * 4, n_keys = n_ring + rows, k0 = c * kChunk;
    for (int i = tid; i < nq * (kH / 2); i += 256) {
        const int qi = i / (kH / 2), d2 = i % (kH / 2), row = qi / 4, hh = qi % 4;
        const float2 f = reinterpret_cast<const float2*>(q + ((size_t) row * kQHeads + g * 4 + hh) * kH)[d2];
        qs[qi][d2] = __floats2half2_rn(f.x * 0.08838834764831845f, f.y * 0.08838834764831845f);   // the score scale folded into the query
    }
    for (int i = tid; i < kChunk * (kH / 2); i += 256) {
        const int j = i / (kH / 2), d2 = i % (kH / 2), kk = k0 + j;
        __half2 kv = __floats2half2_rn(0.f, 0.f), vv = kv;
        if (kk < n_ring) {
            const size_t slot = (size_t) ((lo0 + kk) & (kWindow - 1));
            kv = reinterpret_cast<const __half2*>(kc + slot * kKV + g * kH)[d2];
            vv = reinterpret_cast<const __half2*>(vc + slot * kKV + g * kH)[d2];
        } else if (kk < n_keys) {
            const size_t at = ((size_t) (kk - n_ring) * kKVHeads + g) * kH;
            const float2 a = reinterpret_cast<const float2*>(kb + at)[d2], b = reinterpret_cast<const float2*>(vb + at)[d2];
            kv = __floats2half2_rn(a.x, a.y);
            vv = __floats2half2_rn(b.x, b.y);
        }
        ks[j][d2] = kv;
        vs[j][d2] = vv;

    }
    __syncthreads();
    const int qi = tid / 8, sub = tid % 8;
    float m = -INFINITY, l = 0.f;
    if (qi < nq) {
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
        for (int d2 = 0; d2 < kH / 2; ++d2) {
            const __half2 a = qs[qi][d2];
#pragma unroll
            for (int j = 0; j < 4; ++j) s[j] = __builtin_amdgcn_fdot2(a, ks[sub + 8 * j][d2], s[j], false);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int kj = sub + 8 * j;
            const bool ok = k0 + kj < n_keys;
            s[j] = ok ? s[j] : -INFINITY;
            m = fmaxf(m, s[j]);
        }
        for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 8));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float e = s[j] == -INFINITY ? 0.f : __expf(s[j] - m);
            p[qi][sub + 8 * j] = e;
            l += e;
        }
        for (int o = 4; o > 0; o >>= 1) l += __shfl_xor(l, o, 8);
    }
    __syncthreads();
    if (qi >= nq) return;
    float acc[16] = {};
    for (int j = 0; j < min(kChunk, n_keys - k0); ++j) {
        const float pj = p[qi][j];
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            const float2 v = __half22float2(vs[j][sub * 8 + d]);
            acc[2 * d] += pj * v.x;
            acc[2 * d + 1] += pj * v.y;
        }
    }
    float* o = part + ((size_t) (g * gridDim.y + c) * nq + qi) * (kH + 2);
#pragma unroll
    for (int d = 0; d < 16; ++d) o[sub * 16 + d] = acc[d];
    if (sub == 0) {
        o[kH] = m;
        o[kH + 1] = l;
    }
}


// Keep the original Q6_K bytes. A lane owns one signed scale group of 16 codes.
// Each pair of 16-bit loads is safe at Q6_K's 2-byte super-block alignment.
__device__ __forceinline__ int load_q6_word(const uint8_t* p) {
    const auto* s = reinterpret_cast<const uint16_t*>(p);
    return s[0] | (uint32_t(s[1]) << 16);
}
template <int M>
__global__ void __launch_bounds__(256) q6_head_kernel(Matrix w, Quant x, Candidate* parts) {
    constexpr int L = 16, R = 4, groups = kWidth / 16;
    constexpr int cols = kWidth / 4;
    __shared__ int xs[M * cols];
    __shared__ float xd[M * (kWidth / 32)];
    const int t = threadIdx.x, lane = t % L, local = t / L * R;
    const int first = int(blockIdx.x) * kHeadCta + local;
    for (int i = t; i < M * cols; i += 256) xs[i] = reinterpret_cast<const int*>(x.q)[i];
    for (int i = t; i < M * (kWidth / 32); i += 256) xd[i] = x.d[i];
    __syncthreads();
    float acc[R][M] = {};
    for (int b = lane; b < groups; b += L) {
        const int j = lane * 16, ip = j / 128, q = j % 128 / 32, c = j % 32;
        int codes[R][4];
        float scale[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const auto* wb = w.data + size_t(first + r) * w.stride + (b / 16) * 210;
            scale[r] = half_at(wb + 208) * reinterpret_cast<const int8_t*>(wb)[192 + lane];
#pragma unroll
            for (int z = 0; z < 4; ++z) {
                const int lo = (load_q6_word(wb + ip * 64 + (q % 2) * 32 + c + z * 4) >> (4 * (q / 2))) & 0x0f0f0f0f;
                const int hi = ((load_q6_word(wb + 128 + ip * 32 + c + z * 4) >> (2 * q)) & 0x03030303) << 4;
                // Set upper bits for signed bytes, then flip the low sign bit (u-32).
                const int u = lo | hi;
                const uint32_t sign = (u ^ 0x20202020) & 0x20202020;
                codes[r][z] = (u ^ 0x20202020) | (sign << 1) | (sign << 2);
            }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const int* a = xs + m * cols + b * 4;
            const float d = xd[m * (kWidth / 32) + b / 2];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int z = __builtin_amdgcn_sdot4(codes[r][0], a[0], 0, false);
                z = __builtin_amdgcn_sdot4(codes[r][1], a[1], z, false);
                z = __builtin_amdgcn_sdot4(codes[r][2], a[2], z, false);
                z = __builtin_amdgcn_sdot4(codes[r][3], a[3], z, false);
                acc[r][m] += float(z) * (scale[r] * d);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = acc[r][m];
            for (int o = L / 2; o; o /= 2) v += __shfl_xor(v, o, L);
            if (!lane) xd[m * kHeadCta + local + r] = v;
        }
    }
    __syncthreads();
    local_top16<M, kHeadCta>(xd, parts, w.out);
}

__global__ void __launch_bounds__(kMergeCta) fast_top_merge(const Candidate* parts, DeviceResult* out) {
    __shared__ Candidate waves[kMergeCta / 64];
    const int t = threadIdx.x, row = blockIdx.x;
    const auto* p = parts + size_t(row) * kHeadLists * kTop;
    Candidate heads[kMergeLists];
    int next[kMergeLists];
#pragma unroll
    for (int j = 0; j < kMergeLists; ++j) {
        const int at = t + j * kMergeCta;
        heads[j] = at < kHeadLists ? p[at] : kInvalid;
        next[j] = 1;
    }
    for (int rank = 0; rank < kTop; ++rank) {
        Candidate best = kInvalid;
#pragma unroll
        for (int j = 0; j < kMergeLists; ++j) if (better(heads[j], best)) best = heads[j];
        best = wave_best(best);
        if (t % 64 == 0) waves[t / 64] = best;
        __syncthreads();
        best = waves[0];
        for (int j = 1; j < kMergeCta / 64; ++j) if (better(waves[j], best)) best = waves[j];
        __syncthreads();
        if (t == 0) { out->ids[row * kTop + rank] = best.id; out->unary[row * kTop + rank] = best.logit; }
#pragma unroll
        for (int j = 0; j < kMergeLists; ++j) if (heads[j].id == best.id && best.id != INT32_MAX) {
            const int at = t + j * kMergeCta;
            heads[j] = next[j] < kTop ? p[kHeadLists + size_t(at) * (kTop - 1) + next[j] - 1] : kInvalid;
            ++next[j];
        }
    }
}

template <int M, int L, int R, bool Fused = false>
void packed_launch(Packed w, Quant a, Outs out, cudaStream_t st, int batches = 1, Candidate* p = nullptr) {
    const dim3 grid(ceil_div(w.out, 256 / L * R), batches);
    if (w.q4) planar_kernel<M, L, R, true, Fused><<<grid, 256, 0, st>>>(w.codes, w.scales, a.q, a.d, a.sum, out, p, w.in, w.out);
    else planar_kernel<M, L, R, false, Fused><<<grid, 256, 0, st>>>(w.codes, w.scales, a.q, a.d, a.sum, out, p, w.in, w.out);
}

template <int M>
void packed_rows(Packed w, Quant a, Outs out, int batches, int layout, cudaStream_t st) {
    if (layout == 1) packed_launch<M, 8, 4>(w, a, out, st, batches);
    else if (layout == 2) packed_launch<M, 16, 1>(w, a, out, st, batches);
    else packed_launch<M, 8, 2>(w, a, out, st, batches);
}
void packed_product(Packed w, Quant a, Outs out, int n, int layout, cudaStream_t st) {
    const int bulk = n > kBlock ? n / kBlock * kBlock : 0;
    if (bulk) {
        packed_rows<8>(w, a, out, bulk / kBlock, layout, st);
        a = a.slice(w.in, bulk);
        for (int i = 0; i < 3; ++i) if (out.y[i]) out.y[i] += size_t(bulk) * out.n[i];
    }
    switch (n - bulk) {
#define ROWS(M) case M: packed_rows<M>(w, a, out, 1, layout, st); break
        ROWS(1); ROWS(2); ROWS(3); ROWS(4); ROWS(5); ROWS(6); ROWS(7); ROWS(8);
#undef ROWS
        default: break;
    }
}

template <int M>
void packed_head(Packed w, Matrix raw, Quant a, Candidate* parts, DeviceResult* out, cudaStream_t st) {
    if (w.codes) packed_launch<M, 8, 2, true>(w, a, {}, st, 1, parts);
    else q6_head_kernel<M><<<kHeadLists, 256, 0, st>>>(raw, a, parts);
    fast_top_merge<<<M, kMergeCta, 0, st>>>(parts, out);
}

// Requantize the Q6_K head on load, not on each ask. Q4_0 uses the signed
// abs-max convention (-8 for the extremum); Q8_0 uses a symmetric signed range.
template <bool Q4>
__global__ void requant_head(Matrix w, uint8_t* out, __half* scales) {
    const size_t b = size_t(blockIdx.x) * 8 + threadIdx.x / 32;
    const int j = threadIdx.x % 32;
    if (b >= size_t(w.in / 32) * w.out) return;
    const auto* row = w.data + (b / (w.in / 32)) * w.stride;
    const float v = value<Q6_K>(row, (b % (w.in / 32)) * 32 + j);
    float extreme = v;
    int index = j;
    for (int off = 16; off; off >>= 1) {
        const float a = __shfl_xor(extreme, off, 32);
        const int i = __shfl_xor(index, off, 32);
        if (fabsf(a) > fabsf(extreme) || (fabsf(a) == fabsf(extreme) && i < index)) { extreme = a; index = i; }
    }
    const float d = Q4 ? extreme / -8.f : fabsf(extreme) / 127.f;
    if (!j) scales[b] = __float2half(d);
    if constexpr (Q4) {
        const int c = d == 0.f ? 8 : min(15, max(0, int(v / d + 8.5f)));
        const int hi = __shfl_xor(c, 16, 32);
        if (j < 16) out[b * 16 + j] = c | (hi << 4);
    } else {
        reinterpret_cast<int8_t*>(out)[b * 32 + j] = d == 0.f ? 0 : __float2int_rn(v / d);
    }
}

// FP32-activation control for the private heads. Produce the same lists as
// the int8 kernel without requiring a second copy of the requantized weights.
template <int M, bool Q4>
__global__ void __launch_bounds__(256) float_head_kernel(Packed w, const float* x, Candidate* parts) {
    constexpr int L = 16, R = 4;
    __shared__ float tile[M][256];
    __shared__ float scores[M * kHeadCta];
    const int t = threadIdx.x, lane = t % L, local = t / L * R;
    const int first = int(blockIdx.x) * kHeadCta + local, kb = kWidth / 32;
    const auto* q = reinterpret_cast<const uint8_t*>(w.codes);
    float acc[R][M] = {};
    for (int start = 0; start < kWidth; start += 256) {
#pragma unroll
        for (int m = 0; m < M; ++m) tile[m][t] = x[m * kWidth + start + t];
        __syncthreads();
        for (int j = lane; j < 256; j += L) {
            const int c = start + j;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const size_t b = size_t(first + r) * kb + c / 32;
                const int ix = c % 32;
                int code;
                if constexpr (Q4) code = ((q[b * 16 + ix % 16] >> (4 * (ix / 16))) & 15) - 8;
                else code = reinterpret_cast<const int8_t*>(q)[b * 32 + ix];
                const float d = float(code) * __half2float(w.scales[b]);
#pragma unroll
                for (int m = 0; m < M; ++m) acc[r][m] = fmaf(d, tile[m][j], acc[r][m]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float v = acc[r][m];
            for (int off = L / 2; off; off >>= 1) v += __shfl_xor(v, off, L);
            if (!lane) scores[m * kHeadCta + local + r] = v;
        }
    }
    __syncthreads();
    local_top16<M, kHeadCta>(scores, parts, w.out);
}

template <int M>
void packed_float_head(Packed w, const float* x, Candidate* parts, DeviceResult* out, cudaStream_t st) {
    if (w.q4) float_head_kernel<M, true><<<kHeadLists, 256, 0, st>>>(w, x, parts);
    else float_head_kernel<M, false><<<kHeadLists, 256, 0, st>>>(w, x, parts);
    fast_top_merge<<<M, kMergeCta, 0, st>>>(parts, out);
}

// Weight-BF16 quality control: change only activations; keep the dense
// products in FP32 rather than pretending bf16 weights can use signed dot4.
__global__ void roundtrip_kernel(Quant a, float* out, int count) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < count) out[i] = a.q[i] * a.d[i / 32];
}
