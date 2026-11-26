// The looped Inkling-2-Mini GGUF on one HIP RDNA2 card.
#pragma once

#include "engine.hpp"
#include "inkling_dflash2.hpp"
#include "inkling_kv_helpers.hpp"
#include "tp2_comm.hpp"

#include <array>
#include <condition_variable>
#include <exception>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::poly {

class Inkling2 final : public Engine {
public:
    Inkling2();
    static std::unique_ptr<Engine> open_tp2(const Options& o);
    ~Inkling2() override;

    bool sees() const override { return true; }
#if defined(STRATA_HIP_GFX906)
    void request_budget(int64_t prompt_rows, int64_t max_new) override;
    bool kv_prefix_tags() const override { return mirror_tagging_; }
    void set_kv_prefix_tag(uint64_t tag, int64_t preserved) override;
    void on_keep_rows(int64_t end) override;
    LogitShards project_target_shards(const float* h, int rows, bool window) override;
#endif
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void begin_pass() override;
    void load_layer(int il) override;
    void run_visits(int n) override;
#if defined(STRATA_HIP_GFX906)
    bool read_custom(const int32_t* tokens, int n, const Mark* marks, int n_marks,
                     const std::function<bool(int, int)>* turn, bool& completed) override;
#endif
    void on_loaded() override;
    void on_reset() override;
    void before_state_access() override;
    bool draft_independent_of_model_stream() const override { return decode_async_keep_ && df_gpu23_; }
    void on_keep_snap(size_t snap_idx) override;
    Dev load_output_matrix(const TensorInfo* t, int64_t file_vocab) override;
    void project_logits(const float* h, float* out, int rows) override;
    void alloc_caches() override;
    void assign_scratch() override;
    // the drafter's FFN down projection takes rows of 9216 (kDsFfn), its block 8 of them
    int widest() const override { return draft_ ? std::max(2 * kWidth, kDsFfn) : 2 * kWidth; }
    int streams() const override { return 1; }
    size_t recurrent_floats() const override;
    size_t extra_bytes() const override;
    void save_card_state(uint8_t* dst) override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    bool load_card_state_fast(const uint8_t* src, int64_t pos) override;
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void on_verify_rows(int rows) override;
    void finish_logits(float* logits, int rows) override;
    void route(int il, const float* logits, int32_t, int* ids, float* w, float* tmp) const override;
#if defined(STRATA_HIP_GFX906)
    bool route_inkling_exact(int il, int n, int32_t* ids, float* w, int32_t* invalid) override;
#endif
    void moe_shared(const Layer&, int, int n) override;
    void moe_add(int n) override;
    void load_drafter() override;
    void draft_step(int32_t token, int i) override;
    void drafter_kept(int n) override;
    void drafter_guesses(int32_t token) override;

private:
    static constexpr int kPhysical = 24, kVisits = 48;
    static constexpr int kWidth = 2304, kHead = 128, kHeads = 24, kKv = 8;
    static constexpr int kQuery = kHeads * kHead, kKey = kKv * kHead, kRelative = 16;
    static constexpr int kTaps = 4, kWindow = 1024, kBlock = 6, kRank = 64;
    static constexpr int kCore = 12, kStart = 6, kCoda = 42;
    static constexpr int kConvFloats = (kTaps - 1) * (2 * kKey + 2 * kWidth);
    int tp_size_ = 1, tp_rank_ = 0;
    int first_visit_ = 0, last_visit_ = kVisits, run_last_visit_ = kVisits;
    bool prefill_only_ = false;
    int swa_ring_win_ = 0;
    int owned_full() const { return last_visit_ / kBlock - first_visit_ / kBlock; }
    int owned_swa() const { return last_visit_ - first_visit_ - owned_full(); }
    size_t conv_at(int visit) const { return (size_t) (visit - first_visit_) * conv_floats_; }
    int n_heads_ = kHeads, n_kv_ = kKv, n_query_ = kQuery, n_key_ = kKey;
    int conv_floats_ = kConvFloats;
    std::shared_ptr<Tp2Comm> tp2_;
    std::unique_ptr<Inkling2> peer_;
    enum class PeerCmd { kIdle, kRunVisits, kRunGpuVisits, kLogits, kLogitShards, kKeepSnap, kReset, kStop, kDraftHead };
    int tp2_vocab_lo_ = 0, tp2_vocab_n_ = 0;
    float *tp2_logits_half_ = nullptr, *tp2_logits_peer_ = nullptr;
    std::thread peer_thread_;
    std::mutex peer_mu_;
    std::condition_variable peer_cv_;
    std::atomic<PeerCmd> peer_cmd_{PeerCmd::kIdle};
    bool decode_async_keep_ = false, peer_pending_keep_ = false;
    int peer_arg_n_ = 0;
    size_t peer_arg_snap_ = 0;
    bool peer_done_ = true;
    std::exception_ptr peer_err_;
    std::array<std::vector<uint8_t>, kPhysical> tp2_exp_gu_, tp2_exp_down_;

    void start_peer_thread();
    void stop_peer_thread();
    void peer_dispatch(PeerCmd cmd, int n = 0, size_t snap_idx = 0);
    void peer_wait();
    void run_visits_local(int n, bool gpu_route = false);
    void layer_head_impl(Layer& L, int il, int n, bool gpu_route);
    void load_moe_tp2(int il);

#if defined(STRATA_HIP_GFX906)
    // First word is the sticky invalid flag; the rest are all visited rows' ordered picks.
    int32_t *gpu_route_dev_ = nullptr, *gpu_route_host_ = nullptr;
    int gpu_route_rows_ = 4;
    bool gpu_down_sum_ = false;
    float* grp_part_ = nullptr;     // STRATA_INKLING_MOE_GROUPED: [8n][2304] down partials
    int32_t* grp_ids_ = nullptr;    // STRATA_INKLING_MOE_GROUPED: distinct-expert table of the current visit
    float *gpu_route_bias_ = nullptr, *gpu_route_gu_ = nullptr;
    void* gpu_route_q8_ = nullptr;
    bool gpu_route_failed_ = false, gpu_prefill_route_ = false;
    bool prefill_depth_share_ = false;
    bool compact_swa_asked_ = false;
    int swa_stage_rows_ = 0;   // positive only when the stage fits in input_[0]
    inkling::kv_helpers::MirrorPtr kv_helper_;
    bool mirror_tagging_ = false;
    struct HeadPair;
    std::unique_ptr<HeadPair> head_pair_;
    static std::unique_ptr<HeadPair> init_head_pair(Inkling2& owner, int rows);
    void enqueue_head_pair(const float* h, int rows, bool window);
    void sync_head_pair();
    // Phase-Adaptive Co-Resident DualTP (STRATA_INKLING_PHASE_DUALTP=24): canonical GPU0/1 owns full
    // state, decode, DFlash2 and KV helpers; co-resident Stage B on GPU2/3 owns blocks 6..23 / visits [24,48)
    // and accelerates multi-chunk reads with staggered cut-24 packets and async KV delta return.
    bool phase_dualtp_active_ = false;
    int phase_dualtp_min_calls_ = 2;
    std::unique_ptr<Inkling2> stage_b_;
    int64_t stage_b_kv_valid_ = 0;
    cudaStream_t cut_st_ = nullptr;
    cudaEvent_t cut_ev_ = nullptr;
    Inkling2* cut_dst_ = nullptr;
    float *cut_depth0_ = nullptr, *cut_prelude_ = nullptr, *cut_bank0_ = nullptr, *cut_bank1_ = nullptr;
    cudaStream_t kv_ret_st_ = nullptr;
    cudaEvent_t kv_ret_ev_ = nullptr, inbox_free_ev_ = nullptr;
    bool stage_b_want_df2_ = false, sb_capture_df2_ = false;
    float *sb_tap0_ = nullptr, *sb_tap1_ = nullptr, *sb_tap2_ = nullptr, *sb_hn_ = nullptr;
    void phase_dualtp_init(const Options& o);
    void destroy_phase_dualtp_handles();
    void phase_dualtp_sync_to_b(int64_t pos);
    void phase_dualtp_handoff_cut(int n, bool send_df_taps, int df_first);
    void phase_dualtp_run_b(int64_t chunk_pos, int n, bool do_df_taps, int df_first, uint8_t* checkpoint,
                            size_t r0_state_n, size_t r1_state_n);
    void phase_dualtp_save_range(int first_v, int last_v, int64_t pos, uint8_t* dst,
                                 size_t r0_state_n, size_t r1_state_n);
    void phase_dualtp_sync_from_b(int64_t final_pos);
    void phase_dualtp_df_kept(int64_t chunk_pos, int m, int first);
    bool gpu_route_supported() const;
    bool gpu_decode_supported() const;
    void init_gpu_route(bool decode, bool prefill);
    void gpu_route_moe(Layer& L, int il, int n);
    void count_gpu_route(int n);
#endif

    static constexpr bool local(int visit) { return visit % kBlock != kBlock - 1; }
    static constexpr int physical(int visit) {
        return visit < kStart ? visit : visit < kCoda ? kStart + (visit - kStart) % kCore : visit - 2 * kCore;
    }

    struct Control { Dev down, gain, forget; };
    struct Block {
        Dev q, k, v, o, gate, sink, rel, table, qclip, rclip;
        Dev knorm, vnorm, kproj, vproj, kconv, vconv, aconv, fconv;
        Dev aquery, fquery, msg_query;
        std::vector<float> bias;
        std::array<Control, 4> control;
    };
    /// Split Q8_0 planes (or half values only). Each plane's stride is in bytes.
    struct Plane { uint8_t* data = nullptr; size_t stride = 0; };
    struct Cache { std::array<Plane, 4> planes; };
    std::array<Block, kPhysical> blocks_;
    std::array<Cache, kVisits> caches_;
    Dev token_norm_, output_query_, loop_query_, loop_kv_query_;
    float qk_clip_ = 0, r_clip_ = 0, alpha_ = 0, cap_ = 0, expert_scale_ = 0;
    int floor_ = 0, ring_ = 0, visit_ = 0, depth_n_ = 0, attend_tiles_ = 0;
    bool flash_ = false;    // STRATA_INKLING_FLASH=1 (load_geometry())
    bool decode_ = false;   // STRATA_INKLING_DECODE=1 (load_geometry())
    bool factored_decode_ = false;   // STRATA_INKLING_DECODE_FACTORED=1 (load_geometry())
    bool norm_q8_ = false;  // STRATA_INKLING_NORM_Q8=1 (load_geometry()): a visit's stream norms and Q8_1 together
    bool q8_cols_ = false;  // STRATA_INKLING_Q8_COLS=1 (load_geometry()): a window's Q8_0 projections in T1's order
    bool moe_tail_ = false; // STRATA_INKLING_TAIL_MOE=1 (load_geometry()): a fused visit's expert sum in the tail
    bool cpu_sum_ = false;  // moe_add() left the CPU experts' sum for layer_tail()
    bool fo_clean_ = false; // layer_tail() left fo_ 0 for the next layer's experts (moe_shared())
    bool kv_stored_ = false;   // layer_head() stored the decode row's K and V (heads_kv_kernel) for attention()
    bool heads_kv_rows_ = false;   // STRATA_INKLING_HEADS_KV_ROWS=1: fuse conv2 + head_norms + store2_q8 for windows
    bool q4_0_dense_cols_ = false; // STRATA_INKLING_Q4_0_DENSE_COLS=1: grouped multi-row Q4_0 dense projections
#if defined(STRATA_HIP_GFX906)
    bool mix_rms_fuse_ = false;
    bool prefill_fuse_ = false;
    bool moe_group_ballot_ = false;
    float* state_next_ = nullptr;
    template<bool Ffn>
    void depth_mix_norm(const float* query, const float* weight, float* y, void* q, int n);
#endif
    bool mix_ctrl_fuse_ = false;   // STRATA_INKLING_MIX_CTRL_FUSE=1: fuse depth_mix + control_in on loop visits
    bool kvproj_fuse_ = false;     // STRATA_GEMV_ROWS_SHARED_OPT=1: fuse kproj + vproj into one 2-segment GEMV
    bool df_fast_ = false;         // STRATA_DFLASH2_FAST=1: fused DFlash2 quantize/conv/swiglu and L8/R2 gemv8
    size_t score_floats_ = 0;
    void* attq_ = nullptr;  // the attention's output as Q8_1 (attend_decode()), for the output projection
    void *kq8_ = nullptr, *vq8_ = nullptr;   // K's and V's own inputs as Q8_1 (project())
    // heads(): the three streams' current rows, messages, norm pairs, ranks, normed inputs (one after the other)
    // and those as Q8_1, for decode rows
    float *s3cur_ = nullptr, *s3msg_ = nullptr, *s3pair_ = nullptr, *s3rank_ = nullptr, *s3in_ = nullptr;
    void* s3q8_ = nullptr;
    void project_cols(const Block& b, const float* xq, const float* xk, const float* xv, bool shared_kv,
                      const void* q8, const void* k8, const void* v8, int n);
    void project(const Block& b, const float* xq, const float* xk, const float* xv, bool shared_kv, const void* q8,
                 const void* k8 = nullptr, const void* v8 = nullptr);
    bool heads(const Block& b, int il, bool swa, bool loop, int n, const Dev* const* norm);
    bool fused(int n) const;   // decode_ and fewer rows than a prompt batch
    float* snap_of(float* state) const;

    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *gate_ = nullptr, *rel_ = nullptr,
          *sink_ = nullptr, *taps_ = nullptr, *scores_ = nullptr, *partial_ = nullptr, *halo_ = nullptr,
          *chunks_ = nullptr, *taps2_ = nullptr;
    std::array<float*, 3> input_{};
    std::array<float*, 4> depth_{};
    float *prelude_ = nullptr, *bank0_ = nullptr, *bank1_ = nullptr, *msg_ = nullptr, *pair_ = nullptr,
          *rank_ = nullptr, *gain_ = nullptr, *forget_ = nullptr;

    bool q8() const { return opt_.kv_format == KvFmt::kQ8_0; }
    bool messages() const { return visit_ >= kStart + kCore && visit_ < kCoda; }
    size_t kv_bytes() const;
    void conv(float* x, float* state, const float* w, const float* dynamic, int width, int n);
    void store(const float* x, Cache& c, int side, int n);
    void attention(const Block& b, int n, const float* k_in = nullptr, const float* v_in = nullptr);
#if defined(STRATA_HIP_GFX906)
    void copy_swa(const Cache& live, const Cache& stage, int ring_head, int stage_head, int n, bool gather);
    uint8_t* ckpt_dev_ = nullptr;
    uint8_t* ckpt_pin_ = nullptr;
    size_t ckpt_cap_ = 0;
    uint8_t* park_pin_[2] = {nullptr, nullptr};
    cudaEvent_t park_ev_[2] = {nullptr, nullptr};
    void ensure_ckpt_bufs(size_t bytes);
    void ensure_park_bufs();
    void free_v7r4_bufs();
    void save_rows_pinned(int64_t first, int64_t n, uint8_t* dst);
    void load_rows_pinned(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n);
#endif
    void depth_mix(const float* query, float* dst, int first, int n);
    void control(const Block& b, int il, int stream, float* in, int n, const float* shared_depth = nullptr);
    void ring_state(int visit, int64_t pos, const uint8_t* src, uint8_t* dst);
    bool try_pack_state(uint8_t* dst);

    // The DSpark drafter (--draft-model, a `dflash` GGUF with dspark_hidden_out): Engine::draft_ without a block
    // layer.  It reads the model's rows entering visits kTap (ds_taps_, every call) and its normed output (hn_ + n_embd_);
    // the kept tokens' rows go into its own K/V (ds_cache_: layers 0 and 2 rings of kWindow rows, layer 1 every row),
    // and the last one's mix for the recurrent head into hn_'s first row.
    static constexpr int kDsLayers = 3, kDsFfn = 9216, kDsRank = 256, kDsBlock = 8;
    static constexpr int kTap[3] = {5, 23, 41};
    static constexpr bool ds_swa(int l) { return l != 1; }
    struct DsLayer {
        Dev attn_norm, ffn_norm, ctx_norm, q, k, v, o, k_ctx, v_ctx, q_norm, k_norm, k_ctx_norm, gate, up, down;
    };
    std::array<DsLayer, kDsLayers> ds_;
    std::array<Cache, kDsLayers> ds_cache_;
    Dev ds_out_norm_, ds_mask_, ds_rnn_norm_, ds_joint_, ds_joint_b_;
    float ds_mix_[kDsLayers + 1][4] = {};   // softmax of dspark_fuse's rows: the layers' and the recurrent head's
    const uint8_t* ds_w1_ = nullptr;        // markov_w1: F16 or BF16 rows of kDsRank, in the mapped file
    const uint8_t* ds_w2_host_ = nullptr;   // precomputed markov_w2 [n_vocab_][kDsRank] (BF16/F16) when in GGUF
    const float* ds_b2_host_ = nullptr;     // precomputed markov_w2.bias [n_vocab_] when in GGUF
    bool ds_w1_bf16_ = false, ds_w2_bf16_ = false;
    int ds_mask_rows_ = 1;                  // 1 (shared mask_embd) or 7 (per-position mask_embd)
    std::vector<float> ds_hidden_, ds_hidden_b_;   // dspark_hidden_out [kWidth][kDsRank] and its bias, for prepare_head()
    float ds_eps_ = 0, ds_cap_ = 0, ds_theta_ = 0;
    bool ds_head_ready_ = false;
    float* ds_taps_ = nullptr;              // the rows entering visits kTap of the last call, rows_ apart
    float *ds_x_ = nullptr, *ds_xa_ = nullptr, *ds_q_ = nullptr, *ds_qr_ = nullptr, *ds_k_ = nullptr,
          *ds_kr_ = nullptr, *ds_v_ = nullptr, *ds_ao_ = nullptr, *ds_out_ = nullptr, *ds_g_ = nullptr,
          *ds_u_ = nullptr, *ds_h_ = nullptr, *ds_cur_ = nullptr, *ds_base_ = nullptr, *ds_state_ = nullptr,
          *ds_w1row_ = nullptr, *ds_jx_ = nullptr, *ds_jo_ = nullptr, *ds_lat_ = nullptr, *ds_att_ = nullptr,
          *ds_b2_ = nullptr;
    float* ds_host_ = nullptr;              // pinned: a markov_w1 row, or prepare_head()'s rows, on their way
    uint16_t* ds_w2_ = nullptr;             // W_head W_hidden_out, half [kDsRank][n_vocab_]
    int ds_rows_ = 0;                       // the guesses a block has (base rows): min(7, window - 1)
    int ds_based_ = 0;                      // the block's rows whose head logits ds_base_ holds
    void load_drafter_shape(const Shards& g);
    void prepare_head();
    void ds_block(int32_t token);
    void ds_base(int to);
    size_t ds_row_bytes() const { return 2 * (kKey + kKey / 32 * sizeof(uint16_t)); }
    void ds_ring(int l, int64_t pos, const uint8_t* src, uint8_t* dst);
    bool ds_on() const { return draft_ && !df2_; }   // the DSpark drafter (not DFlash2)

    // The DFlash2 drafter (--draft-model, a `dflash` GGUF with selector_hidden; inkling_dflash2.hpp): five non-causal
    // SWA(2048) layers with dynamic convolutions, its K/V prefix from the same taps as DSpark's (ds_taps_ and
    // hn_ + n_embd_) concatenated through fc and hidden_norm, the block [anchor, MASK x 7] in one pass, the target's
    // head on rows 1..7 and the candidate selector's walk (drafter_guesses()).  Its rings are not part of a saved
    // state: a restored or reset position starts it without context (df_lo_ = df_hi_ = pos).
    static constexpr int kDfLayers = 5, kDfFfn = 6912, kDfSub = 256, kDfMask = 200056, kDfVocab = 200058;
    struct DfLayer {
        Dev attn_norm, ffn_norm, q, k, v, o, q_norm, k_norm, gate, up, down, abase, aproj, fbase, fproj;
        // Planar Q8/Q4 weights; q|k|v and gate|up fused (one launch each).
        dflash2::Mat qkv, gu, wo, wdown, waproj, wfproj;
    };
    bool df_planar_ = false;           // GEMV8 or MIXED: use packed weights in the drafter block
    bool df_kept_opt_ = false;         // KEPT_OPT: quantize a short kept batch once and reuse the QKV slices
    bool df_q4_f32_ = false;           // Q4_F32: expand backbone Q4_0 scales for comparison
    bool df_attend3_ = false;          // STRATA_DFLASH2_ATTEND3=1: dflash2::attend3()
    dflash2::Mat df_sel_;
    // Private planar head weights on both ranks. Rank 1's top 16 cross as 7 x 32 floats.
    bool df_private_head_ = false;    // HEAD8, HEAD_Q4 or HEAD_FUSE on TP2
    bool df_head_q4_ = false, df_head_fuse_ = false;
    dflash2::Mat df_head_;
    int df_head_lo_ = 0;
    void* df_head_parts_ = nullptr;
    float *df_hlog_ = nullptr, *df_htop_ = nullptr, *df_hvals_ = nullptr, *df_hpack_ = nullptr, *df_hx_ = nullptr;
    int32_t *df_hids_ = nullptr, *df_hids1_ = nullptr;
    float* df_hvals1_ = nullptr;
    float* df_hxd_ = nullptr;
    int32_t* df_hxsum_ = nullptr;
    int8_t* df_hxq_ = nullptr;
    void df_head_load(int lo, int n, Inkling2* gpu23_owner = nullptr);
    void df_head_run(const float* h, cudaStream_t stream);   // h: [7][2304] on this card; leaves df_hids_/df_hvals_ (rank-local ids)
    int8_t* df_xq_ = nullptr;          // planar activations: [8][9216] codes
    float* df_xd_ = nullptr;           // [8][288] scales
    int32_t* df_xsum_ = nullptr;       // [8][288] unscaled code sums (Q4 only)
    dflash2::Mat df_pack_rows(std::initializer_list<Dev> weights);
    void df_lin(const Dev& w, const dflash2::Mat& packed, const float* x, float* y, int rows);
    bool df2_ = false;
    std::array<DfLayer, kDfLayers> df_;
    Dev df_fc_, df_hnorm_, df_onorm_, df_selw_;
    dflash2::Book df_pred_, df_succ_;   // selector codebooks [kDfVocab][256]
    __half *df_kc_ = nullptr, *df_vc_ = nullptr;       // [kDfLayers][2048][1024] K (normed, rotated) and V rings
    float df_eps_ = 0, df_cap_ = 0, df_theta_ = 0;
    int64_t df_lo_ = 0, df_hi_ = 0;   // ring rows of positions [df_lo_, df_hi_) are this sequence's
    int64_t df_skip_ = 0;             // a prompt's rows before this position are never attended: not injected
    float *df_x_ = nullptr, *df_h_ = nullptr, *df_hp_ = nullptr, *df_dyn_ = nullptr, *df_q_ = nullptr,
          *df_qr_ = nullptr, *df_k_ = nullptr, *df_kr_ = nullptr, *df_v_ = nullptr, *df_ao_ = nullptr,
          *df_o_ = nullptr, *df_g_ = nullptr, *df_u_ = nullptr, *df_m_ = nullptr, *df_fin_ = nullptr,
          *df_logits_ = nullptr, *df_selh_ = nullptr, *df_vals_ = nullptr, *df_edges_ = nullptr, *df_att_ = nullptr,
          *df_top_ = nullptr, *df_ctx_ = nullptr, *df_c_ = nullptr, *df_cn_ = nullptr, *df_ck_ = nullptr,
          *df_ckr_ = nullptr, *df_cv_ = nullptr, *df_t_ = nullptr;
    int32_t* df_ids_ = nullptr;
    void df_shape(const Shards& g);
    void df_load();
    void df_load_weights();
    void df_scratch();
    void df_linear(const Dev& w, const float* x, float* y, int n);
    void df_context(const float* taps, int64_t pos, int n, bool provisional, cudaEvent_t taps_read = nullptr);
    void df_ask(int32_t token);
    struct DfGpu23;
    std::unique_ptr<DfGpu23> df_gpu23_;
    Dev df_token_norm_;
    void df_gpu23_init();
    void df_gpu23_scratch();
    void df_taps_push(int first, int n, bool provisional);
    void df_gpu23_head(bool reuse_selector_quant);
    cudaStream_t df_stream() const;
    void df_kept(int n);
};

}  // namespace strata::poly
