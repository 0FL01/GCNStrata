// include/strata/spec/dflash2_policy.hpp - how many DFlash2 proposals a verify window takes (STRATA_DFLASH2_T=0).
//
// The drafter is asked for a full block (7 proposals) once; the window then takes the prefix k (T = 1 + k) with the
// most expected kept tokens per millisecond:
//     E(k) = 1 + sum_{j=1..k} prod_{i<j} p_i,    score(k) = E(k) / (W(1 + k) + ask)
// p_i is the acceptance probability of proposal i given that the ones before it were accepted, learned online per
// (depth, confidence bucket); the confidence is the drafter's unary margin of the proposed token (its target-head
// logit minus the best other candidate's).  W(T) is the measured window cost (run + commit), a weighted line in T
// through the recent windows of every width, with a prior line.  Acceptance is the exact-match rule, so the choice changes speed only:
// what is emitted is the target's own token in either case.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace strata::spec {

class Dflash2Policy {
public:
    static constexpr int kSteps = 7, kBuckets = 6, kMaxT = 8;

    Dflash2Policy() {
        static constexpr double prior[kBuckets] = {0.35, 0.55, 0.70, 0.82, 0.92, 0.97};
        for (int d = 0; d < kSteps; ++d)
            for (int b = 0; b < kBuckets; ++b) { n_[d][b] = kPriorWeight; ok_[d][b] = kPriorWeight * prior[b]; }
        // the prior line: two pseudo-windows of weight kPriorLine at T = 1 and T = 8
        for (const int t : {1, kMaxT}) {
            const double x = t - 1, y = prior_ms(t);
            p_[0] += kPriorLine; p_[1] += kPriorLine * x; p_[2] += kPriorLine * y; p_[3] += kPriorLine * x * x; p_[4] += kPriorLine * x * y;
        }
    }

    static int bucket(float margin) {
        if (!(margin >= 0.f)) return 0;   // (NaN: least confident)
        if (margin < 1.f) return 1;
        if (margin < 2.f) return 2;
        if (margin < 4.f) return 3;
        if (margin < 8.f) return 4;
        return 5;
    }
    double p(int depth, float margin) const {
        const int b = bucket(margin);
        return ok_[depth][b] / n_[depth][b];
    }

    /// W(T) = a + b (T - 1): an exponentially weighted least-squares line through the measured windows (all widths
    /// share it, so a width the policy rarely picks is still priced from the others), with a prior line as weight
    double cost(int T) const {
        T = std::clamp(T, 1, kMaxT);
        // the decayed sums of the measured windows plus a prior line of constant weight (it never decays away, so a
        // run of one width alone cannot leave the slope undetermined)
        const double S = s_ + p_[0], X = sx_ + p_[1], Y = sy_ + p_[2], XX = sxx_ + p_[3], XY = sxy_ + p_[4];
        const double det = S * XX - X * X;
        const double b = det > 1e-9 ? (S * XY - X * Y) / det : kPriorB;
        const double a = (Y - b * X) / S;
        return a + std::max(b, 0.5) * (T - 1);
    }
    /// the learned acceptance by depth (rows) and bucket (columns), for the log
    void table(char* out, size_t n) const {
        size_t at = 0;
        for (int d = 0; d < kSteps && at < n; ++d) {
            at += (size_t) std::snprintf(out + at, n - at, d ? " |" : "");
            for (int b = 0; b < kBuckets && at < n; ++b) at += (size_t) std::snprintf(out + at, n - at, " %.2f", ok_[d][b] / n_[d][b]);
        }
    }
    /// the number of proposals (0..n) to verify; margins[i] for i < n
    int choose(const float* margins, int n, double ask_ms) const {
        n = std::min(n, kSteps);
        double e = 1.0, surv = 1.0, best = e / (cost(1) + ask_ms);
        int k = 0;
        for (int j = 1; j <= n; ++j) {
            surv *= p(j - 1, margins[j - 1]);
            e += surv;
            const double s = e / (cost(1 + j) + ask_ms);
            if (s > best) { best = s; k = j; }
        }
        return k;
    }

    /// a window that verified k proposals and accepted the first `accepted` of them
    void observe_accept(const float* margins, int k, int accepted) {
        for (int d = 0; d < std::min(k, kSteps) && d <= accepted; ++d) {
            const int b = bucket(margins[d]);
            n_[d][b] = n_[d][b] * kDecay + 1.0;
            ok_[d][b] = ok_[d][b] * kDecay + (d < accepted ? 1.0 : 0.0);
        }
    }
    void observe_cost(int T, double ms) {
        if (T < 1 || T > kMaxT || !(ms > 0.0)) return;
        if (ms > 2.5 * cost(T)) return;   // a width's first window (graph capture), a stall: not its price
        const double x = T - 1;
        s_ = s_ * kDecay + 1; sx_ = sx_ * kDecay + x; sy_ = sy_ * kDecay + ms; sxx_ = sxx_ * kDecay + x * x; sxy_ = sxy_ * kDecay + x * ms;
    }

private:
    static constexpr double kPriorWeight = 8.0, kPriorLine = 2.0, kDecay = 0.995;
    static constexpr double kPriorB = 7.6;
    static double prior_ms(int T) { return 21.0 + kPriorB * (T - 1); }
    double ok_[kSteps][kBuckets], n_[kSteps][kBuckets];
    double s_ = 0, sx_ = 0, sy_ = 0, sxx_ = 0, sxy_ = 0;   // the decayed regression sums of (T - 1, ms)
    double p_[5] = {};                                     // the prior line's sums
};

}  // namespace strata::spec
