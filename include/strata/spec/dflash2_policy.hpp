// include/strata/spec/dflash2_policy.hpp - how many DFlash2 proposals a verify window takes (STRATA_DFLASH2_T=0).
//
// The drafter is asked for a full block (7 proposals) once; the window then takes the prefix k (T = 1 + k) with the
// most expected kept tokens per millisecond:
//     E(k) = 1 + sum_{j=1..k} prod_{i<j} p_i,    score(k) = E(k) / (W(1 + k) + ask)
// p_i is the acceptance probability of proposal i given that the ones before it were accepted, learned online per
// (depth, confidence bucket); the confidence is the drafter's unary margin of the proposed token (its target-head
// logit minus the best other candidate's).  W(T) is the measured window cost (run + commit) by width, an
// exponential average with a linear prior.  Acceptance is the exact-match rule, so the choice changes speed only:
// what is emitted is the target's own token in either case.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::spec {

class Dflash2Policy {
public:
    static constexpr int kSteps = 7, kBuckets = 6, kMaxT = 8;

    Dflash2Policy() {
        static constexpr double prior[kBuckets] = {0.35, 0.55, 0.70, 0.82, 0.92, 0.97};
        for (int d = 0; d < kSteps; ++d)
            for (int b = 0; b < kBuckets; ++b) { n_[d][b] = kPriorWeight; ok_[d][b] = kPriorWeight * prior[b]; }
        for (int t = 0; t <= kMaxT; ++t) { w_[t] = prior_ms(t); seen_[t] = false; }
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

    /// W(T): the measured average where there is one, else the prior scaled like the nearest measured width
    double cost(int T) const {
        T = std::clamp(T, 1, kMaxT);
        if (seen_[T]) return w_[T];
        for (int d = 1; d <= kMaxT; ++d) {
            if (T - d >= 1 && seen_[T - d]) return w_[T - d] * prior_ms(T) / prior_ms(T - d);
            if (T + d <= kMaxT && seen_[T + d]) return w_[T + d] * prior_ms(T) / prior_ms(T + d);
        }
        return prior_ms(T);
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
        w_[T] = seen_[T] ? w_[T] + kAlpha * (ms - w_[T]) : ms;
        seen_[T] = true;
    }

private:
    static constexpr double kPriorWeight = 8.0, kDecay = 0.995, kAlpha = 0.1;
    static double prior_ms(int T) { return 21.0 + 7.6 * (T - 1); }
    double ok_[kSteps][kBuckets], n_[kSteps][kBuckets];
    double w_[kMaxT + 1];
    bool seen_[kMaxT + 1];
};

}  // namespace strata::spec
